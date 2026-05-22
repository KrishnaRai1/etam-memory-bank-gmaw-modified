# Model wrappers:
#   YOLODetector: per-frame detection (Stage 1 seed)
#   EfficientTAMTracker: segmentation + temporal propagation with bbox prompts
import numpy as np
import torch
from ultralytics import YOLO
from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor
import os
# Disable torch.compile / Triton / Inductor: they conflict with multiprocessing
# and don't help throughput in this pipeline.
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("PYTORCH_DISABLE_TRITON", "1")

class YOLODetector:
    # Thin wrapper over Ultralytics YOLO for per-frame detection.
    def __init__(self, weights: str, conf: float = 0.35, iou: float = 0.5,
                 target_cls: int | None = None, target_classes: list[int] | None = None):
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        # Back-compat: target_cls (single class) is still accepted.
        if target_classes is not None:
            self.target_classes = set(int(c) for c in target_classes)
        elif target_cls is not None:
            self.target_classes = {int(target_cls)}
        else:
            self.target_classes = None  # no filter -> return every class

    def detect_xyxy(self, img_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Returns (boxes Nx4, classes N, confs N), filtered by target_classes if set.
        res = self.model(img_path, conf=self.conf, iou=self.iou, verbose=False)[0]
        if res.boxes is None:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32),
                    np.zeros((0,), dtype=np.float32))
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
        cls   = res.boxes.cls.cpu().numpy().astype(np.int32)
        conf  = res.boxes.conf.cpu().numpy().astype(np.float32)
        keep = conf >= self.conf
        if self.target_classes is not None:
            keep &= np.isin(cls, list(self.target_classes))
        return boxes[keep], cls[keep], conf[keep]

    def detect_by_class(self, img_path: str) -> dict[int, np.ndarray]:
        # {cls_id -> boxes[N,4]} for the (filtered) classes present in this frame.
        boxes, cls, _ = self.detect_xyxy(img_path)
        out: dict[int, list[np.ndarray]] = {}
        for b, c in zip(boxes, cls):
            out.setdefault(int(c), []).append(b.astype(np.float32))
        return {k: (np.stack(v, axis=0) if len(v) else np.zeros((0,4), np.float32))
                for k, v in out.items()}


def _pick_device(pref: str):
    if pref == "cpu":  return torch.device("cpu")
    if pref == "cuda": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EfficientTAMTracker:
    def __init__(self, cfg_path: str, ckpt_path: str, device: str = "auto"):
        self.device = _pick_device(device)

        # Global autocast via __enter__() breaks under multiprocessing —
        # we apply it locally inside propagate() instead.
        self.predictor = build_efficienttam_video_predictor(cfg_path, ckpt_path, device=self.device)

        # eval + freeze keeps any graph from attempting a backward pass (inference only).
        try:
            self.predictor.eval()
            for p in self.predictor.parameters():
                p.requires_grad_(False)
        except Exception:
            pass

        self.state = None

    def segment_boxes_at_frame(self, frame_idx: int, boxes_xyxy: np.ndarray):
        # Segment all bboxes at a single frame: seed and consume one step.
        # Used by Stage 1 (independent per-frame mask).
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return []

        assert self.state is not None, "Call init(video_dir) before segment_boxes_at_frame"
        self.reset()

        # Temporary IDs outside the range used by Stage 3.
        first_id = 100000
        for j, b in enumerate(boxes_xyxy):
            self.seed_box(frame_idx, first_id + j, b)

        masks = [None] * len(boxes_xyxy)

        import torch
        # no_grad (not inference_mode): inference_mode marks tensors and breaks
        # downstream predictor ops.
        with torch.no_grad():
            for f_idx, obj_ids, logits in self.propagate(
                start_frame_idx=frame_idx,
                max_frame_num_to_track=0,          # stay on the seed frame
                reverse=False
            ):
                if f_idx != frame_idx:
                    continue
                id_to_k = {oid: k for k, oid in enumerate(obj_ids)}
                for j in range(len(boxes_xyxy)):
                    k = id_to_k.get(first_id + j, None)
                    if k is not None:
                        m = (logits[k] > 0).detach().cpu().numpy().squeeze().astype(bool)
                        masks[j] = m
                break

        return [m for m in masks if m is not None]

    def init(self, video_dir: str):
        self.state = self.predictor.init_state(video_path=video_dir)

    def reset(self):
        self.predictor.reset_state(self.state)

    def seed_box(self, frame_idx: int, obj_id: int, box_xyxy):
        return self.predictor.add_new_points_or_box(
            inference_state=self.state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=np.array(box_xyxy, dtype=np.float32)
        )

    def propagate(self, **kwargs):
        # bfloat16 + inference_mode: better throughput and lower VRAM during inference.
        import torch
        dev = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.autocast(dev, dtype=torch.bfloat16), torch.inference_mode():
            for f, ids, logits in self.predictor.propagate_in_video(self.state, **kwargs):
                yield int(f), [int(x) for x in ids], logits


    def track_window(self, frame_idx: int, boxes_xyxy: np.ndarray, w: int):
        # Propagate w-1 frames backward and w-1 frames forward for each box.
        # Used by the bidirectional Stage 2 "propagate" variant (paper).
        # Returns: {tmp_id -> {frame -> (ys, xs)}}
        self.reset()
        out = {}

        # Normalize boxes
        if boxes_xyxy is None:
            return out
        boxes = np.asarray(boxes_xyxy, dtype=np.float32)
        if boxes.ndim == 1:
            if boxes.size == 0:
                return out
            boxes = boxes.reshape(1, -1)
        if boxes.size == 0 or boxes.shape[0] == 0:
            return out

        base = 10_000
        for j in range(boxes.shape[0]):
            oid = base + j
            self.seed_box(frame_idx, oid, boxes[j])
            out[oid] = {}

        # backward
        for f, ids, logits in self.propagate(start_frame_idx=frame_idx,
                                            max_frame_num_to_track=w-1, reverse=True):
            for k, oid in enumerate(ids):
                m = (logits[k] > 0).detach().cpu().numpy().squeeze().astype(bool)
                out[oid][int(f)] = np.nonzero(m)

        # forward
        for f, ids, logits in self.propagate(start_frame_idx=frame_idx,
                                            max_frame_num_to_track=w-1, reverse=False):
            for k, oid in enumerate(ids):
                m = (logits[k] > 0).detach().cpu().numpy().squeeze().astype(bool)
                out[oid][int(f)] = np.nonzero(m)

        return out

    def track_masklets_for_seeds(self, seeds_by_frame: dict[int, np.ndarray], w: int | None = None):
        # Track masklets for seeds scattered across multiple frames.
        # If w is given, drop predictions outside [seed-(w-1), seed+(w-1)].
        self.reset()
        gid_map = {}
        start_oid = 200_000  # outside the range used by Stage 1/3
        for f_seed, boxes in sorted(seeds_by_frame.items()):
            for k_local, b in enumerate(boxes):
                oid = start_oid; start_oid += 1
                self.seed_box(f_seed, oid, b)
                gid_map[oid] = (f_seed, k_local)

        out = {}
        dev = 'cuda' if self.device.type == 'cuda' else 'cpu'
        import torch
        # no_grad (not inference_mode) to avoid the "inference tensors" error.
        with torch.autocast(dev, dtype=torch.bfloat16), torch.no_grad():
            for f_idx, obj_ids, logits in self.propagate():
                for j, oid in enumerate(obj_ids):
                    fk = gid_map.get(oid, None)
                    if fk is None:
                        continue
                    f_seed, k_local = fk
                    if (w is not None) and (abs(f_idx - f_seed) > (w - 1)):
                        continue
                    m_bin = (logits[j] > 0).detach().cpu().numpy().squeeze().astype(bool)
                    out.setdefault((f_seed, k_local), {})[int(f_idx)] = m_bin
        return out
