# Core of the paper's 3 stages:
#   Stage 1: per-frame YOLO + EfficientTAM with bbox prompts (per-frame mask).
#   Stage 2: temporal filter (IoU against a window of w frames).
#   Stage 3: long-term tracking + new-object discovery (Optimized).
# Outputs are written as parquet (tracks.parquet, segonly.parquet) + frames_meta.json.
from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .models import YOLODetector, EfficientTAMTracker
from .utils import (
    load_frame_names,
    ensure_run_dir,
    iou_masks,
    bbox_from_mask_bool,
    save_T_json,
    save_trajectory_csv,
)

# ----------------------------
# Type aliases
# ----------------------------
MaskCoords = Tuple[np.ndarray, np.ndarray]   # (ys, xs)
PerFrameMasks = Dict[int, Dict[str, dict]]   # {f: {"mask_ids": {local_id: (ys,xs)}, "boxes": np.ndarray[N,4]}}


# ----------------------------
# Local helpers
# ----------------------------
def _downsample_names(names: List[str], factor: int) -> List[str]:
    if factor <= 1:
        return names
    idx = np.round(np.linspace(0, len(names) - 1, math.ceil(len(names) / factor))).astype(int)
    return [names[i] for i in idx]

def _mask_area(coords: MaskCoords) -> int:
    return int(coords[0].size)

def _coords_to_bool(coords: MaskCoords, H: int, W: int) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    ys, xs = coords
    if xs.size:
        m[ys, xs] = True
    return m

def _union_bool(masks: List[np.ndarray], H: int, W: int) -> np.ndarray:
    u = np.zeros((H, W), dtype=bool)
    for m in masks:
        if m is None:
            continue
        u |= m
    return u

def _count_in_frame(T: Dict[int, Dict[int, MaskCoords]], f: int) -> int:
    return sum(1 for _, frames in T.items() if f in frames)

def _coords_to_flat_indices(coords: MaskCoords, W: int) -> List[int]:
    ys, xs = coords
    if xs.size == 0:
        return []
    return (ys.astype(np.int64) * int(W) + xs.astype(np.int64)).astype(np.int32).tolist()

def _get_gpu_memory_info() -> dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }

def _format_dt(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d_%H-%M-%S")

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _save_experiment_log(log_dir: Path, log_data: dict) -> Path:
    _ensure_dir(log_dir)
    filename = f"experiment_{_format_dt(time.time())}.json"
    filepath = log_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    return filepath

def _load_last_experiment_log(log_dir: Path) -> dict | None:
    if not log_dir.exists() or not log_dir.is_dir():
        return None
    logs = sorted(log_dir.glob("experiment_*.json"), reverse=True)
    if not logs:
        return None
    try:
        with open(logs[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ----------------------------
# Stage 1: per-frame YOLO detection + EfficientTAM segmentation, multi-class.
# ----------------------------
def stage1_detect_and_segment(
    cfg: dict,
    video_dir: str,
    frame_names: List[str],
    H: int,
    W: int,
    etam: EfficientTAMTracker | None = None,
    offset: int = 0,
    classes_to_segment: List[int] | None = None,
):
    yolo_cfg = cfg["yolo"]
    det = YOLODetector(
        weights=yolo_cfg["weights"],
        conf=float(yolo_cfg.get("conf", 0.25)),
        iou=float(yolo_cfg.get("iou", 0.45)),
        target_classes=classes_to_segment,
    )

    tracker = etam or EfficientTAMTracker(
        cfg_path=cfg["efficienttam"]["cfg"],
        ckpt_path=cfg["efficienttam"]["ckpt"],
        device=cfg["run"].get("device", "auto"),
    )
    if etam is None:
        tracker.init(video_dir)

    boxes_by_f_by_cls: Dict[int, Dict[int, np.ndarray]] = {}
    indep_masks_by_f_by_cls: Dict[int, Dict[int, dict]] = {}

    t0 = time.time()
    for f_idx, name in enumerate(frame_names):
        img_path = str(Path(video_dir) / name)
        per_cls_boxes = det.detect_by_class(img_path)
        boxes_by_f_by_cls[f_idx] = {}
        indep_masks_by_f_by_cls[f_idx] = {}

        all_cls = set(per_cls_boxes.keys())
        if classes_to_segment:
            all_cls |= set(int(c) for c in classes_to_segment)
        for cls_id in sorted(all_cls):
            boxes_c = per_cls_boxes.get(int(cls_id), np.zeros((0, 4), dtype=np.float32))
            boxes_by_f_by_cls[f_idx][int(cls_id)] = boxes_c
            indep_masks_by_f_by_cls[f_idx][int(cls_id)] = {"mask_ids": {}, "boxes": boxes_c}

        owners: List[tuple[int, int]] = []
        merged = []
        for cls_id in sorted(all_cls):
            boxes_c = boxes_by_f_by_cls[f_idx][int(cls_id)]
            for j in range(len(boxes_c)):
                owners.append((int(cls_id), j))
            if len(boxes_c):
                merged.append(boxes_c)

        if not merged:
            continue

        merged_boxes = np.concatenate(merged, axis=0)
        abs_idx = offset + f_idx
        masks_bool = tracker.segment_boxes_at_frame(abs_idx, merged_boxes)

        for k, (cls_id, j_local) in enumerate(owners):
            m = masks_bool[k]
            if m is None:
                continue
            ys, xs = np.nonzero(m)
            indep_masks_by_f_by_cls[f_idx][cls_id]["mask_ids"][j_local + 1] = (ys, xs)

    return boxes_by_f_by_cls, indep_masks_by_f_by_cls, (time.time() - t0)


# ----------------------------
# Stage 2 (variant A): forward-only IoU.
# ----------------------------
def stage2_temporal_filter_forward_iou(
    cfg: dict,
    boxes_by_f: Dict[int, np.ndarray],
    indep_masks: PerFrameMasks,
    H: int,
    W: int,
):
    s2 = cfg["stage2"]
    window = int(s2.get("window", 5))
    iou_thr = float(s2.get("match_iou", 0.5))

    t0 = time.time()
    N = len(boxes_by_f)

    bools_by_f: Dict[int, List[np.ndarray]] = {}
    order_ids_by_f: Dict[int, List[int]] = {}
    for f in range(N):
        ms, lids = [], []
        for lid in sorted(indep_masks[f]["mask_ids"].keys()):
            coords = indep_masks[f]["mask_ids"][lid]
            m = np.zeros((H, W), dtype=bool)
            if coords[0].size:
                m[coords] = True
            ms.append(m)
            lids.append(lid)
        bools_by_f[f] = ms
        order_ids_by_f[f] = lids

    filt_boxes: Dict[int, np.ndarray] = {}
    filt_masks: PerFrameMasks = {}

    for i in range(N):
        boxes = boxes_by_f[i]
        if boxes is None or len(boxes) == 0:
            filt_boxes[i] = np.zeros((0, 4), dtype=np.float32)
            filt_masks[i] = {"mask_ids": {}, "boxes": filt_boxes[i]}
            continue

        cur_ms = bools_by_f[i]
        cur_lids = order_ids_by_f[i]

        avail = max(0, min(window - 1, (N - 1) - i))
        if avail == 0:
            filt_boxes[i] = np.zeros((0, 4), dtype=np.float32)
            filt_masks[i] = {"mask_ids": {}, "boxes": filt_boxes[i]}
            continue

        need_hits = max(1, int(math.ceil(0.5 * avail)))
        keep_idx = []

        for j, m0 in enumerate(cur_ms):
            hits = 0
            cur_m = m0
            for step, f in enumerate(range(i + 1, i + 1 + avail), start=1):
                best_iou = 0.0
                best_m = None
                for m in bools_by_f[f]:
                    iou = iou_masks(cur_m, m)
                    if iou > best_iou:
                        best_iou = iou
                        best_m = m
                if best_iou >= iou_thr:
                    hits += 1
                    cur_m = best_m
                if hits + (avail - step) < need_hits:
                    break

            if hits >= need_hits:
                keep_idx.append(j)

        kept = boxes[keep_idx, :] if keep_idx else np.zeros((0, 4), dtype=np.float32)
        mask_ids_new = {}
        for new_k, j in enumerate(keep_idx, start=1):
            mask_ids_new[new_k] = indep_masks[i]["mask_ids"][cur_lids[j]]

        filt_boxes[i] = kept
        filt_masks[i] = {"mask_ids": mask_ids_new, "boxes": kept}

    return filt_boxes, filt_masks, time.time() - t0


# ----------------------------
# Stage 2 (variant B): paper-style bidirectional propagate + IoU.
# ----------------------------
def stage2_temporal_filter(
    cfg: dict,
    etam: EfficientTAMTracker,
    boxes_by_f: Dict[int, np.ndarray],
    indep_masks: PerFrameMasks,
    H: int,
    W: int,
):
    w = int(cfg["stage2"]["window"])
    iou_thr = float(cfg["stage2"].get("match_iou", 0.5))
    if w <= 1:
        return boxes_by_f, indep_masks, 0.0

    t0 = time.time()
    N = len(boxes_by_f)
    filt_boxes, filt_masks = {}, {}

    for i in range(N):
        boxes = boxes_by_f[i]
        if boxes is None or len(boxes) == 0:
            filt_boxes[i] = np.zeros((0, 4), dtype=np.float32)
            filt_masks[i] = {"mask_ids": {}, "boxes": filt_boxes[i]}
            continue

        window_tracks = etam.track_window(i, boxes, w) 

        keep_idx = []
        for j, _b in enumerate(boxes):
            tmp_oid = 10_000 + j
            frames_match = {i: True}
            for f, coords in window_tracks.get(tmp_oid, {}).items():
                ok = False
                m_tr = np.zeros((H, W), dtype=bool)
                m_tr[coords] = True
                for _, coords_ind in indep_masks[f]["mask_ids"].items():
                    m_ind = np.zeros((H, W), dtype=bool)
                    m_ind[coords_ind] = True
                    if iou_masks(m_tr, m_ind) >= iou_thr:
                        ok = True
                        break
                frames_match[f] = ok

            consec = 1
            fcur = i - 1
            while fcur >= max(0, i - (w - 1)) and frames_match.get(fcur, False):
                consec += 1
                fcur -= 1
            fcur = i + 1
            while fcur <= min(N - 1, i + (w - 1)) and frames_match.get(fcur, False):
                consec += 1
                fcur += 1

            if consec >= w:
                keep_idx.append(j)

        kept = boxes[keep_idx, :] if keep_idx else np.zeros((0, 4), dtype=np.float32)
        filt_boxes[i] = kept
        mask_ids_new = {k + 1: indep_masks[i]["mask_ids"][keep_idx[k] + 1] for k in range(len(keep_idx))}
        filt_masks[i] = {"mask_ids": mask_ids_new, "boxes": kept}

    return filt_boxes, filt_masks, time.time() - t0


# ----------------------------
# Stage 3: long-term tracking + new-object discovery.
# (Optimized with Fast BBox Evaluation & Lazy Union Evaluation)
# ----------------------------
def _get_new_boxes_in_frame(
    f: int,
    indep_masks: PerFrameMasks,
    T: Dict[int, Dict[int, MaskCoords]],
    H: int,
    W: int,
    min_px: int,
    overlap_thr: float = 0.0,
) -> List[np.ndarray]:
    """Candidate discovery with a fast bbox prefilter, optional overlap threshold, and cached intersections."""
    
    # STAGE3 OPTIMIZATION: Early pruning. If YOLO didn't detect any independent 
    # masks on this frame, we absolutely cannot discover a new object here.
    if not indep_masks[f].get("mask_ids"):
        return []

    tracked_bboxes: List[tuple[int, int, int, int]] = []
    tracked_here: List[MaskCoords] = []
    
    # STAGE3 OPTIMIZATION: Fast Bbox extraction before expensive boolean operations
    for _oid, frames in T.items():
        coords = frames.get(f)
        if coords is None:
            continue
        ys, xs = coords
        if xs.size == 0:
            continue
        tracked_here.append((ys, xs))
        tracked_bboxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))

    def _bbox_intersects(box1, box2):
        x0a, y0a, x1a, y1a = box1
        x0b, y0b, x1b, y1b = box2
        return not (x1a < x0b or x1b < x0a or y1a < y0b or y1b < y0a)

    # STAGE3 OPTIMIZATION: Lazy union construction. Only build the dense union 
    # mask if a bounding-box collision actually happens.
    union_tracked = None  

    new_boxes: List[np.ndarray] = []
    for _local_id, coords in indep_masks[f]["mask_ids"].items():
        if _mask_area(coords) < min_px:
            continue

        ys, xs = coords
        if xs.size == 0:
            continue
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        if tracked_bboxes:
            # Fast pass: Bounding Box intersection
            if not any(_bbox_intersects(bbox, tb) for tb in tracked_bboxes):
                new_boxes.append(np.array([bbox[0], bbox[1], bbox[2], bbox[3]], dtype=np.float32))
                continue

            # Slow pass: Pixel-level overlap evaluation (Cached dense arrays)
            if union_tracked is None:
                # Build only when absolutely necessary
                bool_masks = [_coords_to_bool(c, H, W) for c in tracked_here]
                union_tracked = _union_bool(bool_masks, H, W) if bool_masks else np.zeros((H,W), dtype=bool)

            m = _coords_to_bool(coords, H, W)
            overlap_px = int((union_tracked & m).sum())
            
            if overlap_px > 0:
                if overlap_thr <= 0.0 or (overlap_px / xs.size >= overlap_thr):
                    continue # Tracked object overlaps too much, ignore as new seed

            xyxy = bbox_from_mask_bool(m)
            if xyxy is not None:
                new_boxes.append(np.array(xyxy, dtype=np.float32))
        else:
            m = _coords_to_bool(coords, H, W)
            xyxy = bbox_from_mask_bool(m)
            if xyxy is not None:
                new_boxes.append(np.array(xyxy, dtype=np.float32))

    return new_boxes


def stage3_track_and_discover(
    cfg: dict,
    etam: EfficientTAMTracker,
    frame_names: List[str],
    boxes_by_f: Dict[int, np.ndarray],
    indep_masks: PerFrameMasks,
    H: int,
    W: int,
    offset: int = 0,
):
    batch_sz = int(cfg["yolo"].get("batch_init", 999999))

    cfg_stage3 = cfg.get("stage3", {})

    min_px = int(cfg_stage3.get("min_px", 0))
    kill_gap = int(cfg_stage3.get("kill_after_gap", 0))
    overlap_thr = float(cfg_stage3.get("overlap_thr", 0.0))
    max_skip = int(cfg_stage3.get("max_skip", 0))

    profiling = bool(cfg_stage3.get("enable_profiling", True))
    progress = bool(cfg_stage3.get("progress", True))
    reuse_outputs = bool(cfg_stage3.get("reuse_existing_outputs", True))

    memory_update_skip = cfg.get("stage2", {}).get(
        "memory_update_skip",
        1
    )

    print("\n========== MEMORY CONFIG DEBUG ==========")
    print(f"memory_update_skip = {memory_update_skip}")
    print("=========================================\n")

    log_dir = Path(cfg_stage3.get("experiment_log_dir", "experiment_logs"))

    if cfg.get("data", {}).get("output_root"):
        log_dir = Path(cfg["data"]["output_root"]) / log_dir

    T: Dict[int, Dict[int, MaskCoords]] = {}

    N = len(frame_names)

    i0 = 0

    for j in range(N):
        if boxes_by_f.get(j) is not None and len(boxes_by_f[j]) > 0:
            i0 = j
            break

    alive: Dict[int, Dict[str, int | bool]] = {}

    def _register_oids(oids, f_seed_local):
        for oid in oids:
            alive[oid] = {
                "start": int(f_seed_local),
                "gap": 0,
                "dead": False,
            }

    prop_perf = {
        "propagation_calls": 0,
        "propagated_frames": 0,
        "computed_frames": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "objects_processed": 0,
    }

    discovery_stats = {
        "candidate_frames": 0,
        "candidates_seen": 0,
        "new_seeds": 0,
        "frames_skipped_by_max_skip": 0,
    }

    def _consume_propagate(start_frame_local: int):

        prop_perf["propagation_calls"] += 1

        for f_abs, ids, logits in etam.propagate(
            perf_stats=prop_perf,
            start_frame_idx=offset + start_frame_local,
            max_frame_num_to_track=(N - 1 - start_frame_local),
            reverse=False,
            show_progress=progress,
            memory_update_skip=memory_update_skip,
        ):

            f_abs = int(f_abs)

            f_loc = f_abs - offset

            if f_loc < 0 or f_loc >= N:
                continue

            prop_perf["propagated_frames"] += 1

            for k, oid in enumerate(ids):

                oid = int(oid)

                info = alive.get(oid, None)

                if (info is None) or info["dead"]:
                    continue

                m = (
                    (logits[k] > 0)
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                    .astype(bool)
                )

                area_px = int(m.sum())

                if area_px < min_px:

                    info["gap"] = int(info["gap"]) + 1

                    if kill_gap and info["gap"] >= kill_gap:
                        info["dead"] = True

                    continue

                info["gap"] = 0

                coords = np.nonzero(m)

                if oid not in T:
                    T[oid] = {}

                T[oid][f_loc] = coords

    t0 = time.time()

    first_batch = True
    has_state = False

    boxes0 = boxes_by_f.get(i0, np.zeros((0, 4), dtype=np.float32))

    oid_base = 0

    print("\n[Stage3] Starting track-and-discover\n")

    if len(boxes0) > 0:

        for start in tqdm(
            range(0, len(boxes0), batch_sz),
            desc="[Stage3] initial seeds",
            disable=not progress,
        ):

            if not has_state or not reuse_outputs:
                etam.reset()
                has_state = True

            sl = slice(start, min(start + batch_sz, len(boxes0)))

            seeded_oids = []

            for j, b in enumerate(boxes0[sl], start=1):

                oid = oid_base + j

                etam.seed_box(offset + i0, oid, b)

                seeded_oids.append(oid)

            _register_oids(seeded_oids, i0)

            _consume_propagate(i0)

            oid_base = max(oid_base, max(seeded_oids))

        oid_base = max(T.keys()) if T else 0

    for i in tqdm(
        range(i0, N - 1),
        desc="[Stage3] discovery",
        disable=not progress,
    ):

        f_next = i + 1

        if max_skip and ((i - i0) % (max_skip + 1) != 0):

            discovery_stats["frames_skipped_by_max_skip"] += 1

            continue

        discovery_stats["candidate_frames"] += 1

        new_boxes = _get_new_boxes_in_frame(
            f_next,
            indep_masks,
            T,
            H,
            W,
            min_px,
            overlap_thr=overlap_thr,
        )

        discovery_stats["candidates_seen"] += len(new_boxes)

        if not new_boxes:
            continue

        print(
            f"[Stage3] Frame {f_next}: "
            f"discovered {len(new_boxes)} new candidates"
        )

        for start in range(0, len(new_boxes), batch_sz):

            if not has_state or not reuse_outputs:
                etam.reset()
                has_state = True

            sl = slice(start, min(start + batch_sz, len(new_boxes)))

            seeded_oids = []

            for j, b in enumerate(new_boxes[sl], start=1):

                oid = oid_base + j

                etam.seed_box(offset + f_next, oid, b)

                seeded_oids.append(oid)

            _register_oids(seeded_oids, f_next)

            discovery_stats["new_seeds"] += len(seeded_oids)

            _consume_propagate(f_next)

            oid_base += (sl.stop - sl.start)

    t3 = time.time() - t0

    print("\n========== STAGE3 PROFILE ==========")

    print(f"Propagation calls : {prop_perf['propagation_calls']}")
    print(f"Propagated frames : {prop_perf['propagated_frames']}")
    print(f"Computed frames   : {prop_perf['computed_frames']}")
    print(f"Cache hits        : {prop_perf['cache_hits']}")
    print(f"Cache misses      : {prop_perf['cache_misses']}")
    print(f"Objects processed : {prop_perf['objects_processed']}")
    print(f"New seeds         : {discovery_stats['new_seeds']}")

    print("====================================\n")
    
    return T, t3, prop_perf, discovery_stats


# ----------------------------
# Orchestrator: wires Stages 1-3 together and writes parquet outputs.
# ----------------------------
def run_pipeline(
    cfg: dict,
    frame_start: int | None = None,
    frame_end: int | None = None,
    force_run_dir: Path | None = None,
) -> Path:
    video_dir = Path(cfg["data"]["video_dir"]).resolve()
    output_root = Path(cfg["data"]["output_root"]).resolve()

    if force_run_dir is not None:
        run_dir = Path(force_run_dir)
        (run_dir / "masks").mkdir(parents=True, exist_ok=True)
    else:
        run_dir = ensure_run_dir(str(output_root))

    ds_factor = int(cfg["run"].get("downsample", 1))

    yolo_cfg = cfg["yolo"]
    classes_cfg = (yolo_cfg.get("classes") or {})
    track_classes = list(map(int, classes_cfg.get("track", []))) if isinstance(classes_cfg, dict) else []
    segment_only_classes = list(map(int, classes_cfg.get("segment_only", []))) if isinstance(classes_cfg, dict) else []

    if not track_classes and "classes" not in yolo_cfg:
        track_classes = [int(yolo_cfg.get("cls_id", 0))]
        segment_only_classes = []

    classes_to_segment = sorted(set(track_classes) | set(segment_only_classes))

    frame_names_all = load_frame_names(str(video_dir))
    assert len(frame_names_all) > 0, f"No frames (.jpg) found in {video_dir}"

    frame_names_ds = frame_names_all if ds_factor <= 1 else _downsample_names(frame_names_all, ds_factor)
    total_len = len(frame_names_ds)
    fs = 0 if frame_start is None else int(frame_start)
    fe = (total_len - 1) if frame_end is None else int(frame_end)
    frame_names = frame_names_ds[fs: fe + 1]
    offset = fs

    W, H = Image.open(video_dir / frame_names[0]).size

    etam = EfficientTAMTracker(
        cfg_path=cfg["efficienttam"]["cfg"],
        ckpt_path=cfg["efficienttam"]["ckpt"],
        device=cfg["run"].get("device", "auto"),
    )
    etam.init(str(video_dir))

    boxes_by_f_by_cls, indep_by_f_by_cls, t1 = stage1_detect_and_segment(
        cfg, str(video_dir), frame_names, H, W, etam=etam, offset=offset, classes_to_segment=classes_to_segment
    )

    segonly_coords: Dict[int, List[MaskCoords]] = {}
    if len(segment_only_classes) > 0:
        for fi in range(len(frame_names)):
            per_frame = indep_by_f_by_cls.get(fi, {})
            lst: List[MaskCoords] = []
            for c in segment_only_classes:
                entry = per_frame.get(int(c))
                if not entry:
                    continue
                for _, coords in (entry.get("mask_ids") or {}).items():
                    lst.append(coords)
            if lst:
                segonly_coords[fi] = lst

    with open(run_dir / "segonly_by_frame.json", "w") as f:
        json.dump({str(fi): [[ys.tolist(), xs.tolist()] for (ys, xs) in lst] for fi, lst in segonly_coords.items()}, f, indent=2)

    stage1_only_flag = bool(cfg.get("run", {}).get("stage1_only", False))
    if stage1_only_flag or (len(track_classes) == 0):
        frames_meta = {
            "image_size": {"W": int(W), "H": int(H)},
            "offset": int(offset),
            "downsample": int(ds_factor),
            "frame_names": frame_names,
            "classes": {"track": track_classes, "segment_only": segment_only_classes},
        }
        with open(run_dir / "frames_meta.json", "w") as f:
            json.dump(frames_meta, f, indent=2)
        return run_dir

    track_cls = int(track_classes[0])

    boxes_by_f: Dict[int, np.ndarray] = {
        fi: (boxes_by_f_by_cls.get(fi, {}).get(track_cls, np.zeros((0, 4), np.float32)))
        for fi in range(len(frame_names))
    }
    indep_masks: PerFrameMasks = {}
    for fi in range(len(frame_names)):
        e = indep_by_f_by_cls.get(fi, {}).get(track_cls, {"mask_ids": {}, "boxes": np.zeros((0, 4), np.float32)})
        indep_masks[fi] = {"mask_ids": dict(e["mask_ids"]), "boxes": e["boxes"]}

    stage2_mode = str(cfg["stage2"].get("mode", "propagate")).lower()
    if stage2_mode == "forward_iou":
        boxes_filt, masks_filt, t2 = stage2_temporal_filter_forward_iou(cfg, boxes_by_f, indep_masks, H, W)
    else:
        boxes_filt, masks_filt, t2 = stage2_temporal_filter(cfg, etam, boxes_by_f, indep_masks, H, W)

    T, t3, prop_perf, discovery_stats = stage3_track_and_discover(cfg, etam, frame_names, boxes_filt, masks_filt, H, W, offset=offset)

    # BENCHMARK FRAMEWORK: Inject benchmark metadata if present in config
    benchmark_meta = cfg.get("benchmark_meta", {})
    experiment_log_dir = Path(cfg.get("stage3", {}).get("experiment_log_dir", "experiment_logs"))
    
    full_run_log = {
        "stage1_runtime": t1,
        "stage2_runtime": t2,
        "stage3_runtime": t3,
        "total_runtime": t1 + t2 + t3,
        "memory_update_skip":
            cfg.get("stage2", {}).get(
                    "memory_update_skip",
                    1
                ),
        "runtime_per_frame":
            (t1 + t2 + t3) / max(len(frame_names), 1),
        "stage3_propagation_calls":
            prop_perf["propagation_calls"],
        "stage3_propagated_frames":
            prop_perf["propagated_frames"],
        "stage3_computed_frames":
            prop_perf["computed_frames"],
        "stage3_cache_hits":
            prop_perf["cache_hits"],
        "stage3_cache_misses":
            prop_perf["cache_misses"],
        "stage3_objects_processed":
            prop_perf["objects_processed"],
        "stage3_new_seeds":
            discovery_stats["new_seeds"],
        "droplet_count": len(T),
        "frame_count": len(frame_names),
        "offset": int(offset),
        "device": str(cfg["run"].get("device", "auto")),
        "benchmark_mode": bool(benchmark_meta),
        "benchmark_category": benchmark_meta.get("category", "N/A"),
        "benchmark_interval": benchmark_meta.get("interval_id", "N/A"),
        "benchmark_manual_count": benchmark_meta.get("manual_count", -1),
        "gpu": _get_gpu_memory_info(),
        "config": {
            "stage2_mode": cfg["stage2"].get("mode", "forward_iou"),
            "stage3": cfg.get("stage3", {}),
            "yolo": {"conf": cfg["yolo"].get("conf"), "iou": cfg["yolo"].get("iou")},
        },
    }
    _save_experiment_log(
            experiment_log_dir,
            full_run_log
        )

    frames_meta = {
        "image_size": {"W": int(W), "H": int(H)},
        "offset": int(offset),
        "downsample": int(ds_factor),
        "frame_names": frame_names,
        "classes": {"track": track_classes, "segment_only": segment_only_classes},
    }
    with open(run_dir / "frames_meta.json", "w") as f:
        json.dump(frames_meta, f, indent=2)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as e:
        raise RuntimeError(
            "pyarrow is required to write Parquet outputs. Install with: pip install pyarrow"
        ) from e

    track_rows = {
        "abs_frame": [], "rel_frame": [], "global_id": [], "class_id": [],
        "area_px": [], "centroid_x": [], "centroid_y": [],
        "bbox_x0": [], "bbox_y0": [], "bbox_x1": [], "bbox_y1": [],
        "mask_px": [],
    }

    N = len(frame_names)
    for oid, frames in T.items():
        for fi, coords in frames.items():
            ys, xs = coords
            if xs.size == 0:
                continue
            area = int(xs.size)
            cx = float(xs.mean())
            cy = float(ys.mean())
            bbox = bbox_from_mask_bool(_coords_to_bool((ys, xs), H, W))
            if bbox is None:
                bbox = [float(cx), float(cy), float(cx), float(cy)]
            px_flat = _coords_to_flat_indices(coords, W)

            track_rows["abs_frame"].append(int(offset + fi))
            track_rows["rel_frame"].append(int(fi))
            track_rows["global_id"].append(int(oid))
            track_rows["class_id"].append(int(track_cls))
            track_rows["area_px"].append(area)
            track_rows["centroid_x"].append(cx)
            track_rows["centroid_y"].append(cy)
            track_rows["bbox_x0"].append(float(bbox[0]))
            track_rows["bbox_y0"].append(float(bbox[1]))
            track_rows["bbox_x1"].append(float(bbox[2]))
            track_rows["bbox_y1"].append(float(bbox[3]))
            track_rows["mask_px"].append(px_flat)

    if len(track_rows["abs_frame"]) > 0:
        schema = pa.schema([
            pa.field("abs_frame", pa.int32()), pa.field("rel_frame", pa.int32()),
            pa.field("global_id", pa.int32()), pa.field("class_id", pa.int16()),
            pa.field("area_px", pa.int32()), pa.field("centroid_x", pa.float32()), pa.field("centroid_y", pa.float32()),
            pa.field("bbox_x0", pa.float32()), pa.field("bbox_y0", pa.float32()),
            pa.field("bbox_x1", pa.float32()), pa.field("bbox_y1", pa.float32()),
            pa.field("mask_px", pa.list_(pa.int32())),
        ])
        table = pa.Table.from_pydict(track_rows, schema=schema)
        pq.write_table(table, run_dir / "tracks.parquet")

    seg_rows = {
        "abs_frame": [], "rel_frame": [], "local_id": [], "class_id": [],
        "area_px": [], "centroid_x": [], "centroid_y": [],
        "bbox_x0": [], "bbox_y0": [], "bbox_x1": [], "bbox_y1": [],
        "mask_px": [],
    }

    if len(segment_only_classes) > 0:
        for fi in range(N):
            local_counter = 0
            for cls_id in segment_only_classes:
                entry = indep_by_f_by_cls.get(fi, {}).get(int(cls_id), None)
                if not entry:
                    continue
                for _, coords in sorted(entry.get("mask_ids", {}).items(), key=lambda kv: kv[0]):
                    ys, xs = coords
                    if xs.size == 0:
                        continue
                    area = int(xs.size)
                    cx = float(xs.mean())
                    cy = float(ys.mean())
                    bbox = bbox_from_mask_bool(_coords_to_bool((ys, xs), H, W))
                    if bbox is None:
                        bbox = [float(cx), float(cy), float(cx), float(cy)]
                    px_flat = _coords_to_flat_indices(coords, W)

                    seg_rows["abs_frame"].append(int(offset + fi))
                    seg_rows["rel_frame"].append(int(fi))
                    seg_rows["local_id"].append(int(local_counter))
                    seg_rows["class_id"].append(int(cls_id))
                    seg_rows["area_px"].append(area)
                    seg_rows["centroid_x"].append(cx)
                    seg_rows["centroid_y"].append(cy)
                    seg_rows["bbox_x0"].append(float(bbox[0]))
                    seg_rows["bbox_y0"].append(float(bbox[1]))
                    seg_rows["bbox_x1"].append(float(bbox[2]))
                    seg_rows["bbox_y1"].append(float(bbox[3]))
                    seg_rows["mask_px"].append(px_flat)
                    local_counter += 1

        if len(seg_rows["abs_frame"]) > 0:
            seg_schema = pa.schema([
                pa.field("abs_frame", pa.int32()), pa.field("rel_frame", pa.int32()),
                pa.field("local_id", pa.int32()), pa.field("class_id", pa.int16()),
                pa.field("area_px", pa.int32()), pa.field("centroid_x", pa.float32()), pa.field("centroid_y", pa.float32()),
                pa.field("bbox_x0", pa.float32()), pa.field("bbox_y0", pa.float32()),
                pa.field("bbox_x1", pa.float32()), pa.field("bbox_y1", pa.float32()),
                pa.field("mask_px", pa.list_(pa.int32())),
            ])
            seg_table = pa.Table.from_pydict(seg_rows, schema=seg_schema)
            pq.write_table(seg_table, run_dir / "segonly.parquet")

    save_T_json(T, run_dir / "T.json")
    save_trajectory_csv(T, run_dir / "trajectories.csv")

    return run_dir