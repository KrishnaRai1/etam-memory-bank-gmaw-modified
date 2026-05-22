# Shared utilities: frame listing, mask geometry,
# track/overlay saving, and EfficientTAM package path resolution.
import os, re, json, time
from pathlib import Path
import numpy as np
from importlib.resources import files as pkg_files
from PIL import Image, ImageDraw, ImageFont


# -------- frames --------
def load_frame_names(video_dir: str):
    # .jpg/.jpeg only. Sort by the number embedded in the filename (e.g. 000123.jpg).
    names = [p for p in os.listdir(video_dir) if os.path.splitext(p)[-1].lower() in (".jpg", ".jpeg")]
    def _key(p: str):
        m = re.search(r"\d+", p)
        return int(m.group()) if m else p
    names.sort(key=_key)
    return names

def ensure_run_dir(output_root: str) -> Path:
    # Timestamped subfolder per run so we don't clobber previous runs.
    run_dir = Path(output_root) / time.strftime("%Y-%m-%d_%H-%M-%S")
    (run_dir / "overlays").mkdir(parents=True, exist_ok=True)
    (run_dir / "masks").mkdir(parents=True, exist_ok=True)
    return run_dir

# -------- geometry --------
def iou_xyxy(a, b) -> float:
    xa0, ya0, xa1, ya1 = a
    xb0, yb0, xb1, yb1 = b
    iw = max(0.0, min(xa1, xb1) - max(xa0, xb0))
    ih = max(0.0, min(ya1, yb1) - max(ya0, yb0))
    inter = iw * ih
    ua = max(0.0, xa1 - xa0) * max(0.0, ya1 - ya0)
    ub = max(0.0, xb1 - xb0) * max(0.0, yb1 - yb0)
    return float(inter / max(ua + ub - inter, 1e-6))

def bbox_from_mask_bool(mask2d: np.ndarray):
    ys, xs = np.nonzero(mask2d)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def iou_masks(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2).sum(dtype=np.float64)
    union = np.logical_or(m1, m2).sum(dtype=np.float64)
    return float(inter / max(union, 1.0))


# -------- saving --------
def _to_list(a):
    # Coerce to a plain list (accepts ndarray, scalar, list/tuple).
    if a is None:
        return []
    if isinstance(a, (list, tuple)):
        return list(a)
    try:
        return a.tolist()
    except AttributeError:
        return [float(a)] if np.isscalar(a) else list(a)

def save_T_json(T: dict, path: Path):
    out = {}
    for oid, frames in T.items():
        out[str(oid)] = {}
        for fi, (ys, xs) in frames.items():
            out[str(oid)][str(fi)] = [_to_list(ys), _to_list(xs)]
    with open(path, "w") as f:
        json.dump(out, f)

def save_trajectory_csv(T: dict, path: Path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx","obj_id","centroid_x","centroid_y","area_px"])
        for oid, frames in T.items():
            for fi,(ys,xs) in frames.items():
                xs = np.array(xs); ys = np.array(ys)
                if xs.size:
                    w.writerow([fi, oid, float(xs.mean()), float(ys.mean()), int(xs.size)])
                else:
                    w.writerow([fi, oid, np.nan, np.nan, 0])

def save_overlay_union(frame_path: Path, masks_bin: list[np.ndarray], out_path: Path, alpha=0.45):
    img = Image.open(frame_path).convert("RGBA")
    H, W = img.size[1], img.size[0]
    overlay = Image.new("RGBA", (W, H), (0,0,0,0))
    arr = np.zeros((H, W, 4), dtype=np.uint8)
    if masks_bin:
        union = np.zeros((H, W), dtype=bool)
        for m in masks_bin:
            union |= m.astype(bool)
        arr[union] = np.array([0, 180, 255, int(alpha*255)], dtype=np.uint8)
    overlay = Image.fromarray(arr, mode="RGBA")
    out = Image.alpha_composite(img, overlay)
    out.save(out_path)

# -------- EfficientTAM package path resolution --------
def pkg_path(package: str, rel: str, fallback: str) -> str:
    # Works whether EfficientTAM is installed as a package
    # or sits as a local folder in the repo.
    try:
        return str(pkg_files(package).joinpath(rel))
    except Exception:
        return str((Path(fallback) / rel).resolve())


def centroid_from_coords(coords):
    ys, xs = coords
    if xs.size == 0: return None
    return float(xs.mean()), float(ys.mean())

def save_segonly_by_frame_json(seg_by_frame: dict[int, list[tuple[np.ndarray, np.ndarray]]], path: Path):
    # Segment-only masks (classes we don't track) in sparse form:
    #   {"<frame>": [[ys, xs], [ys, xs], ...]}
    out = {}
    for fi, masks in seg_by_frame.items():
        out[str(fi)] = []
        for (ys, xs) in masks:
            out[str(fi)].append([_to_list(ys), _to_list(xs)])
    with open(path, "w") as f:
        json.dump(out, f)

def save_overlay_with_ids(
    frame_path: Path,
    items: list[tuple[tuple[np.ndarray, np.ndarray], int]],
    out_path: Path,
    alpha: float = 0.45,
    extra_masks: list[tuple[np.ndarray, np.ndarray]] | None = None,
    extra_alpha: float = 0.30,
):
    # Two layers: extra_masks (no IDs, behind) and tracked items with
    # per-ID color plus a label at the centroid on top.
    base = Image.open(frame_path).convert("RGBA")
    W, H = base.size

    overlay_arr = np.zeros((H, W, 4), dtype=np.uint8)

    # segment-only layer (drawn first, behind tracked items)
    if extra_masks:
        seg_col = np.array([0, 180, 255, int(max(0.0, min(1.0, extra_alpha)) * 255)], dtype=np.uint8)
        for ys, xs in extra_masks:
            if ys.size and xs.size:
                ys_clamped = np.clip(ys, 0, H - 1)
                xs_clamped = np.clip(xs, 0, W - 1)
                overlay_arr[ys_clamped, xs_clamped] = seg_col

    # deterministic color per ID
    def _color_for(oid: int) -> np.ndarray:
        r = (37 * oid) % 255
        g = (91 * oid) % 255
        b = (173 * oid) % 255
        a = int(max(0.0, min(1.0, alpha)) * 255)
        return np.array([r, g, b, a], dtype=np.uint8)

    for coords, oid in items:
        ys, xs = coords
        if ys.size and xs.size:
            ys_clamped = np.clip(ys, 0, H - 1)
            xs_clamped = np.clip(xs, 0, W - 1)
            overlay_arr[ys_clamped, xs_clamped] = _color_for(int(oid))

    overlay = Image.fromarray(overlay_arr, mode="RGBA")
    comp = Image.alpha_composite(base, overlay).convert("RGB")

    # IDs at the centroid, with a white stroke for contrast
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(comp)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for coords, oid in items:
        ys, xs = coords
        if xs.size:
            cx, cy = int(xs.mean()), int(ys.mean())
            draw.text(
                (cx, cy),
                str(oid),
                fill=(0, 0, 0),
                font=font,
                stroke_width=2,
                stroke_fill=(255, 255, 255),
            )

    comp.save(out_path)
