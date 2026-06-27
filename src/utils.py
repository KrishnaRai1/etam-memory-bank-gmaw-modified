# Shared utilities: frame listing, mask geometry,
# track/overlay saving, EfficientTAM package path resolution,
# and benchmark reporting tools.
import os
import re
import json
import time
import csv
from collections import Counter
from pathlib import Path
import numpy as np
from importlib.resources import files as pkg_files
from PIL import Image, ImageDraw, ImageFont


# -------- frames --------
def load_frame_names(video_dir: str):
    root = Path(video_dir).expanduser()
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths.extend(root.rglob(ext))
    frame_names = [str(p.relative_to(root)) for p in image_paths if p.is_file()]

    def _key(p: str):
        m = re.search(r"\d+", p)
        return int(m.group()) if m else p
    frame_names.sort(key=_key)
    return frame_names

def ensure_run_dir(output_root: str) -> Path:
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
    try:
        return str(pkg_files(package).joinpath(rel))
    except Exception:
        return str((Path(fallback) / rel).resolve())

def centroid_from_coords(coords):
    ys, xs = coords
    if xs.size == 0: return None
    return float(xs.mean()), float(ys.mean())

def save_segonly_by_frame_json(seg_by_frame: dict[int, list[tuple[np.ndarray, np.ndarray]]], path: Path):
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
    base = Image.open(frame_path).convert("RGBA")
    W, H = base.size

    overlay_arr = np.zeros((H, W, 4), dtype=np.uint8)

    if extra_masks:
        seg_col = np.array([0, 180, 255, int(max(0.0, min(1.0, extra_alpha)) * 255)], dtype=np.uint8)
        for ys, xs in extra_masks:
            if ys.size and xs.size:
                ys_clamped = np.clip(ys, 0, H - 1)
                xs_clamped = np.clip(xs, 0, W - 1)
                overlay_arr[ys_clamped, xs_clamped] = seg_col

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

# -------- Benchmark Evaluation Tools --------
def export_benchmark_summary(log_data_list: list[dict], out_dir: Path):
    """Generates an aggregated Markdown report and CSV for benchmark runs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / "benchmark_summary.csv"
    headers = [
        "interval_id", "category", "frame_count", "droplet_count", 
        "manual_count", "count_delta", "total_runtime", "stage3_runtime"
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for log in log_data_list:
            man_count = log.get("benchmark_manual_count", 0)
            sys_count = log.get("droplet_count", 0)
            
            writer.writerow({
                "interval_id": log.get("benchmark_interval"),
                "category": log.get("benchmark_category"),
                "frame_count": log.get("frame_count"),
                "droplet_count": sys_count,
                "manual_count": man_count,
                "count_delta": abs(man_count - sys_count) if man_count != -1 else "N/A",
                "total_runtime": round(log.get("total_runtime", 0), 2),
                "stage3_runtime": round(log.get("stage3_runtime", 0), 2)
            })

    md_path = out_dir / "benchmark_summary.md"
    categories = Counter([log.get("benchmark_category") for log in log_data_list])
    
    with open(md_path, 'w') as f:
        f.write("# Benchmark Evaluation Summary\n\n")
        f.write("## Overview\n")
        f.write(f"- Total Intervals Processed: {len(log_data_list)}\n")
        for cat, count in categories.items():
            f.write(f"  - **{cat}**: {count}\n")
            
        f.write("\n## Detailed Results\n")
        f.write("| Interval ID | Category | Frames | Sys Count | Manual Count | Stage 3 Time (s) |\n")
        f.write("|-------------|----------|--------|-----------|--------------|------------------|\n")
        for log in log_data_list:
            man_count = log.get("benchmark_manual_count", -1)
            man_str = str(man_count) if man_count != -1 else "N/A"
            f.write(f"| {log.get('benchmark_interval')} | {log.get('benchmark_category')} | {log.get('frame_count')} | "
                    f"{log.get('droplet_count')} | {man_str} | {round(log.get('stage3_runtime', 0), 2)} |\n")

def visualize_id_switches(
    frames_dir: Path,
    frame_names: list[str],
    T: dict[int, dict[int, tuple]],
    out_dir: Path,
    highlight_ids: list[int] = None
):
    """Renders frames highlighting specific IDs. Useful for debugging ID switch errors."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for fi, fname in enumerate(frame_names):
        frame_path = frames_dir / fname
        if not frame_path.exists(): 
            continue
        
        active_items = []
        for oid, frames in T.items():
            if highlight_ids and oid not in highlight_ids:
                continue
            if fi in frames:
                active_items.append((frames[fi], oid))
                
        if not active_items: 
            continue 
            
        out_path = out_dir / f"switch_debug_{fi:06d}.jpg"
        save_overlay_with_ids(
            frame_path=frame_path,
            items=active_items,
            out_path=out_path,
            alpha=0.6,
        )