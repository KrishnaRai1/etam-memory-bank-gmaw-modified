"""
Tool: check_frame_alignment.py
Verifies frame alignment between reference masks and frame images by sampling frames,
overlaying reference droplet masks, and saving a JSON report with centroid/area info.

Usage examples:
  python tools/check_frame_alignment.py --reference-dir /path/to/ALS29T7_data/VID_data --frames-dir /path/to/VID_frames --video-id VID --samples 20 --outdir alignment_check

"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import pandas as pd


def find_frame_file(frames_dir: Path, abs_frame: int) -> Path | None:
    patterns = [f"*{abs_frame}*.jpg", f"*{abs_frame}*.png", f"*{abs_frame}*.jpeg"]
    for patt in patterns:
        matches = list(frames_dir.glob(patt))
        if matches:
            return matches[0]
    # try zero-padded variants
    for pad in (6, 5, 4, 3):
        name = str(abs_frame).zfill(pad)
        for ext in ('.jpg', '.png', '.jpeg'):
            p = frames_dir / f"{name}{ext}"
            if p.exists():
                return p
    return None


def px_to_mask(px_list: list[int], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not px_list:
        return mask
    arr = np.asarray(px_list, dtype=np.int32)
    ys = (arr // width).astype(int)
    xs = (arr % width).astype(int)
    valid = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
    mask[ys[valid], xs[valid]] = 255
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True, help="Path to reference *_data directory containing tracks_clean.parquet")
    parser.add_argument("--frames-dir", required=True, help="Path to video frames folder")
    parser.add_argument("--video-id", default=None, help="Video id for labeling")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--outdir", default="alignment_check")
    args = parser.parse_args()

    reference_dir = Path(args.reference_dir)
    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks_path = reference_dir / "tracks_clean.parquet"
    if not tracks_path.exists():
        print(f"[ERROR] tracks_clean.parquet not found at {tracks_path}")
        return

    tracks = pd.read_parquet(tracks_path)
    # select droplet class (supervisor says class_id==3)
    if "class_id" in tracks.columns:
        droplets = tracks[tracks["class_id"] == 3]
    else:
        droplets = tracks

    if droplets.empty:
        print("[WARN] No droplet rows (class_id==3) in tracks_clean.parquet")
        return

    frames = sorted(droplets["abs_frame"].unique().tolist())
    samples = min(args.samples, len(frames))
    chosen = random.sample(frames, samples)

    report: list[dict[str, Any]] = []

    for frame_idx in chosen:
        rows = droplets[droplets["abs_frame"] == frame_idx]
        frame_file = find_frame_file(frames_dir, int(frame_idx))
        if frame_file is None:
            print(f"[WARN] Frame image for abs_frame={frame_idx} not found in {frames_dir}")
            continue
        img = Image.open(frame_file).convert("RGB")
        width, height = img.size
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for _, r in rows.iterrows():
            px = r.get("mask_px")
            if not px:
                continue
            mask = px_to_mask(px, width=width, height=height)
            ys, xs = np.where(mask > 0)
            if ys.size == 0:
                continue
            area = int(ys.size)
            centroid_x = float(xs.mean())
            centroid_y = float(ys.mean())
            # draw contour-ish overlay by marking centroid and bounding box
            minx, maxx = int(xs.min()), int(xs.max())
            miny, maxy = int(ys.min()), int(ys.max())
            draw.rectangle([(minx, miny), (maxx, maxy)], outline=(255, 0, 0, 160))
            draw.ellipse([(centroid_x - 3, centroid_y - 3), (centroid_x + 3, centroid_y + 3)], fill=(255, 0, 0, 200))
            report.append({
                "frame_index": int(frame_idx),
                "mask_area": area,
                "centroid": [centroid_x, centroid_y],
                "frame_image": str(frame_file.resolve()),
            })
        out_path = out_dir / f"overlay_{frame_idx}.png"
        combined = Image.alpha_composite(img.convert("RGBA"), overlay)
        combined.save(out_path)

    report_path = out_dir / "alignment_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"video_id": args.video_id, "samples": len(report), "entries": report}, f, indent=2)

    print(f"Wrote alignment overlays to {out_dir} and report {report_path}")


if __name__ == "__main__":
    main()
