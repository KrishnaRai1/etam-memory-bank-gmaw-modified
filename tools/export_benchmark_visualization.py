#!/usr/bin/env python3
"""Generate benchmark visualizations by overlaying reference and predicted masks with IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _load_parquet_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Unable to load parquet file {path}: {exc}") from exc


def _infer_frame_column(df: pd.DataFrame) -> str:
    for candidate in ("frame_idx", "abs_frame", "rel_frame", "frame"):
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Dataframe is missing a frame index column. Found: {list(df.columns)}")


def _mask_to_coords(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if "ys" in row.index and "xs" in row.index and row["ys"] is not None and row["xs"] is not None:
        ys = np.asarray(row["ys"]).astype(int)
        xs = np.asarray(row["xs"]).astype(int)
        return ys, xs
    if "mask_px" in row.index and row["mask_px"] is not None and "width" in row.index and row["width"] is not None:
        px = np.asarray(row["mask_px"]).astype(int)
        width = int(row["width"])
        ys = (px // width).astype(int)
        xs = (px % width).astype(int)
        return ys, xs
    raise ValueError("Row missing mask coordinates or width metadata.")


def _load_masks(path: Path, add_width: int | None = None) -> pd.DataFrame:
    df = _load_parquet_table(path)
    if add_width is not None and "width" not in df.columns:
        df["width"] = add_width
    return df


def _frame_image_paths(images_dir: Path) -> list[Path]:
    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not image_paths:
        raise FileNotFoundError(f"No image files found in {images_dir}")
    return image_paths


def _draw_mask(draw_arr: np.ndarray, ys: np.ndarray, xs: np.ndarray, color: tuple[int, int, int, int]):
    ys = np.clip(ys, 0, draw_arr.shape[0] - 1)
    xs = np.clip(xs, 0, draw_arr.shape[1] - 1)
    draw_arr[ys, xs] = color


def _draw_label(draw, x: int, y: int, text: str, color: tuple[int, int, int]):
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((x, y), text, fill=color, font=font)


def render_visualization(
    images_dir: Path,
    reference_dir: Path,
    predicted_dir: Path,
    output_dir: Path,
    mp4_path: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = _frame_image_paths(images_dir)

    ref_masks = _load_masks(reference_dir / "seg_masks.parquet")
    pred_tracks = _load_masks(predicted_dir / "tracks.parquet")
    width = None
    if not pred_tracks.empty and "mask_px" in pred_tracks.columns:
        first_image = Image.open(image_paths[0])
        width = first_image.width
        pred_tracks["width"] = width

    frame_col_ref = _infer_frame_column(ref_masks)
    frame_col_pred = _infer_frame_column(pred_tracks)
    ref_by_frame = {int(k): g for k, g in ref_masks.groupby(frame_col_ref)}
    pred_by_frame = {int(k): g for k, g in pred_tracks.groupby(frame_col_pred)}

    frame_files: list[Path] = []
    for frame_idx, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        overlay = np.zeros((H, W, 4), dtype=np.uint8)

        ref_rows = ref_by_frame.get(frame_idx, pd.DataFrame())
        for _, row in ref_rows.iterrows():
            ys, xs = _mask_to_coords(row)
            _draw_mask(overlay, ys, xs, (0, 255, 0, 120))

        pred_rows = pred_by_frame.get(frame_idx, pd.DataFrame())
        for _, row in pred_rows.iterrows():
            ys, xs = _mask_to_coords(row)
            _draw_mask(overlay, ys, xs, (255, 0, 0, 140))

        comp = Image.alpha_composite(image.convert("RGBA"), Image.fromarray(overlay))
        draw = ImageDraw.Draw(comp)
        _draw_label(draw, 6, 6, f"frame {frame_idx}", (255, 255, 255))
        _draw_label(draw, 6, 24, f"ref={len(ref_rows)} pred={len(pred_rows)}", (255, 255, 0))

        out_path = output_dir / f"vis_{frame_idx:05d}.png"
        comp.convert("RGB").save(out_path)
        frame_files.append(out_path)

    if mp4_path is not None:
        with iio.get_writer(str(mp4_path), fps=10) as writer:
            for path in frame_files:
                writer.append_data(np.asarray(Image.open(path).convert("RGB")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export benchmark visualization overlays for reference and predicted masks.")
    parser.add_argument("--images-dir", required=True, help="Directory containing frames for the interval")
    parser.add_argument("--reference-dir", required=True, help="Directory containing reference seg_masks.parquet")
    parser.add_argument("--predicted-dir", required=True, help="Pipeline run directory containing tracks.parquet")
    parser.add_argument("--out-dir", default="outputs/benchmark_visualizations", help="Directory to write visualization images")
    parser.add_argument("--mp4", default=None, help="Optional output MP4 path")
    args = parser.parse_args()

    render_visualization(
        Path(args.images_dir).expanduser().resolve(),
        Path(args.reference_dir).expanduser().resolve(),
        Path(args.predicted_dir).expanduser().resolve(),
        Path(args.out_dir).expanduser().resolve(),
        Path(args.mp4).expanduser().resolve() if args.mp4 else None,
    )
    print(f"Visualizations written to {args.out_dir}")
    if args.mp4:
        print(f"MP4 written to {args.mp4}")
