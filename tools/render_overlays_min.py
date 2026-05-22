#!/usr/bin/env python3
# tools/render_overlays_min.py
# -*- coding: utf-8 -*-
# Renders one PNG per frame with:
#   - translucent mask per tracked ID (deterministic color)
#   - numeric ID near each centroid
#   - top-left: frame index + cumulative droplet count
#   - optional: segment-only masks (e.g. weld pool) painted under the tracked ones
# Workers process frames in parallel; output goes to <out-dir>/<frame>.png.

from __future__ import annotations
from pathlib import Path
import argparse, json
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, get_start_method
from functools import partial

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ---------------- style ----------------
ALPHA_TRACKED   = 0.45
ALPHA_SEGONLY   = 0.30
SEGONLY_COLOR   = (255, 165, 0)   # orange — distinguishes seg-only from tracked masks
DROPLET_COLOR_FIXED = None        # set to e.g. (255, 0, 0) to force a single color for all IDs

DRAW_FRAME_ID      = True
DRAW_OBJECT_IDS    = True
DRAW_RUNNING_COUNT = True

FONT_PATH       = "DejaVuSans-Bold.ttf"
FONT_SIZE_SMALL = 10
FONT_SIZE_BIG   = 13

NUM_WORKERS = "auto"
CHUNK_SIZE  = 16
# ---------------------------------------


def _rgb_for_id(oid: int) -> tuple[int, int, int]:
    # Deterministic prime-multiplier hash: same ID always gets the same color.
    r = (37 * oid) % 255
    g = (91 * oid) % 255
    b = (173 * oid) % 255
    return (r, g, b)


def _try_load_font():
    # Falls back to PIL's default bitmap font when DejaVu is not installed.
    try:
        return (ImageFont.truetype(FONT_PATH, FONT_SIZE_SMALL),
                ImageFont.truetype(FONT_PATH, FONT_SIZE_BIG))
    except Exception:
        f = ImageFont.load_default()
        return f, f


def _draw_text_outline(draw, xy, text, fill=(255, 255, 255), outline=(0, 0, 0), font=None):
    # 4-pixel outline so labels stay readable on any background.
    x, y = xy
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text((x + dx, y + dy), text, fill=outline, font=font)
    draw.text((x, y), text, fill=fill, font=font)


def _series_to_np(a):
    if isinstance(a, np.ndarray):
        arr = a
    else:
        arr = np.asarray(list(a), dtype=np.float64)
    if arr.size:
        arr = arr[~np.isnan(arr)]
        arr = arr.astype(np.int64, copy=False)
    return arr


def _frames_from_meta(final_dir: Path, images_dir: Path) -> list[Path]:
    # Use frame_names from frames_meta.json when available so the overlay index
    # always lines up with the parquet frame_idx. Fall back to scanning images_dir.
    exts = {".jpg", ".jpeg", ".png"}
    meta_p = final_dir / "frames_meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        names = meta.get("frame_names") or []
        paths = []
        for n in names:
            p = Path(n)
            cand = p if p.exists() else (images_dir / p.name)
            if cand.exists() and cand.suffix.lower() in exts:
                paths.append(cand)
        if paths:
            return paths
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


def _load_parquets(final_dir: Path):
    # Prefer the post-processed tracks; fall back to merged if postprocess was skipped.
    tr_clean = final_dir / "tracks_clean.parquet"
    tr_merge = final_dir / "tracks_merged.parquet"
    seg_p    = final_dir / "segonly_merged.parquet"

    tracks = pd.read_parquet(tr_clean) if tr_clean.exists() else \
             (pd.read_parquet(tr_merge) if tr_merge.exists() else pd.DataFrame())
    seg    = pd.read_parquet(seg_p) if seg_p.exists() else pd.DataFrame()

    if not tracks.empty:
        for col in ("frame_idx", "id"):
            tracks[col] = tracks[col].astype("int64", errors="ignore")
    if not seg.empty:
        seg["frame_idx"] = seg["frame_idx"].astype("int64", errors="ignore")
    return tracks, seg


def _group_by_frame(df: pd.DataFrame):
    if df is None or df.empty:
        return {}
    return {int(k): g for k, g in df.sort_values("frame_idx").groupby("frame_idx", sort=True)}


def _pack_jobs(frames, g_tracks, g_seg, ids_seen_up_to):
    # Build self-contained job tuples per frame so workers don't need pandas
    # or any shared state at runtime.
    jobs = []
    for fidx, img_path in enumerate(frames):
        tr = g_tracks.get(fidx)
        sg = g_seg.get(fidx)

        tracked = []
        if tr is not None:
            for _, row in tr.iterrows():
                ys = _series_to_np(row["ys"]); xs = _series_to_np(row["xs"])
                if ys.size and xs.size:
                    tracked.append((int(row["id"]), ys, xs))

        seg_masks = []
        if sg is not None:
            for _, row in sg.iterrows():
                ys = _series_to_np(row["ys"]); xs = _series_to_np(row["xs"])
                if ys.size and xs.size:
                    seg_masks.append((ys, xs))

        jobs.append((fidx, str(img_path), tracked, seg_masks, int(ids_seen_up_to.get(fidx, 0))))
    return jobs


def _render_one(job, out_dir: Path):
    # Compose layers on top of the source frame, then burn text labels with PIL.
    fidx, img_path, tracked, seg_masks, running_count = job

    base = Image.open(img_path).convert("RGBA")
    W, H = base.size
    overlay_arr = np.zeros((H, W, 4), dtype=np.uint8)

    # seg-only goes underneath so tracked masks overlay it cleanly.
    if seg_masks:
        a = int(max(0.0, min(1.0, ALPHA_SEGONLY)) * 255)
        col = np.array([SEGONLY_COLOR[0], SEGONLY_COLOR[1], SEGONLY_COLOR[2], a], dtype=np.uint8)
        for ys, xs in seg_masks:
            ys = np.clip(ys, 0, H - 1); xs = np.clip(xs, 0, W - 1)
            overlay_arr[ys, xs] = col

    # tracked
    a = int(max(0.0, min(1.0, ALPHA_TRACKED)) * 255)
    for oid, ys, xs in tracked:
        rgb = DROPLET_COLOR_FIXED if DROPLET_COLOR_FIXED is not None else _rgb_for_id(int(oid))
        col = np.array([rgb[0], rgb[1], rgb[2], a], dtype=np.uint8)
        ys = np.clip(ys, 0, H - 1); xs = np.clip(xs, 0, W - 1)
        overlay_arr[ys, xs] = col

    overlay = Image.fromarray(overlay_arr, mode="RGBA")
    comp = Image.alpha_composite(base, overlay).convert("RGB")
    draw = ImageDraw.Draw(comp)
    f_small, f_big = _try_load_font()

    # ID labels
    if DRAW_OBJECT_IDS:
        for oid, ys, xs in tracked:
            cx = int(np.mean(xs)); cy = int(np.mean(ys))
            _draw_text_outline(draw, (cx, cy), f"{int(oid)}",
                               fill=(255, 255, 255), outline=(0, 0, 0), font=f_small)

    # Frame index + running count, both stacked top-left.
    y_cursor = 6
    if DRAW_FRAME_ID:
        _draw_text_outline(draw, (8, y_cursor), f"f {fidx:05d}",
                           fill=(255, 255, 0), outline=(0, 0, 0), font=f_big)
        y_cursor += 16
    if DRAW_RUNNING_COUNT:
        _draw_text_outline(draw, (8, y_cursor), f"droplets = {running_count}",
                           fill=(255, 255, 255), outline=(0, 0, 0), font=f_small)

    out_path = out_dir / f"{fidx:05d}.png"
    comp.save(out_path)
    return fidx


def main():
    ap = argparse.ArgumentParser("Render minimal overlays: masks + IDs + count")
    ap.add_argument("--images-dir", type=str, required=True)
    ap.add_argument("--final-dir",  type=str, required=True)
    ap.add_argument("--out-dir",    type=str, required=True)
    args = ap.parse_args()

    images_dir = Path(args.images_dir).expanduser().resolve()
    final_dir  = Path(args.final_dir).expanduser().resolve()
    out_dir    = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = _frames_from_meta(final_dir, images_dir)
    tracks, seg = _load_parquets(final_dir)
    g_tr = _group_by_frame(tracks)
    g_sg = _group_by_frame(seg)

    # Running count per frame = number of IDs whose first_frame <= fidx.
    # Computed once via searchsorted on the sorted first-frame array.
    ids_seen_up_to: dict[int, int] = {}
    if not tracks.empty:
        starts = tracks.groupby("id")["frame_idx"].min().sort_values()
        sorted_starts = starts.to_numpy()
        for fidx in range(len(frames)):
            ids_seen_up_to[fidx] = int(np.searchsorted(sorted_starts, fidx, side="right"))

    jobs = _pack_jobs(frames, g_tr, g_sg, ids_seen_up_to)

    workers = cpu_count() if NUM_WORKERS == "auto" else int(NUM_WORKERS)
    workers = max(1, workers)
    # spawn keeps PIL/numpy state from leaking between workers; ignore if already set.
    try:
        if get_start_method(allow_none=True) is None:
            import multiprocessing as mp
            mp.set_start_method("spawn")
    except RuntimeError:
        pass

    worker_fn = partial(_render_one, out_dir=out_dir)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(worker_fn, job) for job in jobs]
        for _ in tqdm(as_completed(futs), total=len(futs), desc=f"Render x{workers}w"):
            pass

    print(f"[OK] wrote overlays → {out_dir}")
    if not tracks.empty:
        print(f"[count] total unique droplets = {int(tracks['id'].nunique())}")


if __name__ == "__main__":
    main()
