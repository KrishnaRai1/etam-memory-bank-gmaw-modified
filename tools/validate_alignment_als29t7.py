"""Phase 3: Frame alignment validation.
Samples random frames and overlays reference droplet masks (class_id==3 from tracks_clean.parquet).
Verifies abs_frame aligns with frame filenames and produces alignment_report.json + overlay PNGs.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def px_to_mask(px_list: list[int] | np.ndarray | None, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if px_list is None:
        return mask
    arr = np.asarray(px_list, dtype=np.int32)
    if arr.size == 0:
        return mask
    ys = (arr // width).astype(int)
    xs = (arr % width).astype(int)
    valid = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
    mask[ys[valid], xs[valid]] = 255
    return mask


def find_frame_image(frames_dir: Path, abs_frame: int, width: int | None = None) -> tuple[Path | None, int, int]:
    """Find frame image by abs_frame number. Return (path, W, H) or (None, 0, 0)."""
    if not frames_dir.exists():
        return None, 0, 0
    # try various naming patterns
    for pattern in [f'{abs_frame:06d}.*', f'{abs_frame}.*', f'*{abs_frame}.*']:
        matches = list(frames_dir.glob(pattern))
        if matches:
            img = Image.open(matches[0])
            return matches[0], img.width, img.height
    return None, 0, 0


def validate_alignment(
    ref_dir: Path, frames_dir: Path, out_dir: Path, num_samples: int = 20
) -> dict[str, Any]:
    """Validate frame alignment between reference masks and frame images."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    report: dict[str, Any] = {
        'frames_dir': str(frames_dir),
        'ref_dir': str(ref_dir),
        'num_samples': num_samples,
        'samples': [],
        'issues': []
    }

    # Load reference tracks (droplet class_id==3)
    tracks_path = ref_dir / 'tracks_clean.parquet'
    if not tracks_path.exists():
        report['issues'].append(f'tracks_clean.parquet not found at {tracks_path}')
        return report

    try:
        tracks = pd.read_parquet(tracks_path)
    except Exception as exc:
        report['issues'].append(f'Failed to load tracks: {exc}')
        return report

    # Filter droplets (class_id == 3)
    if 'class_id' not in tracks.columns:
        report['issues'].append('class_id column missing from tracks_clean.parquet')
        return report
    
    droplets = tracks[tracks['class_id'] == 3].copy()
    if droplets.empty:
        report['issues'].append('No droplets (class_id==3) found in tracks_clean.parquet')
        return report

    # Sample frames
    all_frames = sorted(droplets['abs_frame'].unique().tolist())
    if not all_frames:
        report['issues'].append('No frames in droplet tracks')
        return report
    
    sample_frames = random.sample(all_frames, min(num_samples, len(all_frames)))

    # Get frame dimensions from first available image
    frame_w, frame_h = 0, 0
    for frame_idx in all_frames[:10]:
        path, w, h = find_frame_image(frames_dir, int(frame_idx))
        if path:
            frame_w, frame_h = w, h
            break

    if frame_w == 0 or frame_h == 0:
        report['issues'].append('Could not determine frame dimensions from images')
        # Continue anyway with best-effort

    # Process samples
    for frame_idx in sample_frames:
        frame_data: dict[str, Any] = {
            'abs_frame': int(frame_idx),
            'droplet_count': 0,
            'mask_areas': [],
            'centroids': [],
            'image_found': False,
            'overlay_saved': False
        }

        # Find frame image
        img_path, w, h = find_frame_image(frames_dir, int(frame_idx))
        if img_path is None:
            frame_data['image_found'] = False
            report['samples'].append(frame_data)
            continue

        frame_data['image_found'] = True
        frame_data['image_dims'] = [w, h]
        
        try:
            img = Image.open(img_path).convert('RGB')
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
        except Exception as exc:
            frame_data['image_error'] = str(exc)
            report['samples'].append(frame_data)
            continue

        # Draw droplet masks on overlay
        rows = droplets[droplets['abs_frame'] == frame_idx]
        frame_data['droplet_count'] = int(len(rows))

        for _, row in rows.iterrows():
            px = row.get('mask_px')
            if px is None or (hasattr(px, '__len__') and len(px) == 0):
                continue
            try:
                mask = px_to_mask(px, width=w, height=h)
                ys, xs = np.where(mask > 0)
                if ys.size == 0:
                    continue
                area = int(ys.size)
                frame_data['mask_areas'].append(area)
                centroid_x = float(xs.mean())
                centroid_y = float(ys.mean())
                frame_data['centroids'].append([centroid_x, centroid_y])
                
                # Draw contour
                minx, maxx = int(xs.min()), int(xs.max())
                miny, maxy = int(ys.min()), int(ys.max())
                draw.rectangle([(minx, miny), (maxx, maxy)], outline=(255, 0, 0, 160))
                draw.ellipse(
                    [(centroid_x - 3, centroid_y - 3), (centroid_x + 3, centroid_y + 3)],
                    fill=(255, 0, 0, 200)
                )
            except Exception as exc:
                frame_data['mask_error'] = str(exc)
                continue

        # Save overlay
        try:
            combined = Image.alpha_composite(img.convert('RGBA'), overlay)
            overlay_path = out_dir / f'alignment_{int(frame_idx):06d}.png'
            combined.save(overlay_path)
            frame_data['overlay_saved'] = True
            frame_data['overlay_path'] = str(overlay_path.name)
        except Exception as exc:
            frame_data['overlay_error'] = str(exc)

        report['samples'].append(frame_data)

    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-dir', default='New_experiments_v3_final')
    parser.add_argument('--frames-dir', default='.')
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--out-dir', default='outputs/alignment_validation')
    args = parser.parse_args()

    rep = validate_alignment(
        Path(args.reference_dir),
        Path(args.frames_dir),
        Path(args.out_dir),
        args.num_samples
    )
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_dir) / 'alignment_report.json'
    out_json.write_text(json.dumps(rep, indent=2), encoding='utf-8')
    print(f'Alignment validation report written to {out_json}')
