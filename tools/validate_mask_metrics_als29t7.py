"""Phase 5: Reference mask metrics validation.
Self-validate the mask metrics pipeline by comparing reference masks against themselves (should get perfect scores).
Runs on a small sample interval.

Generates outputs/mask_metric_validation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.processed_dataset_loader import discover_processed_datasets


def _row_mask_coords(row: pd.Series, width: int | None = None) -> set[tuple[int, int]]:
    """Extract mask pixel coordinates from row."""
    if "mask_px" in row.index and row["mask_px"] is not None:
        if width is None:
            raise ValueError("Width required for mask_px")
        px = np.asarray(row["mask_px"], dtype=np.int32)
        ys = (px // width).astype(int)
        xs = (px % width).astype(int)
        return set(zip(ys.tolist(), xs.tolist()))
    return set()


def _pairwise_stats(ref_mask: set[tuple[int, int]], pred_mask: set[tuple[int, int]]) -> dict[str, float]:
    """Compute IoU and Dice for two masks."""
    if not ref_mask or not pred_mask:
        return {"iou": 0.0, "dice": 0.0}
    intersection = len(ref_mask & pred_mask)
    union = len(ref_mask | pred_mask)
    iou = float(intersection / union) if union > 0 else 0.0
    dice = float((2 * intersection) / (len(ref_mask) + len(pred_mask))) if (len(ref_mask) + len(pred_mask)) > 0 else 0.0
    return {"iou": iou, "dice": dice}


def validate_mask_metrics(ref_dir: Path, sample_frame_start: int = 0, sample_frame_end: int = 100) -> dict[str, Any]:
    """Validate mask metrics by self-comparing reference masks."""
    report: dict[str, Any] = {
        'ref_dir': str(ref_dir),
        'sample_frames': [sample_frame_start, sample_frame_end],
        'metrics': {},
        'issues': []
    }

    # Load reference masks
    masks_path = ref_dir / 'seg_masks.parquet'
    if not masks_path.exists():
        report['issues'].append(f'seg_masks.parquet not found at {masks_path}')
        return report

    try:
        masks = pd.read_parquet(masks_path)
    except Exception as exc:
        report['issues'].append(f'Failed to load masks: {exc}')
        return report

    if 'abs_frame' not in masks.columns:
        report['issues'].append('abs_frame column missing from seg_masks.parquet')
        return report

    # Filter sample frames
    sample_masks = masks[
        (masks['abs_frame'] >= sample_frame_start) & (masks['abs_frame'] <= sample_frame_end)
    ].copy()

    if sample_masks.empty:
        report['issues'].append(f'No masks found in frame range [{sample_frame_start}, {sample_frame_end}]')
        return report

    # Determine frame width from first frame image or metadata
    frame_width = 512  # default fallback
    if 'width' in sample_masks.columns:
        frame_width = int(sample_masks['width'].iloc[0])
    report['frame_width'] = frame_width

    # Group by frame
    ious = []
    dices = []
    frames_processed = 0
    
    for frame_idx, grp in sample_masks.groupby('abs_frame'):
        frames_processed += 1
        for idx, (_, row) in enumerate(grp.iterrows()):
            try:
                mask_coords = _row_mask_coords(row, width=frame_width)
                if mask_coords:
                    # Self-comparison: mask should match itself perfectly
                    stats = _pairwise_stats(mask_coords, mask_coords)
                    ious.append(stats['iou'])
                    dices.append(stats['dice'])
            except Exception as exc:
                report['issues'].append(f'Frame {frame_idx} mask error: {exc}')

    report['metrics'] = {
        'frames_processed': frames_processed,
        'mask_objects_validated': len(ious),
        'mean_iou': float(np.mean(ious)) if ious else None,
        'mean_dice': float(np.mean(dices)) if dices else None,
        'min_iou': float(np.min(ious)) if ious else None,
        'max_iou': float(np.max(ious)) if ious else None,
    }

    # Self-comparison should yield perfect scores (IOU=1, Dice=1)
    expected_perfect = all(iou == 1.0 for iou in ious) if ious else False
    report['self_comparison_perfect'] = expected_perfect

    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-dir', default='New_experiments_v3_final')
    parser.add_argument('--frame-start', type=int, default=0)
    parser.add_argument('--frame-end', type=int, default=100)
    parser.add_argument('--out', default='outputs/mask_metric_validation.json')
    args = parser.parse_args()

    datasets = discover_processed_datasets(args.reference_dir)
    reports = []
    for dataset in datasets:
        ref_dir = Path(dataset['reference_dir']) if dataset.get('reference_dir') else Path(args.reference_dir)
        if ref_dir.exists():
            reports.append(validate_mask_metrics(ref_dir, sample_frame_start=args.frame_start, sample_frame_end=args.frame_end))

    report = {
        'dataset_root': args.reference_dir,
        'videos_found': len(datasets),
        'reports': reports,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Mask metric validation report written to {args.out}')
