"""Phase 4: Reference count validation.
Compares counts from:
1. counts.json (in als29t7_data)
2. tracks_clean.parquet unique global_id count
3. Excel manual counts (if available)

Generates outputs/count_validation_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.processed_dataset_loader import discover_processed_datasets


def validate_counts(ref_dir: Path, manual_counts_cfg: Path | None = None) -> dict[str, Any]:
    """Validate reference count sources."""
    report: dict[str, Any] = {
        'ref_dir': str(ref_dir),
        'sources': {
            'counts_json': {'exists': False, 'count': None, 'error': None},
            'tracks_clean': {'exists': False, 'count': None, 'error': None, 'droplet_class_count': None},
            'manual_count': {'exists': False, 'count': None, 'error': None}
        },
        'comparison': {}
    }

    # Source 1: counts.json
    counts_path = ref_dir / 'counts.json'
    if counts_path.exists():
        report['sources']['counts_json']['exists'] = True
        try:
            data = json.loads(counts_path.read_text(encoding='utf-8'))
            # assume counts.json has a top-level count or similar structure
            if isinstance(data, dict) and 'count' in data:
                report['sources']['counts_json']['count'] = int(data['count'])
            elif isinstance(data, dict):
                # try to find a count-like key
                for k in ('total', 'count', 'droplet_count', 'num_droplets'):
                    if k in data:
                        report['sources']['counts_json']['count'] = int(data[k])
                        break
                if report['sources']['counts_json']['count'] is None:
                    report['sources']['counts_json']['error'] = f'No count field found; keys: {list(data.keys())}'
        except Exception as exc:
            report['sources']['counts_json']['error'] = str(exc)

    # Source 2: tracks_clean.parquet
    tracks_path = ref_dir / 'tracks_clean.parquet'
    if tracks_path.exists():
        report['sources']['tracks_clean']['exists'] = True
        try:
            tracks = pd.read_parquet(tracks_path)
            if 'global_id' in tracks.columns:
                report['sources']['tracks_clean']['count'] = int(tracks['global_id'].nunique())
            # Also count droplets specifically (class_id == 3)
            if 'class_id' in tracks.columns:
                droplets = tracks[tracks['class_id'] == 3]
                report['sources']['tracks_clean']['droplet_class_count'] = int(droplets['global_id'].nunique() if 'global_id' in droplets.columns else 0)
        except Exception as exc:
            report['sources']['tracks_clean']['error'] = str(exc)

    # Source 3: Excel manual counts
    if manual_counts_cfg and manual_counts_cfg.exists():
        try:
            manual = json.loads(manual_counts_cfg.read_text(encoding='utf-8'))
            manual_mapping = manual.get('video_manual_counts', {})
            # Try to find entry for ALS29T7 or AIS29T7
            count_val = None
            for key in ('ALS29T7', 'AIS29T7', 'als29t7', 'ais29t7'):
                if key in manual_mapping:
                    count_val = manual_mapping[key]
                    break
            if count_val is not None and count_val > 0:
                report['sources']['manual_count']['exists'] = True
                report['sources']['manual_count']['count'] = int(count_val)
        except Exception as exc:
            report['sources']['manual_count']['error'] = str(exc)

    # Comparison
    counts = []
    for source, data in report['sources'].items():
        if data['exists'] and data['count'] is not None:
            counts.append((source, data['count']))

    if len(counts) >= 2:
        report['comparison']['sources_match'] = all(c[1] == counts[0][1] for c in counts)
        report['comparison']['sources_summary'] = {s: c for s, c in counts}
        if not report['comparison']['sources_match']:
            min_c = min(c[1] for _, c in counts)
            max_c = max(c[1] for _, c in counts)
            report['comparison']['discrepancy_range'] = [min_c, max_c]

    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-dir', default='New_experiments_v3_final')
    parser.add_argument('--manual-counts', default='configs/video_manual_counts.json')
    parser.add_argument('--out', default='outputs/count_validation_report.json')
    args = parser.parse_args()

    datasets = discover_processed_datasets(args.reference_dir)
    reports = []
    for dataset in datasets:
        ref_dir = Path(dataset['reference_dir']) if dataset.get('reference_dir') else Path(args.reference_dir)
        if ref_dir.exists():
            reports.append(validate_counts(ref_dir, Path(args.manual_counts) if Path(args.manual_counts).exists() else None))

    rep = {
        'dataset_root': args.reference_dir,
        'videos_found': len(datasets),
        'reports': reports,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2), encoding='utf-8')
    print(f'Count validation report written to {args.out}')
