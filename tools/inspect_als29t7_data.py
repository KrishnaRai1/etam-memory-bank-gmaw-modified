"""Inspect ALS29T7 reference data: parquet files and JSON metadata.
Exports outputs/reference_inspection.json with schema and sample rows.
Also generates outputs/schema_validation_report.json with expected columns check.
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


def inspect_reference_dir(ref_dir: Path) -> dict[str, Any]:
    """Inspect all parquets and JSON files in reference directory."""
    report = {'path': str(ref_dir), 'exists': ref_dir.exists(), 'files': {}}
    if not ref_dir.exists():
        return report

    # Parquet files
    for name in ('tracks_clean.parquet', 'tracks.parquet', 'seg_masks.parquet', 'daq.parquet'):
        p = ref_dir / name
        entry = {'exists': p.exists()}
        if p.exists():
            try:
                df = pd.read_parquet(p)
                entry['rows'] = int(len(df))
                entry['columns'] = {str(c): str(df[c].dtype) for c in df.columns}
                # sample first and last rows
                try:
                    sample_first = df.iloc[0].to_dict()
                    # convert non-serializable types to strings
                    entry['sample_first'] = {k: str(v) for k, v in sample_first.items()}
                except Exception:
                    pass
                try:
                    sample_last = df.iloc[-1].to_dict()
                    entry['sample_last'] = {k: str(v) for k, v in sample_last.items()}
                except Exception:
                    pass
            except Exception as exc:
                entry['error'] = str(exc)
        report['files'][name] = entry

    # JSON files
    for name in ('counts.json', 'run_summary.json', 'render_report.json', 'postprocess_report.json'):
        p = ref_dir / name
        entry = {'exists': p.exists()}
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding='utf-8'))
                entry['type'] = type(data).__name__
                if isinstance(data, dict):
                    entry['keys'] = list(data.keys())
                elif isinstance(data, list):
                    entry['length'] = len(data)
                entry['sample'] = str(data)[:200]  # first 200 chars
            except Exception as exc:
                entry['error'] = str(exc)
        report['files'][name] = entry

    return report


def validate_schema(ref_dir: Path) -> dict[str, Any]:
    """Validate expected columns in reference parquets."""
    result = {'ref_dir': str(ref_dir), 'validations': {}}

    # tracks_clean.parquet expected
    entry = {'expected': ['global_id', 'abs_frame', 'mask_px', 'class_id'], 'present': [], 'missing': [], 'ok': False}
    p = ref_dir / 'tracks_clean.parquet'
    if p.exists():
        try:
            df = pd.read_parquet(p)
            entry['present'] = [c for c in entry['expected'] if c in df.columns]
            entry['missing'] = [c for c in entry['expected'] if c not in df.columns]
            entry['ok'] = len(entry['missing']) == 0
            entry['actual_columns'] = list(df.columns)
        except Exception as exc:
            entry['error'] = str(exc)
    else:
        entry['error'] = 'File not found'
    result['validations']['tracks_clean.parquet'] = entry

    # seg_masks.parquet expected
    entry = {'expected': ['mask_px', 'cls_id'], 'present': [], 'missing': [], 'ok': False}
    p = ref_dir / 'seg_masks.parquet'
    if p.exists():
        try:
            df = pd.read_parquet(p)
            entry['present'] = [c for c in entry['expected'] if c in df.columns]
            entry['missing'] = [c for c in entry['expected'] if c not in df.columns]
            entry['ok'] = len(entry['missing']) == 0
            entry['actual_columns'] = list(df.columns)
        except Exception as exc:
            entry['error'] = str(exc)
    else:
        entry['error'] = 'File not found'
    result['validations']['seg_masks.parquet'] = entry

    result['overall_ok'] = all(v.get('ok', False) for v in result['validations'].values())
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-dir', default='New_experiments_v3_final', help='Path to processed dataset root')
    parser.add_argument('--out-inspection', default='outputs/reference_inspection.json')
    parser.add_argument('--out-validation', default='outputs/schema_validation_report.json')
    args = parser.parse_args()

    ref_dir = Path(args.reference_dir).resolve()
    datasets = discover_processed_datasets(ref_dir)
    inventory = {'dataset_root': str(ref_dir), 'videos_found': len(datasets), 'videos': datasets}
    Path(args.out_inspection).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_inspection).write_text(json.dumps(inventory, indent=2, default=str), encoding='utf-8')
    print(f'Inspection written to {args.out_inspection}')

    if datasets:
        first_ref = Path(datasets[0]['reference_dir']) if datasets[0].get('reference_dir') else ref_dir
        schema = validate_schema(first_ref)
    else:
        schema = {'ref_dir': str(ref_dir), 'overall_ok': False, 'validations': {}}
    Path(args.out_validation).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_validation).write_text(json.dumps(schema, indent=2), encoding='utf-8')
    print(f'Schema validation written to {args.out_validation}')
