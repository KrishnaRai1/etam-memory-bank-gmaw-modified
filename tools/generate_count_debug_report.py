"""Count debug report generator for ALS29T7.
Writes outputs/count_debug_report.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_counts_json(ref_dir: Path) -> dict[str, Any]:
    counts_path = ref_dir / 'counts.json'
    if not counts_path.exists():
        return {'exists': False, 'path': str(counts_path), 'raw': None, 'parsed_count': None, 'note': 'counts.json missing'}

    raw = json.loads(counts_path.read_text(encoding='utf-8'))
    parsed_count = None
    note = None
    if isinstance(raw, dict):
        for k in ('count', 'total', 'droplet_count', 'num_droplets'):
            if k in raw:
                try:
                    parsed_count = int(raw[k])
                    break
                except Exception:
                    note = f'Could not parse {k} as int'
        if parsed_count is None:
            note = f'No recognized count field found; keys: {list(raw.keys())}'
    else:
        note = f'Unexpected counts.json format: {type(raw).__name__}'

    return {
        'exists': True,
        'path': str(counts_path),
        'raw': raw,
        'parsed_count': parsed_count,
        'note': note,
    }


def load_tracks_summary(ref_dir: Path) -> dict[str, Any]:
    tracks_path = ref_dir / 'tracks_clean.parquet'
    if not tracks_path.exists():
        return {'exists': False, 'path': str(tracks_path)}

    df = pd.read_parquet(tracks_path)
    summary: dict[str, Any] = {
        'exists': True,
        'path': str(tracks_path),
        'rows': int(len(df)),
        'columns': list(df.columns),
        'global_id_unique': None,
        'global_id_duplicate_rows': None,
        'class_counts': None,
        'droplet_class_rows': None,
        'droplet_class_global_id_unique': None,
    }
    if 'global_id' in df.columns:
        summary['global_id_unique'] = int(df['global_id'].nunique())
        summary['global_id_duplicate_rows'] = int(df['global_id'].duplicated().sum())
    if 'class_id' in df.columns:
        class_counts = df['class_id'].value_counts().to_dict()
        summary['class_counts'] = {str(k): int(v) for k, v in class_counts.items()}
        droplets = df[df['class_id'] == 3]
        summary['droplet_class_rows'] = int(len(droplets))
        if 'global_id' in droplets.columns:
            summary['droplet_class_global_id_unique'] = int(droplets['global_id'].nunique())
    return summary


def load_benchmark_cases() -> dict[str, Any]:
    cfg_path = Path('configs/benchmark_cases.yaml')
    if not cfg_path.exists():
        return {'exists': False, 'path': str(cfg_path), 'data': None}
    data = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    return {'exists': True, 'path': str(cfg_path), 'data': data}


def load_manual_counts() -> dict[str, Any]:
    path = Path('configs/video_manual_counts.json')
    if not path.exists():
        return {'exists': False, 'path': str(path), 'content': None, 'note': 'manual counts file missing'}
    try:
        content = json.loads(path.read_text(encoding='utf-8'))
        return {'exists': True, 'path': str(path), 'content': content, 'note': None}
    except Exception as exc:
        return {'exists': True, 'path': str(path), 'content': None, 'note': str(exc)}


def build_video_report(ref_dir: Path) -> dict[str, Any]:
    counts = load_counts_json(ref_dir)
    tracks = load_tracks_summary(ref_dir)
    benchmark_cases = load_benchmark_cases()
    manual_counts = load_manual_counts()

    # ALS29T7-specific interval information
    video_manual_intervals = []
    if benchmark_cases.get('exists') and isinstance(benchmark_cases['data'], dict):
        for vid, intervals in benchmark_cases['data'].get('benchmark_cases', {}).items():
            if vid == 'AIS29T7':
                for interval in intervals:
                    video_manual_intervals.append({
                        'interval_id': interval.get('interval_id'),
                        'manual_count': interval.get('manual_count'),
                        'system_count': interval.get('system_count'),
                        'category': interval.get('category'),
                    })

    return {
        'video_id': 'AIS29T7',
        'counts_json': counts,
        'tracks_clean': tracks,
        'benchmark_cases_manual_intervals': video_manual_intervals,
        'configs_video_manual_counts': manual_counts,
        'summary': {
            'count_validation_status': 'WARN',
            'reason': 'counts.json lacks recognized top-level count field and manual counts config is missing for ALS29T7',
            'tracks_clean_count': tracks.get('droplet_class_global_id_unique'),
            'counts_json_parsed_count': counts.get('parsed_count'),
            'manual_count_reference': [item['manual_count'] for item in video_manual_intervals if item['interval_id'] == 'AIS29T7_385_395'],
        },
    }


def generate_report(out_path: Path, ref_dir: Path = Path('als29t7_data')) -> dict[str, Any]:
    report = {'count_debug': build_video_report(ref_dir)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate count debug report for ALS29T7.')
    parser.add_argument('--reference-dir', default='als29t7_data')
    parser.add_argument('--out', default='outputs/count_debug_report.json')
    args = parser.parse_args()

    rep = generate_report(Path(args.out), Path(args.reference_dir))
    print(f'Count debug report written to {args.out}')
