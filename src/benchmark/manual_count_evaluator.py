from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _infer_tracks_in_dir(root: Path) -> list[Path]:
    candidates = []
    for name in ('tracks_clean.parquet','tracks.parquet','tracks_merged.parquet'):
        # direct children
        p = root / name
        if p.exists():
            candidates.append(p)
    # search one level deep
    for child in root.iterdir() if root.exists() else []:
        if child.is_dir():
            for name in ('tracks_clean.parquet','tracks.parquet','tracks_merged.parquet'):
                p = child / name
                if p.exists():
                    candidates.append(p)
    return candidates


def _predicted_count_from_tracks(path: Path) -> int:
    df = pd.read_parquet(path)
    # find id column
    id_col = None
    for c in ('global_id','id','track_id','droplet_id'):
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        return 0
    # filter droplet class if present
    if 'class_id' in df.columns:
        df = df[df['class_id'] == 3]
    return int(df[id_col].nunique())


def evaluate_manual_counts(manual_counts_path: Path, output_root: Path = Path('outputs')) -> Path:
    manual = json.loads(manual_counts_path.read_text(encoding='utf-8'))
    mapping = manual.get('video_manual_counts', {}) if isinstance(manual, dict) else manual
    results: Dict[str, Any] = {}
    for vid, manual_count in mapping.items():
        entry: Dict[str, Any] = {'reference_count': None, 'predicted_count': None, 'absolute_error': None,'relative_error': None,'percent_error': None, 'issues': []}
        if manual_count is None:
            entry['issues'].append('manual_count_missing')
        else:
            entry['reference_count'] = int(manual_count) if int(manual_count) >=0 else None

        # locate predicted tracks under outputs
        candidates = []
        # check common locations
        for p in [output_root / vid, output_root / 'benchmark_runs' / vid, output_root]:
            candidates.extend(_infer_tracks_in_dir(p))

        if not candidates:
            entry['issues'].append('no_predicted_tracks_found')
            results[vid] = entry
            continue

        # choose the largest candidate by unique ids
        best_count = 0
        best_path = None
        for c in candidates:
            try:
                cnt = _predicted_count_from_tracks(c)
            except Exception as exc:
                entry['issues'].append(f'failed_read_{c}:{exc}')
                continue
            if cnt > best_count:
                best_count = cnt
                best_path = c

        if best_path is None:
            entry['issues'].append('no_valid_tracks_parquet')
            results[vid] = entry
            continue

        entry['predicted_count'] = int(best_count)
        if entry['reference_count'] is not None:
            entry['absolute_error'] = int(entry['predicted_count'] - entry['reference_count'])
            try:
                entry['relative_error'] = float(abs(entry['absolute_error']) / float(entry['reference_count'])) if entry['reference_count']>0 else None
                entry['percent_error'] = float(entry['relative_error']*100) if entry['relative_error'] is not None else None
            except Exception:
                entry['relative_error'] = None
                entry['percent_error'] = None

        results[vid] = entry

    out_path = Path('outputs') / 'manual_count_metrics.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    return out_path


if __name__ == '__main__':
    p = Path('configs') / 'video_manual_counts.json'
    if not p.exists():
        print('configs/video_manual_counts.json not found; run tools/parse_manual_counts.py')
    else:
        out = evaluate_manual_counts(p, output_root=Path('outputs'))
        print(f'Wrote {out}')
