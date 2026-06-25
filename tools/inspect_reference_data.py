"""Inspect reference parquet files: tracks_clean.parquet, seg_masks.parquet, daq.
Print schemas, dtypes, and row counts for debugging.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def inspect(reference_dir: Path) -> dict:
    report = {}
    for name in ('tracks_clean.parquet','seg_masks.parquet'):
        p = reference_dir / name
        if not p.exists():
            report[name] = {'exists': False}
            continue
        try:
            df = pd.read_parquet(p)
            report[name] = {'exists': True, 'rows': int(len(df)), 'columns': {str(c): str(dtype) for c, dtype in zip(df.columns, df.dtypes)}}
        except Exception as exc:
            report[name] = {'exists': True, 'error': str(exc)}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-dir', required=True)
    args = parser.parse_args()
    rep = inspect(Path(args.reference_dir))
    import json
    print(json.dumps(rep, indent=2))
