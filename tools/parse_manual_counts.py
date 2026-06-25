"""Parse manual counts from available CSV/Excel annotation files and write configs/video_manual_counts.json

Searches for common annotation files in repo root and extracts per-video manual counts.
Reports duplicates/conflicts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd


def _find_annotation_files(root: Path) -> list[Path]:
    candidates = []
    for ext in ('.csv', '.xls', '.xlsx'):
        candidates.extend(sorted(root.glob(f'*{ext}')))
    return candidates


def _extract_from_df(df: pd.DataFrame) -> Dict[str, int]:
    # find candidate columns
    cols = [c.lower() for c in df.columns.astype(str)]
    video_cols = [c for c in df.columns if 'video' in str(c).lower()]
    manual_cols = [c for c in df.columns if 'manual' in str(c).lower() or 'manually' in str(c).lower()]

    mapping: Dict[str, set[int]] = {}

    if not video_cols or not manual_cols:
        # attempt heuristics: last two columns are video id and manual count
        if df.shape[1] >= 2:
            video_col = df.columns[-2]
            manual_col = df.columns[-1]
        else:
            return {}
    else:
        video_col = video_cols[0]
        manual_col = manual_cols[0]

    for _, row in df.iterrows():
        v = row.get(video_col)
        m = row.get(manual_col)
        if pd.isna(v) or pd.isna(m):
            continue
        try:
            v_s = str(v).strip()
            m_i = int(float(m))
        except Exception:
            continue
        mapping.setdefault(v_s, set()).add(m_i)

    # resolve sets to single int or mark conflict
    result: Dict[str, int] = {}
    for v, s in mapping.items():
        if len(s) == 1:
            result[v] = next(iter(s))
        else:
            # prefer max as a conservative choice but record conflict by using -1 placeholder
            result[v] = -1
    return result


def parse_and_write(root: Path, out_path: Path) -> Dict[str, Any]:
    files = _find_annotation_files(root)
    combined: Dict[str, set[int]] = {}
    issues = []
    for f in files:
        try:
            if f.suffix.lower() == '.csv':
                df = pd.read_csv(f, dtype=str, keep_default_na=False)
            else:
                df = pd.read_excel(f, dtype=str)
        except Exception as exc:
            issues.append(f"Failed to read {f}: {exc}")
            continue
        extracted = _extract_from_df(df)
        for k, v in extracted.items():
            if v == -1:
                issues.append(f"Conflict for {k} in file {f}")
                combined.setdefault(k, set()).add(-1)
            else:
                combined.setdefault(k, set()).add(int(v))

    final: Dict[str, int] = {}
    conflicts = {}
    for k, s in combined.items():
        if -1 in s or len(s) > 1:
            conflicts[k] = sorted(list(s))
            final[k] = -1
        else:
            final[k] = next(iter(s))

    out = {'video_manual_counts': final, 'conflicts': conflicts, 'sources': [str(p) for p in files]}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    return out


if __name__ == '__main__':
    root = Path('.').resolve()
    out = Path('configs') / 'video_manual_counts.json'
    report = parse_and_write(root, out)
    print(f"Wrote {out}: {report}")
