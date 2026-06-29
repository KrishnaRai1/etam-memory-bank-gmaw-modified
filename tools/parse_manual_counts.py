"""Parse manual counts from available CSV/Excel annotation files and write configs/video_manual_counts.json

Searches for common annotation files in repo root and extracts per-video manual counts.
Reports duplicates/conflicts.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any

import pandas as pd


VIDEO_ID_RE = re.compile(r"^(AIS|ALS)\d{2}T\d+(?:R\d+)?$", re.IGNORECASE)


def _find_annotation_files(root: Path) -> list[Path]:
    candidates = []
    for ext in ('.csv', '.xls', '.xlsx'):
        candidates.extend(sorted(root.glob(f'*{ext}')))
    return candidates


def _normalize_video_id(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = re.sub(r"\s+", "", text)
    text = text.replace("_", "")
    upper = text.upper()
    if VIDEO_ID_RE.match(upper):
        return upper
    return None


def _load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, header=None)
    try:
        return pd.read_excel(path, header=None)
    except Exception:
        return pd.DataFrame()


def _extract_from_df(df: pd.DataFrame) -> Dict[str, int]:
    if df.empty:
        return {}

    video_col_idx = None
    manual_col_idx = None
    system_col_idx = None

    for row_idx in range(min(len(df), 20)):
        row = df.iloc[row_idx]
        for col_idx, cell in enumerate(row.tolist()):
            text = str(cell).strip().lower() if not pd.isna(cell) else ""
            if text == "video id" or text == "video_id":
                video_col_idx = col_idx
            elif "manually counted droplet" in text or text == "manual count" or text == "manual_count":
                manual_col_idx = col_idx
            elif "system counted droplet" in text or text == "system count" or text == "system_count":
                system_col_idx = col_idx
        if video_col_idx is not None and manual_col_idx is not None:
            break

    mapping: Dict[str, set[int]] = {}

    if video_col_idx is not None and manual_col_idx is not None:
        for row_idx in range(video_col_idx + 1, len(df)):
            row = df.iloc[row_idx]
            video = _normalize_video_id(row.iloc[video_col_idx])
            manual = row.iloc[manual_col_idx]
            if video is None or pd.isna(manual):
                continue
            try:
                manual_count = int(float(manual))
            except Exception:
                continue
            mapping.setdefault(video, set()).add(manual_count)

    if not mapping:
        # fallback: scan every row for the first valid video id and a nearby numeric count
        for _, row in df.iterrows():
            values = [cell for cell in row.tolist() if not pd.isna(cell)]
            video = next((v for v in (_normalize_video_id(cell) for cell in values) if v is not None), None)
            numeric_values = []
            for cell in values:
                try:
                    numeric_values.append(int(float(cell)))
                except Exception:
                    pass
            if video and numeric_values:
                mapping.setdefault(video, set()).add(numeric_values[0])

    result: Dict[str, int] = {}
    for video, counts in mapping.items():
        result[video] = next(iter(counts)) if len(counts) == 1 else -1
    return result


def parse_and_write(root: Path, out_path: Path) -> Dict[str, Any]:
    files = _find_annotation_files(root)
    combined: Dict[str, set[int]] = {}
    issues = []
    for f in files:
        try:
            df = _load_table(f)
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
