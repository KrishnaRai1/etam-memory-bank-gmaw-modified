from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _infer_id_column(df: pd.DataFrame) -> str:
    for candidate in ("id", "track_id", "droplet_id", "obj_id", "global_id"):
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Count metrics dataframe is missing an ID column. Found columns: {list(df.columns)}")


def _unique_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    id_col = _infer_id_column(df)
    return int(df[id_col].nunique())


def compute_count_metrics(reference_tracks: pd.DataFrame, predicted_tracks: pd.DataFrame) -> dict[str, Any]:
    reference_count = _unique_count(reference_tracks)
    predicted_count = _unique_count(predicted_tracks)
    count_error = predicted_count - reference_count
    absolute_error = abs(count_error)
    relative_error = None if reference_count == 0 else round(absolute_error / float(reference_count), 4)

    return {
        "reference_droplet_count": reference_count,
        "tracked_droplet_count": predicted_count,
        "count_error": int(count_error),
        "absolute_error": int(absolute_error),
        "relative_error": relative_error,
    }


def save_count_metrics(metrics: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output_path
