from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.benchmark.ontology import filter_semantic_droplets


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


def _filter_droplets(df: pd.DataFrame) -> pd.DataFrame:
    return filter_semantic_droplets(df)


def compute_count_metrics(reference_tracks: pd.DataFrame, predicted_tracks: pd.DataFrame, interval_frames: set[int] | None = None) -> dict[str, Any]:
    reference_tracks = _filter_droplets(reference_tracks)
    if interval_frames is not None:
        frame_col = None
        for candidate in ("abs_frame", "frame_idx", "rel_frame", "frame"):
            if candidate in reference_tracks.columns:
                frame_col = candidate
                break
        if frame_col is not None:
            reference_tracks = reference_tracks[reference_tracks[frame_col].isin(interval_frames)].copy()
    if interval_frames is not None and reference_tracks.empty:
        reference_tracks = reference_tracks.iloc[0:0].copy()
    predicted_tracks = _filter_droplets(predicted_tracks)
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


def compute_interval_reference_count(reference_tracks: pd.DataFrame, start_frame: int, end_frame: int) -> int | None:
    if reference_tracks.empty:
        return None
    if "class_id" in reference_tracks.columns or "cls_id" in reference_tracks.columns:
        reference_tracks = filter_semantic_droplets(reference_tracks)
    frame_col = None
    for candidate in ("abs_frame", "frame_idx", "rel_frame", "frame"):
        if candidate in reference_tracks.columns:
            frame_col = candidate
            break
    if frame_col is None:
        return None
    interval_rows = reference_tracks[(reference_tracks[frame_col] >= start_frame) & (reference_tracks[frame_col] <= end_frame)]
    if interval_rows.empty:
        return None
    id_col = _infer_id_column(interval_rows)
    return int(interval_rows[id_col].nunique())


def save_count_metrics(metrics: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output_path
