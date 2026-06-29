from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.benchmark.ontology import filter_semantic_droplets


@dataclass
class TrackPoint:
    frame_idx: int
    track_id: int | str
    centroid_x: float
    centroid_y: float


def _infer_frame_column(df: pd.DataFrame) -> str:
    for candidate in ("frame_idx", "abs_frame", "rel_frame", "frame"):
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Track dataframe is missing a frame index column. Found columns: {list(df.columns)}")


def _infer_id_column(df: pd.DataFrame) -> str:
    for candidate in ("id", "track_id", "droplet_id", "obj_id", "global_id"):
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Track dataframe is missing an ID column. Found columns: {list(df.columns)}")


def _build_points(df: pd.DataFrame) -> dict[int, list[TrackPoint]]:
    frame_col = _infer_frame_column(df)
    id_col = _infer_id_column(df)
    points: dict[int, list[TrackPoint]] = defaultdict(list)

    for _, row in df.iterrows():
        if "centroid_x" not in row.index or "centroid_y" not in row.index:
            continue
        if pd.isna(row["centroid_x"]) or pd.isna(row["centroid_y"]):
            continue
        points[int(row[frame_col])].append(
            TrackPoint(
                frame_idx=int(row[frame_col]),
                track_id=int(row[id_col]) if pd.notna(row[id_col]) else row[id_col],
                centroid_x=float(row["centroid_x"]),
                centroid_y=float(row["centroid_y"]),
            )
        )
    return points


def _compute_continuity(df: pd.DataFrame) -> float:
    id_col = _infer_id_column(df)
    frame_col = _infer_frame_column(df)
    continuity_scores = []
    for track_id, group in df.groupby(id_col, sort=False):
        frames = sorted(int(f) for f in group[frame_col].unique())
        if len(frames) <= 1:
            continuity_scores.append(1.0)
            continue
        gaps = sum(1 for a, b in zip(frames, frames[1:]) if b - a > 1)
        continuity_scores.append(max(0.0, 1.0 - (gaps / (len(frames) - 1))))
    return float(np.mean(continuity_scores)) if continuity_scores else 0.0


def _filter_droplets(df: pd.DataFrame) -> pd.DataFrame:
    return filter_semantic_droplets(df)


def compute_track_metrics(reference_tracks: pd.DataFrame, predicted_tracks: pd.DataFrame, interval_frames: set[int] | None = None) -> dict[str, Any]:
    reference_tracks = _filter_droplets(reference_tracks)
    if interval_frames is not None:
        frame_col = _infer_frame_column(reference_tracks)
        reference_tracks = reference_tracks[reference_tracks[frame_col].isin(interval_frames)].copy()
    if interval_frames is not None and reference_tracks.empty:
        reference_tracks = reference_tracks.iloc[0:0].copy()
    predicted_tracks = _filter_droplets(predicted_tracks)
    ref_points = _build_points(reference_tracks)
    pred_points = _build_points(predicted_tracks)
    all_distances = []
    all_mismatches = []
    matched_frames = 0
    total_frames = 0

    for frame_idx, preds in pred_points.items():
        refs = ref_points.get(frame_idx, [])
        if not refs:
            continue
        total_frames += len(preds)
        for pred in preds:
            best_dist = float("inf")
            if not refs:
                continue
            for ref in refs:
                dist = float(np.hypot(pred.centroid_x - ref.centroid_x, pred.centroid_y - ref.centroid_y))
                if dist < best_dist:
                    best_dist = dist
            if best_dist < float("inf"):
                all_distances.append(best_dist)
                matched_frames += 1

    track_continuity = _compute_continuity(predicted_tracks)
    avg_centroid_deviation = float(np.mean(all_distances)) if all_distances else 0.0
    trajectory_deviation = float(1.0 - (matched_frames / total_frames)) if total_frames else 0.0

    return {
        "reference_track_count": int(reference_tracks[_infer_id_column(reference_tracks)].nunique()) if not reference_tracks.empty else 0,
        "predicted_track_count": int(predicted_tracks[_infer_id_column(predicted_tracks)].nunique()) if not predicted_tracks.empty else 0,
        "avg_centroid_deviation": avg_centroid_deviation,
        "trajectory_deviation": trajectory_deviation,
        "track_continuity": track_continuity,
    }


def save_track_metrics(metrics: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output_path
