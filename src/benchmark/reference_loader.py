from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.benchmark.processed_dataset_loader import discover_processed_dataset


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Reference file not found: {path}")
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Unable to load parquet file {path}: {exc}") from exc
    return df


def _print_summary(name: str, df: pd.DataFrame, path: Path) -> None:
    print(f"[Reference] Loaded {name}: {path}")
    print(f"           rows={len(df)}, cols={len(df.columns)}")
    print(f"           columns={list(df.columns)}")


def _validate_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Reference file {path} is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _resolve_reference_dir(reference_root: Path, video_id: str | None = None) -> Path:
    candidates = [reference_root]
    if video_id:
        candidates += [reference_root / f"{video_id}_data", reference_root / video_id, reference_root / video_id / "data"]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir() and (
            (candidate / "tracks_clean.parquet").exists() or (candidate / "seg_masks.parquet").exists()
        ):
            return candidate

    if video_id:
        discovered = discover_processed_dataset(reference_root, video_id=video_id)
        if discovered.get("reference_dir"):
            resolved = Path(discovered["reference_dir"])
            if resolved.exists():
                return resolved

    raise FileNotFoundError(
        f"Could not resolve reference dataset directory from {reference_root} "
        f"for video_id={video_id}. Checked: {candidates}"
    )


def _infer_id_column(df: pd.DataFrame) -> str:
    for candidate in ("id", "track_id", "droplet_id", "obj_id"):
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Reference dataframe is missing an ID column. Found columns: {list(df.columns)}")


def _infer_frame_column(df: pd.DataFrame) -> str:
    for candidate in ("frame_idx", "abs_frame", "rel_frame", "frame"):
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Reference dataframe is missing a frame index column. Found columns: {list(df.columns)}")


def load_tracks_clean(reference_root: str | Path, video_id: str | None = None) -> pd.DataFrame:
    reference_root = Path(reference_root).expanduser().resolve()
    reference_dir = _resolve_reference_dir(reference_root, video_id)
    path = reference_dir / "tracks_clean.parquet"
    df = _load_parquet(path)
    required = ["id", "frame_idx"]
    if "frame_idx" not in df.columns and "abs_frame" in df.columns:
        df = df.rename(columns={"abs_frame": "frame_idx"})
    _validate_columns(df, required, path)
    _print_summary("tracks_clean", df, path)
    return df


def load_seg_masks(reference_root: str | Path, video_id: str | None = None) -> pd.DataFrame:
    reference_root = Path(reference_root).expanduser().resolve()
    reference_dir = _resolve_reference_dir(reference_root, video_id)
    path = reference_dir / "seg_masks.parquet"
    df = _load_parquet(path)
    if "frame_idx" not in df.columns and "abs_frame" in df.columns:
        df = df.rename(columns={"abs_frame": "frame_idx"})

    if not any(col in df.columns for col in ("mask_px", "ys", "xs")):
        raise ValueError(
            f"Reference seg_masks.parquet {path} must contain 'mask_px' or both 'ys' and 'xs'. "
            f"Found columns: {list(df.columns)}"
        )

    required = ["frame_idx"]
    _validate_columns(df, required, path)
    _print_summary("seg_masks", df, path)
    return df


def load_daq(reference_root: str | Path, video_id: str | None = None) -> pd.DataFrame:
    reference_root = Path(reference_root).expanduser().resolve()
    reference_dir = _resolve_reference_dir(reference_root, video_id)
    path = reference_dir / "daq.parquet"
    df = _load_parquet(path)
    _print_summary("daq", df, path)
    return df


def summary(reference_root: str | Path, video_id: str | None = None) -> dict[str, int]:
    summary_data: dict[str, int] = {}
    tracks = load_tracks_clean(reference_root, video_id)
    seg = load_seg_masks(reference_root, video_id)
    summary_data["tracks_rows"] = len(tracks)
    summary_data["seg_masks_rows"] = len(seg)
    summary_data["reference_videos"] = 1
    return summary_data
