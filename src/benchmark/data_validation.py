from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.benchmark.processed_dataset_loader import discover_processed_datasets


@dataclass
class ValidationResult:
    ok: bool
    issues: list[str]
    details: dict[str, Any]


def _exists(p: Path) -> bool:
    return p.exists() and p.is_file()


def validate_reference_dataset(reference_dir: Path) -> ValidationResult:
    issues: list[str] = []
    details: dict[str, Any] = {}

    tracks_path = reference_dir / "tracks_clean.parquet"
    seg_path = reference_dir / "seg_masks.parquet"

    if not _exists(tracks_path):
        issues.append(f"Missing tracks_clean.parquet at {tracks_path}")
    if not _exists(seg_path):
        issues.append(f"Missing seg_masks.parquet at {seg_path}")

    if issues:
        return ValidationResult(ok=False, issues=issues, details={})

    try:
        tracks = pd.read_parquet(tracks_path)
    except Exception as exc:  # pragma: no cover - defensive
        return ValidationResult(ok=False, issues=[f"Failed to read {tracks_path}: {exc}"], details={})

    try:
        segs = pd.read_parquet(seg_path)
    except Exception as exc:  # pragma: no cover - defensive
        return ValidationResult(ok=False, issues=[f"Failed to read {seg_path}: {exc}"], details={})

    # Required columns
    req_tracks = ["global_id", "abs_frame", "mask_px", "class_id"]
    missing_tracks = [c for c in req_tracks if c not in tracks.columns]
    if missing_tracks:
        issues.append(f"tracks_clean.parquet missing columns: {missing_tracks}")

    req_segs = ["mask_px", "cls_id"]
    missing_segs = [c for c in req_segs if c not in segs.columns]
    if missing_segs:
        issues.append(f"seg_masks.parquet missing columns: {missing_segs}")

    # Frame range / alignment checks
    try:
        tr_frames = sorted(pd.unique(tracks["abs_frame"]).tolist()) if "abs_frame" in tracks.columns else []
        sg_frames = sorted(pd.unique(segs["abs_frame"]).tolist()) if "abs_frame" in segs.columns else []
    except Exception:
        tr_frames = []
        sg_frames = []

    details["tracks_frame_count"] = len(tr_frames)
    details["seg_frame_count"] = len(sg_frames)
    if tr_frames and sg_frames:
        if abs(len(tr_frames) - len(sg_frames)) > max(5, 0.1 * max(len(tr_frames), len(sg_frames))):
            issues.append("Large discrepancy between tracks and seg frame coverage")

    # Minimal sanity
    if tracks.empty:
        issues.append("tracks_clean.parquet is empty")
    if segs.empty:
        issues.append("seg_masks.parquet is empty")

    return ValidationResult(ok=(len(issues) == 0), issues=issues, details=details)


def validate_interval_bounds(reference_dir: Path, start_frame: int, end_frame: int) -> ValidationResult:
    tracks_path = reference_dir / "tracks_clean.parquet"
    if not _exists(tracks_path):
        return ValidationResult(ok=False, issues=[f"Missing tracks_clean.parquet at {tracks_path}"], details={})
    try:
        tracks = pd.read_parquet(tracks_path)
    except Exception as exc:  # pragma: no cover - defensive
        return ValidationResult(ok=False, issues=[f"Failed to read {tracks_path}: {exc}"], details={})

    if "abs_frame" not in tracks.columns:
        return ValidationResult(ok=False, issues=["tracks_clean.parquet missing 'abs_frame' column"], details={})

    min_f = int(tracks["abs_frame"].min())
    max_f = int(tracks["abs_frame"].max())
    issues = []
    if start_frame < min_f or end_frame > max_f:
        issues.append(f"Interval [{start_frame},{end_frame}] outside reference frame range [{min_f},{max_f}]")
    if start_frame > end_frame:
        issues.append("Interval start_frame > end_frame")

    details = {"min_frame": min_f, "max_frame": max_f}
    return ValidationResult(ok=(len(issues) == 0), issues=issues, details=details)


def validate_processed_dataset_root(dataset_root: Path) -> ValidationResult:
    issues: list[str] = []
    details: dict[str, Any] = {"dataset_root": str(dataset_root), "videos": [], "videos_valid": 0}
    datasets = discover_processed_datasets(dataset_root)
    if not datasets:
        issues.append(f"No processed datasets discovered under {dataset_root}")
        return ValidationResult(ok=False, issues=issues, details=details)

    for dataset in datasets:
        video_id = dataset.get("video_id")
        ref_dir = Path(dataset.get("reference_dir", "")) if dataset.get("reference_dir") else None
        video_details: dict[str, Any] = {
            "video_id": video_id,
            "timestamp": dataset.get("timestamp"),
            "reference_dir": str(ref_dir) if ref_dir else None,
            "missing_files": dataset.get("missing_files", []),
            "exists": bool(dataset.get("exists")),
        }
        if not ref_dir or not ref_dir.exists():
            issues.append(f"Missing reference directory for {video_id}: {ref_dir}")
            video_details["ok"] = False
        else:
            validation = validate_reference_dataset(ref_dir)
            video_details["ok"] = validation.ok
            video_details["issues"] = validation.issues
            if not validation.ok:
                issues.extend([f"{video_id}: {issue}" for issue in validation.issues])
            else:
                details["videos_valid"] += 1
        details["videos"].append(video_details)

    return ValidationResult(ok=(len(issues) == 0), issues=issues, details=details)
