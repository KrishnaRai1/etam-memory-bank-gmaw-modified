from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src.benchmark.video_id_matcher import find_matching_video_id, normalize_video_id

REQUIRED_REFERENCE_FILES = (
    "counts.json",
    "run_summary.json",
    "tracks_clean.parquet",
    "tracks.parquet",
    "seg_masks.parquet",
    "tracking.mp4",
)


def _timestamp_sort_key(path: Path) -> tuple[Any, ...]:
    try:
        parsed = datetime.strptime(path.name, "%Y-%m-%d_%H-%M-%S")
        return (parsed.year, parsed.month, parsed.day, parsed.hour, parsed.minute, parsed.second)
    except ValueError:
        return (0, 0, 0, 0, 0, 0, path.name)


def _normalize_root(root: str | Path | None) -> Path | None:
    if root is None:
        return None
    root_path = Path(str(root)).expanduser()
    try:
        return root_path.resolve()
    except Exception:
        return root_path


def _is_reference_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return all((path / name).exists() for name in REQUIRED_REFERENCE_FILES)


def discover_processed_dataset(root: str | Path | None, video_id: str | None = None) -> dict[str, Any]:
    """Discover the latest processed dataset for one video under a processed dataset root."""
    root_path = _normalize_root(root)
    if root_path is None or not root_path.exists():
        return {
            "video_id": video_id,
            "root": str(root_path) if root_path is not None else None,
            "timestamp": None,
            "reference_dir": None,
            "reference_files": list(REQUIRED_REFERENCE_FILES),
            "found_files": [],
            "missing_files": list(REQUIRED_REFERENCE_FILES),
            "exists": False,
            "config_used_path": None,
        }

    # If the root itself contains reference files directly, treat it as a single dataset
    root_is_reference_dir = _is_reference_dir(root_path)
    if root_is_reference_dir and (video_id is None or find_matching_video_id(video_id, [root_path.name]) is not None):
        found_files = [name for name in REQUIRED_REFERENCE_FILES if (root_path / name).exists()]
        missing_files = [name for name in REQUIRED_REFERENCE_FILES if not (root_path / name).exists()]
        return {
            "video_id": root_path.name,
            "video_dir": str(root_path),
            "timestamp": None,
            "reference_dir": str(root_path),
            "reference_files": list(REQUIRED_REFERENCE_FILES),
            "found_files": found_files,
            "missing_files": missing_files,
            "exists": not missing_files,
            "config_used_path": None,
            "root": str(root_path),
        }

    if video_id:
        video_candidates = [
            p for p in sorted(root_path.iterdir(), key=lambda p: p.name.lower())
            if p.is_dir() and find_matching_video_id(video_id, [p.name]) is not None
        ]
    else:
        video_candidates = [p for p in sorted(root_path.iterdir(), key=lambda p: p.name.lower()) if p.is_dir()]

    for video_path in video_candidates:
        # support both dataset root/video_id/timestamp/final and dataset root/video_id/final structures
        timestamp_dirs = [p for p in sorted(video_path.iterdir(), key=lambda p: p.name.lower()) if p.is_dir()]
        candidate_dirs = []
        for p in timestamp_dirs:
            if p.name.lower() == "final":
                candidate_dirs.append(p)
            else:
                final_dir = p / "final"
                if final_dir.exists() and final_dir.is_dir():
                    candidate_dirs.append(final_dir)
        if not candidate_dirs:
            # also support direct output under video_path
            if _is_reference_dir(video_path):
                candidate_dirs.append(video_path)
        if not candidate_dirs:
            continue

        chosen = max(candidate_dirs, key=lambda p: _timestamp_sort_key(p.parent if p.name.lower() == "final" else p))
        reference_dir = chosen
        found_files = [name for name in REQUIRED_REFERENCE_FILES if (reference_dir / name).exists()]
        missing_files = [name for name in REQUIRED_REFERENCE_FILES if not (reference_dir / name).exists()]
        config_path = reference_dir.parent / "config_used.json"
        return {
            "video_id": video_path.name,
            "video_dir": str(video_path),
            "timestamp": str(chosen.parent.name) if chosen.parent else None,
            "reference_dir": str(reference_dir),
            "reference_files": list(REQUIRED_REFERENCE_FILES),
            "found_files": found_files,
            "missing_files": missing_files,
            "exists": not missing_files,
            "config_used_path": str(config_path) if config_path.exists() else None,
            "root": str(root_path),
        }

    return {
        "video_id": video_id,
        "root": str(root_path),
        "timestamp": None,
        "reference_dir": None,
        "reference_files": list(REQUIRED_REFERENCE_FILES),
        "found_files": [],
        "missing_files": list(REQUIRED_REFERENCE_FILES),
        "exists": False,
        "config_used_path": None,
    }


def discover_processed_datasets(root: str | Path | None) -> list[dict[str, Any]]:
    """Discover all processed datasets under a processed dataset root."""
    root_path = _normalize_root(root)
    if root_path is None or not root_path.exists():
        return []

    datasets: list[dict[str, Any]] = []
    if _is_reference_dir(root_path):
        discovery = discover_processed_dataset(root_path, video_id=root_path.name)
        if discovery.get("reference_dir"):
            datasets.append(discovery)
        return datasets

    for video_path in sorted(root_path.iterdir(), key=lambda p: p.name.lower()):
        if not video_path.is_dir():
            continue
        discovery = discover_processed_dataset(root_path, video_id=video_path.name)
        if discovery.get("reference_dir"):
            datasets.append(discovery)
    return datasets


def save_processed_dataset_inventory(root: str | Path | None, output_path: str | Path | None = None) -> dict[str, Any]:
    """Discover all datasets and write an inventory json report."""
    root_path = _normalize_root(root)
    if root_path is None:
        root_path = Path(".")
    datasets = discover_processed_datasets(root_path)
    report = {
        "dataset_root": str(root_path),
        "videos_found": len(datasets),
        "videos": datasets,
    }
    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
