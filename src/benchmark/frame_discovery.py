from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from src.benchmark.video_id_matcher import normalize_video_id

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _is_frame_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(p for p in path.iterdir() if _is_image_file(p))


def _frame_count(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if _is_image_file(p))


def _gather_candidates(root: Path, video_id: str, max_depth: int = 3) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    norm_video_id = normalize_video_id(video_id)

    def add(path: Path, reason: str) -> None:
        if path.exists() and path.is_dir() and _is_frame_dir(path):
            candidates.append((path.resolve(), reason))

    # direct candidate patterns
    add(root, "root itself")
    add(root / video_id, "root/video_id")
    add(root / normalize_video_id(video_id), "root/normalized_video_id")
    add(root / "frames" / video_id, "root/frames/video_id")
    add(root / "frames" / normalize_video_id(video_id), "root/frames/normalized_video_id")
    add(root / "images" / video_id, "root/images/video_id")
    add(root / "jpg" / video_id, "root/jpg/video_id")
    add(root / "image" / video_id, "root/image/video_id")
    add(root / "imgs" / video_id, "root/imgs/video_id")

    # nested search
    stack = [(root, 0)]
    seen: set[Path] = set()
    while stack:
        current, depth = stack.pop()
        if depth > max_depth or current in seen:
            continue
        seen.add(current)
        for child in current.iterdir():
            if not child.is_dir():
                continue
            if normalize_video_id(child.name) == norm_video_id:
                add(child, "nested normalized name match")
            elif video_id.lower() in child.name.lower() or child.name.lower() in video_id.lower():
                add(child, "nested name contains id")
            elif _is_frame_dir(child):
                add(child, "nested directory contains frames")
            stack.append((child, depth + 1))

    # candidate directories inside video-specific folders
    if root.name and normalize_video_id(root.name) != norm_video_id:
        possible = [root / video_id, root / normalize_video_id(video_id)]
        for candidate in possible:
            if candidate.exists() and candidate.is_dir():
                add(candidate, "named child under root")

    return candidates


def _score_candidate(candidate: Path, reason: str, video_id: str) -> int:
    score = 0
    score += _frame_count(candidate)
    if normalize_video_id(candidate.name) == normalize_video_id(video_id):
        score += 200
    if video_id.lower() in candidate.name.lower() or candidate.name.lower() in video_id.lower():
        score += 50
    if "normalized name match" in reason:
        score += 100
    if "contains frames" in reason:
        score += 10
    if "root itself" in reason:
        score += 20
    return score


def find_video_frame_dir(video_root: str | Path | None, reference_root: str | Path | None, video_id: str) -> tuple[Path | None, list[dict[str, str]]]:
    video_root_path = Path(video_root).expanduser().resolve() if video_root else None
    reference_root_path = Path(reference_root).expanduser().resolve() if reference_root else None
    diagnostics: list[dict[str, str]] = []
    candidates: list[tuple[Path, str]] = []

    if video_root_path and video_root_path.exists() and video_root_path.is_dir():
        candidates.extend(_gather_candidates(video_root_path, video_id))
    if reference_root_path and reference_root_path.exists() and reference_root_path.is_dir():
        candidates.extend(_gather_candidates(reference_root_path, video_id))

    unique: dict[str, tuple[Path, str]] = {}
    for path, reason in candidates:
        if str(path) not in unique:
            unique[str(path)] = (path, reason)
    candidates = list(unique.values())

    for path, reason in candidates:
        diagnostics.append({
            "video_id": video_id,
            "candidate_path": str(path),
            "reason": reason,
            "frame_count": str(_frame_count(path)),
        })

    if not candidates:
        return None, diagnostics

    candidates.sort(key=lambda item: _score_candidate(item[0], item[1], video_id), reverse=True)
    return candidates[0][0], diagnostics
