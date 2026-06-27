from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


def normalize_video_id(video_id: str | None) -> str:
    if video_id is None:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", "", str(video_id).strip().lower())
    return normalized


def _extract_digits(value: str | None) -> str:
    if value is None:
        return ""
    return "".join(re.findall(r"\d+", str(value)))


def build_video_id_map(candidates: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for candidate in candidates:
        norm = normalize_video_id(candidate)
        if norm:
            mapping[norm] = candidate
    return mapping


def find_matching_video_id(video_id: str, candidates: Iterable[str]) -> str | None:
    norm = normalize_video_id(video_id)
    if not norm:
        return None
    candidate_map = build_video_id_map(candidates)
    if norm in candidate_map:
        return candidate_map[norm]

    lower = video_id.lower()
    for candidate in candidates:
        if candidate.lower() == lower:
            return candidate

    for candidate in candidates:
        cand_norm = normalize_video_id(candidate)
        if cand_norm.startswith(norm) or norm.startswith(cand_norm) or norm in cand_norm or cand_norm in norm:
            return candidate

    video_digits = _extract_digits(norm)
    if video_digits:
        for candidate in candidates:
            candidate_digits = _extract_digits(normalize_video_id(candidate))
            if candidate_digits and candidate_digits == video_digits:
                return candidate

    return None
