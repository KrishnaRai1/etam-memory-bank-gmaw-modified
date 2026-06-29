from __future__ import annotations

from typing import Iterable

import pandas as pd


def infer_frame_offset(predicted_frames: Iterable[int] | None, reference_frames: Iterable[int] | None, preferred_offset: int | None = None) -> int:
    """Infer a shared frame offset that best aligns prediction frames with reference frames."""

    pred_values = sorted({int(frame) for frame in (predicted_frames or []) if pd.notna(frame)})
    ref_values = sorted({int(frame) for frame in (reference_frames or []) if pd.notna(frame)})

    if not pred_values or not ref_values:
        return int(preferred_offset) if preferred_offset is not None else 0

    ref_set = set(ref_values)
    best_offset: int | None = None
    best_overlap = -1

    for pred_frame in pred_values:
        for ref_frame in ref_values:
            candidate_offset = int(ref_frame - pred_frame)
            overlap = sum(1 for frame in pred_values if (frame + candidate_offset) in ref_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_offset = candidate_offset
            elif overlap == best_overlap and best_offset is not None and abs(candidate_offset) < abs(best_offset):
                best_offset = candidate_offset

    if best_offset is None or best_overlap <= 0:
        return int(preferred_offset) if preferred_offset is not None else 0

    return int(best_offset)


def remap_prediction_frames_to_reference(
    df: pd.DataFrame,
    frame_offset: int | None = None,
    reference_frames: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Map interval-local prediction frames into the reference frame coordinate system.

    The benchmark runner extracts a subclip of frames and then invokes the pipeline with
    frame_start=0, so the pipeline writes local frame indices (relative to the subclip)
    into the prediction parquet. The evaluation layer must add an offset to align those
    local indices with the original reference frame numbers.
    """

    if df.empty:
        return df.copy()

    remapped = df.copy()

    if frame_offset is None:
        frame_offset = infer_frame_offset(
            predicted_frames=[int(v) for v in remapped.get("abs_frame", remapped.get("rel_frame", [])) if pd.notna(v)] if "abs_frame" in remapped.columns or "rel_frame" in remapped.columns else None,
            reference_frames=reference_frames,
        )

    frame_offset = int(frame_offset)

    for col in ("abs_frame", "rel_frame", "frame_idx", "frame"):
        if col in remapped.columns:
            remapped[col] = remapped[col].astype(int) + frame_offset

    if "frame_idx" in remapped.columns and "abs_frame" in remapped.columns:
        remapped["frame_idx"] = remapped["abs_frame"]

    return remapped
