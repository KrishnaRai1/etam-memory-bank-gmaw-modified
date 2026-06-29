from __future__ import annotations

from typing import Any

import pandas as pd


SEMANTIC_DROPLET_CLASS = 3
PREDICTION_DROPLET_CLASS = 1
REFERENCE_DROPLET_CLASS = 3


def _infer_class_column(df: pd.DataFrame) -> str | None:
    for candidate in ("class_id", "cls_id"):
        if candidate in df.columns:
            return candidate
    return None


def normalize_semantic_class_ids(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    class_col = _infer_class_column(normalized)
    if class_col is None:
        return normalized

    semantic_col = "semantic_class_id"
    if semantic_col in normalized.columns:
        return normalized

    normalized[semantic_col] = normalized[class_col].astype(object)
    normalized[semantic_col] = normalized[semantic_col].map(
        lambda value: SEMANTIC_DROPLET_CLASS if int(value) in {PREDICTION_DROPLET_CLASS, REFERENCE_DROPLET_CLASS} else int(value)
    )
    return normalized


def filter_semantic_droplets(df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_semantic_class_ids(df)
    class_col = _infer_class_column(normalized)
    if class_col is None:
        return normalized

    if "semantic_class_id" in normalized.columns:
        return normalized[normalized["semantic_class_id"] == SEMANTIC_DROPLET_CLASS].copy()

    return normalized[normalized[class_col] == SEMANTIC_DROPLET_CLASS].copy()


def explain_class_alignment(df: pd.DataFrame) -> dict[str, Any]:
    normalized = normalize_semantic_class_ids(df)
    class_col = _infer_class_column(normalized)
    if class_col is None:
        return {"class_column": None, "raw_class_ids": [], "semantic_class_ids": []}

    return {
        "class_column": class_col,
        "raw_class_ids": sorted({int(value) for value in normalized[class_col].dropna().tolist()}),
        "semantic_class_ids": sorted({int(value) for value in normalized["semantic_class_id"].dropna().tolist()}),
    }
