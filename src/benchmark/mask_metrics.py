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
class MaskObject:
    frame_idx: int
    obj_id: int | str
    mask_px: list[int] | None = None
    ys: np.ndarray | None = None
    xs: np.ndarray | None = None
    centroid_x: float | None = None
    centroid_y: float | None = None
    area: int | None = None


def _infer_frame_column(df: pd.DataFrame) -> str:
    for candidate in ("frame_idx", "abs_frame", "rel_frame", "frame"):
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Mask dataframe is missing a frame index column. Found columns: {list(df.columns)}")


def _infer_id_column(df: pd.DataFrame) -> str:
    for candidate in ("global_id", "id", "track_id", "droplet_id", "obj_id"):
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Mask dataframe is missing an ID column. Found columns: {list(df.columns)}")


def _infer_class_column(df: pd.DataFrame) -> str | None:
    for candidate in ("class_id", "cls_id"):
        if candidate in df.columns:
            return candidate
    return None


def _row_mask_coords(row: pd.Series, width: int | None = None) -> set[tuple[int, int]]:
    if "ys" in row.index and "xs" in row.index and row["ys"] is not None and row["xs"] is not None:
        ys = np.asarray(row["ys"])
        xs = np.asarray(row["xs"])
        return set(zip(ys.astype(int).tolist(), xs.astype(int).tolist()))

    if "mask_px" in row.index and row["mask_px"] is not None:
        if width is None:
            raise ValueError("Width is required to convert mask_px to coordinate pairs.")
        px = np.asarray(row["mask_px"])
        ys = (px // width).astype(int)
        xs = (px % width).astype(int)
        return set(zip(ys.tolist(), xs.tolist()))

    raise ValueError(f"Mask row does not contain usable mask data. Columns: {list(row.index)}")


def _mask_centroid(row: pd.Series) -> tuple[float, float] | None:
    if "centroid_x" in row.index and "centroid_y" in row.index and row["centroid_x"] is not None and row["centroid_y"] is not None:
        return float(row["centroid_x"]), float(row["centroid_y"])
    if "ys" in row.index and "xs" in row.index and row["ys"] is not None and row["xs"] is not None:
        ys = np.asarray(row["ys"])
        xs = np.asarray(row["xs"])
        if ys.size and xs.size:
            return float(xs.mean()), float(ys.mean())
    return None


def _pairwise_stats(ref_mask: set[tuple[int, int]], pred_mask: set[tuple[int, int]]) -> dict[str, Any]:
    if not ref_mask or not pred_mask:
        return {"iou": 0.0, "dice": 0.0, "intersection": 0, "union": len(ref_mask | pred_mask)}
    intersection = len(ref_mask & pred_mask)
    union = len(ref_mask | pred_mask)
    iou = float(intersection / union) if union > 0 else 0.0
    dice = float((2 * intersection) / (len(ref_mask) + len(pred_mask))) if (len(ref_mask) + len(pred_mask)) > 0 else 0.0
    return {"iou": iou, "dice": dice, "intersection": intersection, "union": union}


def _build_frame_objects(df: pd.DataFrame, width: int | None = None, label_prefix: str = "") -> dict[int, list[MaskObject]]:
    frame_col = _infer_frame_column(df)
    id_col = _infer_id_column(df)
    objs: dict[int, list[MaskObject]] = defaultdict(list)
    for _, row in df.iterrows():
        frame_idx = int(row[frame_col])
        centroid = _mask_centroid(row)
        area = None
        if "area_px" in row.index and row["area_px"] is not None:
            area = int(row["area_px"])
        mask_obj = MaskObject(
            frame_idx=frame_idx,
            obj_id=int(row[id_col]) if pd.notna(row[id_col]) else row.get("local_id", row.get("obj_id", "unknown")),
            mask_px=row.get("mask_px") if "mask_px" in row.index else None,
            ys=np.asarray(row["ys"]).astype(int) if "ys" in row.index and row["ys"] is not None else None,
            xs=np.asarray(row["xs"]).astype(int) if "xs" in row.index and row["xs"] is not None else None,
            centroid_x=float(centroid[0]) if centroid is not None else None,
            centroid_y=float(centroid[1]) if centroid is not None else None,
            area=int(area) if area is not None else None,
        )
        objs[frame_idx].append(mask_obj)
    return objs


def _filter_droplets(df: pd.DataFrame) -> pd.DataFrame:
    return filter_semantic_droplets(df)


def _frame_alignment(ref_frames: set[int], pred_frames: set[int]) -> dict[str, Any]:
    if not ref_frames or not pred_frames:
        return {
            "reference_frame_count": len(ref_frames),
            "predicted_frame_count": len(pred_frames),
            "offset": None,
            "matched": False,
            "issues": ["missing_reference_or_predicted_frames"],
        }
    ref_sorted = sorted(ref_frames)
    pred_sorted = sorted(pred_frames)
    offsets = []
    for ref_frame, pred_frame in zip(ref_sorted[:min(len(ref_sorted), len(pred_sorted), 10)], pred_sorted[:min(len(ref_sorted), len(pred_sorted), 10)]):
        offsets.append(int(pred_frame - ref_frame))
    offset = int(round(float(np.median(offsets)))) if offsets else 0
    matched = offset == 0
    issues = [] if matched else [f"frame_offset_detected:{offset}"]
    return {
        "reference_frame_count": len(ref_frames),
        "predicted_frame_count": len(pred_frames),
        "offset": offset,
        "matched": matched,
        "issues": issues,
    }


def compute_mask_metrics(reference_masks: pd.DataFrame, predicted_masks: pd.DataFrame, width: int | None = None, interval_frames: set[int] | None = None) -> dict[str, Any]:
    ref_masks = _filter_droplets(reference_masks)
    if interval_frames is not None:
        frame_col = _infer_frame_column(ref_masks)
        ref_masks = ref_masks[ref_masks[frame_col].isin(interval_frames)].copy()
    if interval_frames is not None and ref_masks.empty:
        ref_masks = ref_masks.iloc[0:0].copy()
    pred_masks = _filter_droplets(predicted_masks)
    ref_objs = _build_frame_objects(ref_masks, width=width, label_prefix="ref")
    pred_objs = _build_frame_objects(pred_masks, width=width, label_prefix="pred")

    ious = []
    dices = []
    centroid_distances = []
    area_diffs = []
    matched = 0
    total_pred_masks = sum(len(v) for v in pred_objs.values())
    total_ref_masks = sum(len(v) for v in ref_objs.values())
    alignment = _frame_alignment(set(ref_objs.keys()), set(pred_objs.keys()))

    for frame_idx, preds in pred_objs.items():
        refs = ref_objs.get(frame_idx, [])
        if not refs:
            continue
        for pred in preds:
            try:
                pred_coords = _row_mask_coords(pd.Series(pred.__dict__), width=width)
            except Exception:
                continue
            best_score = -1.0
            best_stats = None
            best_ref = None
            for ref in refs:
                try:
                    ref_coords = _row_mask_coords(pd.Series(ref.__dict__), width=width)
                except Exception:
                    continue
                stats = _pairwise_stats(ref_coords, pred_coords)
                if stats["iou"] > best_score:
                    best_score = stats["iou"]
                    best_stats = stats
                    best_ref = ref
            if best_stats is None:
                continue
            matched += 1
            ious.append(best_stats["iou"])
            dices.append(best_stats["dice"])
            pred_centroid = (pred.centroid_x, pred.centroid_y)
            ref_centroid = (best_ref.centroid_x, best_ref.centroid_y)
            if pred_centroid[0] is not None and ref_centroid[0] is not None:
                centroid_distances.append(float(np.hypot(pred_centroid[0] - ref_centroid[0], pred_centroid[1] - ref_centroid[1])))
            if pred.area is not None and best_ref.area is not None:
                area_diffs.append(abs(pred.area - best_ref.area))

    metrics = {
        "reference_mask_count": total_ref_masks,
        "predicted_mask_count": total_pred_masks,
        "matched_mask_count": matched,
        "frame_alignment": alignment,
        "mean_iou": float(np.mean(ious)) if ious else None,
        "mean_dice": float(np.mean(dices)) if dices else None,
        "mean_centroid_distance": float(np.mean(centroid_distances)) if centroid_distances else None,
        "mean_area_difference": float(np.mean(area_diffs)) if area_diffs else None,
        "reference_match_rate": float(matched / total_ref_masks) if total_ref_masks else None,
        "predicted_match_rate": float(matched / total_pred_masks) if total_pred_masks else None,
    }
    return metrics


def save_mask_metrics(metrics: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output_path
