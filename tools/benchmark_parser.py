#!/usr/bin/env python3
"""Robust benchmark annotation parser for CSV/XLSX inputs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


CATEGORY_KEYWORDS = {
    "segmentation_errors": ["segmentation", "mask", "merge", "pool", "outline"],
    "id_switches": ["id switch", "switch", "switches", "identity shift"],
    "counting_failures": ["count", "counting", "number of droplets"],
    "difficult_motion": ["difficult", "motion", "fast motion", "blur"],
    "false_positive_spatter": ["false droplet", "spatter", "false positive"],
    "incorrect_id_assignment": ["incorrect id", "incorrectly assigned", "assigned to a"],
}


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_header(name: Any) -> str:
    text = normalize_text(name).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def parse_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        text = normalize_text(value)
        if not text:
            return None
        return int(float(text.replace(",", "")))
    except Exception:
        return None


def detect_header_row(df: pd.DataFrame) -> int:
    alias_sets = {
        "video_id": {"video_id", "videoid", "video"},
        "frame_range": {"frame_number", "frame_range", "frame", "frames"},
        "error_description": {"error_type", "error", "description", "issue"},
        "notes": {"notes", "notes_optional", "comment", "comments"},
        "manual_count": {"manual_count", "manually_counted", "manually_counted_droplet", "manual"},
        "system_count": {"system_count", "system_counted", "system_counted_droplet", "system"},
    }
    all_aliases = set().union(*alias_sets.values())

    best_score = -1
    best_idx = 0

    for idx, row in df.iterrows():
        normalized = [normalize_header(v) for v in row.tolist()]
        score = sum(1 for name in normalized if name in all_aliases)

        # Prefer rows that contain the main annotation fields rather than spreadsheet title text.
        has_main_fields = any(name in alias_sets["video_id"] for name in normalized) and any(
            name in alias_sets["frame_range"] for name in normalized
        )
        if has_main_fields and score >= 4:
            return idx

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def infer_columns(df: pd.DataFrame) -> dict[str, str]:
    normalized = {}
    for column in df.columns:
        key = normalize_header(column)
        if key not in normalized:
            normalized[key] = column
    candidates = {
        "video_id": ("video_id", "videoid", "video", "clip_id"),
        "frame_range": ("frame_number", "frame_range", "frame", "frames"),
        "error_description": ("error_type", "error", "description", "issue"),
        "notes": ("notes", "notes_optional", "comment", "comments"),
        "manual_count": ("manual_count", "manually_counted_droplet", "manually_counted", "manual"),
        "system_count": ("system_count", "system_counted_droplet", "system_counted", "system"),
    }

    mapping = {}
    for target, aliases in candidates.items():
        for alias in aliases:
            if alias in normalized:
                mapping[target] = normalized[alias]
                break
    return mapping


def load_annotations(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(input_path, header=None, engine="openpyxl")
        except Exception as exc:
            raise RuntimeError(f"Failed to read Excel annotation file: {exc}") from exc

    if suffix == ".csv":
        try:
            return pd.read_csv(
                input_path,
                header=None,
                encoding="utf-8",
                encoding_errors="replace",
                on_bad_lines="skip",
                engine="python",
                dtype=str,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to read CSV annotation file: {exc}") from exc

    raise ValueError(f"Unsupported annotation format: {input_path.suffix}")


def infer_category(error_text: str, notes_text: str) -> str:
    combined = f"{error_text} {notes_text}".lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in combined for keyword in keywords):
            return category
    return "uncategorized"


def extract_frame_range(frame_text: str) -> tuple[int, int] | None:
    text = normalize_text(frame_text)
    if not text:
        return None
    match = re.search(r"(\d+)\s*(?:-|to|–|—)\s*(\d+)", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_annotations(input_path: Path) -> dict[str, list[dict[str, Any]]]:
    df = load_annotations(input_path)
    if df.empty:
        return {}

    header_row = detect_header_row(df)
    if header_row > 0:
        header_values = [normalize_text(v) for v in df.iloc[header_row].tolist()]
        df = df.iloc[header_row + 1 :].copy()
        df.columns = header_values

    columns = infer_columns(df)
    if not columns:
        raise ValueError("Could not identify annotation columns from the input file.")

    benchmark_dict: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for idx, row in df.iterrows():
        row_values = {name: row.get(col, "") for name, col in columns.items()}
        video_id = normalize_text(row_values.get("video_id", ""))
        frame_text = normalize_text(row_values.get("frame_range", ""))
        error_text = normalize_text(row_values.get("error_description", ""))
        notes_text = normalize_text(row_values.get("notes", ""))
        manual_count = parse_int(row_values.get("manual_count", ""))
        system_count = parse_int(row_values.get("system_count", ""))

        if not video_id or not frame_text:
            continue

        frame_range = extract_frame_range(frame_text)
        if frame_range is None:
            print(f"[WARN] Skipping row {idx}: invalid frame range -> {frame_text}")
            continue

        start_frame, end_frame = frame_range
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame

        interval_data = {
            "interval_id": f"{video_id}_{start_frame}_{end_frame}",
            "start_frame": start_frame,
            "end_frame": end_frame,
            "category": infer_category(error_text, notes_text),
            "error_description": error_text,
            "notes": notes_text,
            "manual_count": manual_count if manual_count is not None else -1,
            "system_count": system_count if system_count is not None else -1,
        }
        benchmark_dict[video_id].append(interval_data)

    return dict(benchmark_dict)


def write_outputs(benchmark_dict: dict[str, list[dict[str, Any]]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    yaml_config = {"benchmark_cases": benchmark_dict}
    yaml_path = output_dir / "benchmark_cases.yaml"
    json_path = output_dir / "parsed_benchmark_intervals.json"

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_config, f, sort_keys=False, allow_unicode=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"benchmark_cases": benchmark_dict}, f, indent=2, ensure_ascii=False)

    return yaml_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse benchmark annotations into YAML/JSON outputs.")
    parser.add_argument("--input", "-i", default="Exploratory Data Analysis (EDA)_Data Annotation.xlsx", help="Input CSV/XLSX annotation file")
    parser.add_argument("--output-dir", "-o", default="configs", help="Directory for benchmark_cases.yaml and parsed_benchmark_intervals.json")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input annotation file not found: {input_path}")

    benchmark_dict = parse_annotations(input_path)
    yaml_path, json_path = write_outputs(benchmark_dict, output_dir)

    total_intervals = sum(len(v) for v in benchmark_dict.values())
    category_counts = Counter(item["category"] for intervals in benchmark_dict.values() for item in intervals)

    print("\n========== PARSER SUMMARY ==========")
    print(f"Input file            : {input_path}")
    print(f"Videos found          : {len(benchmark_dict)}")
    print(f"Total intervals parsed: {total_intervals}")
    print("Category counts:")
    for category, count in category_counts.most_common():
        print(f"  - {category}: {count}")
    print("Sample intervals:")
    for video_id, intervals in list(benchmark_dict.items())[:5]:
        for item in intervals[:2]:
            print(f"  - {video_id}: {item['interval_id']} -> {item['start_frame']}-{item['end_frame']} ({item['category']})")
    print(f"YAML saved            : {yaml_path}")
    print(f"JSON saved            : {json_path}")
    print("====================================")


if __name__ == "__main__":
    main()