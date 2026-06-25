#!/usr/bin/env python3
"""Validate benchmark_cases.yaml schema and interval fields."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def validate_benchmark_config(path: Path) -> bool:
    if not path.exists():
        print(f"[ERROR] Config file not found: {path}")
        return False

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[ERROR] YAML load failed: {exc}")
        return False

    if not isinstance(data, dict) or "benchmark_cases" not in data:
        print("[ERROR] Expected top-level 'benchmark_cases' section.")
        return False

    benchmark_cases = data.get("benchmark_cases", {}) or {}
    if not isinstance(benchmark_cases, dict):
        print("[ERROR] 'benchmark_cases' must be a mapping of video_id -> intervals.")
        return False

    total_intervals = 0
    errors = []

    for video_id, intervals in benchmark_cases.items():
        if not isinstance(intervals, list):
            errors.append(f"{video_id}: intervals must be a list")
            continue

        for idx, interval in enumerate(intervals, start=1):
            if not isinstance(interval, dict):
                errors.append(f"{video_id}[{idx}]: interval must be an object")
                continue

            required = ["interval_id", "start_frame", "end_frame", "category"]
            missing = [name for name in required if name not in interval]
            if missing:
                errors.append(f"{video_id}[{idx}]: missing {', '.join(missing)}")
                continue

            start = interval.get("start_frame")
            end = interval.get("end_frame")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                errors.append(f"{video_id}[{idx}]: start_frame/end_frame must be numeric")
                continue

            if int(end) < int(start):
                errors.append(f"{video_id}[{idx}]: end_frame must be >= start_frame")
                continue

            total_intervals += 1

    if errors:
        print("[VALIDATION FAILED]")
        for item in errors:
            print("  -", item)
        return False

    print("[VALIDATION OK]")
    print(f"Videos: {len(benchmark_cases)}")
    print(f"Intervals: {total_intervals}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate benchmark_cases.yaml")
    parser.add_argument("--config", default="configs/benchmark_cases.yaml", help="Path to benchmark_cases.yaml")
    args = parser.parse_args()

    ok = validate_benchmark_config(Path(args.config).expanduser().resolve())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
