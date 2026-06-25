from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _experiment_log_path(run_dir: Path) -> Path | None:
    logs = sorted(run_dir.glob("**/experiment_*.json"))
    return logs[-1] if logs else None


def aggregate_results(benchmark_runs_root: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    benchmark_runs_root = Path(benchmark_runs_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    rows: list[dict[str, Any]] = []

    for skip_dir in sorted(benchmark_runs_root.rglob("skip_*")):
        if not skip_dir.is_dir():
            continue
        interval_id = skip_dir.parent.name
        video_id = skip_dir.parent.parent.name
        count_metrics = _load_json(skip_dir / "count_metrics.json") or {}
        mask_metrics = _load_json(skip_dir / "mask_metrics.json") or {}
        track_metrics = _load_json(skip_dir / "track_metrics.json") or {}
        exp_log_path = _experiment_log_path(skip_dir)
        exp_metrics = _load_json(exp_log_path) if exp_log_path else {}

        row = {
            "video_id": video_id,
            "interval_id": interval_id,
            "category": exp_metrics.get("benchmark_category"),
            "memory_update_skip": int(exp_metrics.get("memory_update_skip", 0) or 0),
            "total_runtime": float(exp_metrics.get("total_runtime", 0.0) or 0.0),
            "stage3_runtime": float(exp_metrics.get("stage3_runtime", 0.0) or 0.0),
            "runtime_per_frame": float(exp_metrics.get("runtime_per_frame", 0.0) or 0.0),
            "propagation_calls": int(exp_metrics.get("stage3_propagation_calls", 0) or 0),
            "cache_hits": int(exp_metrics.get("stage3_cache_hits", 0) or 0),
            "cache_misses": int(exp_metrics.get("stage3_cache_misses", 0) or 0),
            "computed_frames": int(exp_metrics.get("stage3_computed_frames", 0) or 0),
            "new_seeds": int(exp_metrics.get("stage3_new_seeds", 0) or 0),
            "droplet_count": int(exp_metrics.get("droplet_count", 0) or 0),
            "reference_droplet_count": int(count_metrics.get("reference_droplet_count", 0) or 0),
            "tracked_droplet_count": int(count_metrics.get("tracked_droplet_count", 0) or 0),
            "count_error": int(count_metrics.get("count_error", 0) or 0),
            "absolute_error": int(count_metrics.get("absolute_error", 0) or 0),
            "relative_error": float(count_metrics.get("relative_error", 0.0) or 0.0),
            "mean_iou": float(mask_metrics.get("mean_iou", 0.0) or 0.0),
            "mean_dice": float(mask_metrics.get("mean_dice", 0.0) or 0.0),
            "mean_centroid_distance": float(mask_metrics.get("mean_centroid_distance", 0.0) or 0.0),
            "mean_area_difference": float(mask_metrics.get("mean_area_difference", 0.0) or 0.0),
            "avg_centroid_deviation": float(track_metrics.get("avg_centroid_deviation", 0.0) or 0.0),
            "trajectory_deviation": float(track_metrics.get("trajectory_deviation", 0.0) or 0.0),
            "track_continuity": float(track_metrics.get("track_continuity", 0.0) or 0.0),
            "experiment_log": str(exp_log_path) if exp_log_path else None,
            "count_metrics": str(skip_dir / "count_metrics.json") if count_metrics else None,
            "mask_metrics": str(skip_dir / "mask_metrics.json") if mask_metrics else None,
            "track_metrics": str(skip_dir / "track_metrics.json") if track_metrics else None,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "benchmark_summary.csv"
        json_path = output_dir / "benchmark_summary.json"
        df.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps({"benchmark_summary": []}, indent=2), encoding="utf-8")
        return csv_path, json_path

    baseline = df["total_runtime"].replace(0, float("nan")).min()
    if pd.notna(baseline):
        df["speedup_vs_baseline"] = df["total_runtime"].apply(lambda x: round(baseline / x, 3) if x > 0 else 0.0)
    else:
        df["speedup_vs_baseline"] = 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_summary.csv"
    json_path = output_dir / "benchmark_summary.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"benchmark_summary": df.to_dict(orient="records")}, indent=2), encoding="utf-8")
    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate benchmark results from benchmark run directories.")
    parser.add_argument("--benchmark-runs", default="outputs/benchmark_runs", help="Benchmark run root directory")
    parser.add_argument("--summary-dir", default="outputs/benchmark_summary", help="Directory where summary CSV/JSON will be written")
    args = parser.parse_args()

    csv_path, json_path = aggregate_results(args.benchmark_runs, args.summary_dir)
    print(f"[Benchmark] Aggregated summary written to:\n  - {csv_path}\n  - {json_path}")


if __name__ == "__main__":
    main()
