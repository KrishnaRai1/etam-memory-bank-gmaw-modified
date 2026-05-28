#!/usr/bin/env python3
"""Generate a robust final benchmark summary from experiment_*.json logs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = REPO_ROOT / "outputs"
SUMMARY_CSV = REPO_ROOT / "final_benchmark_summary.csv"


def _collect_experiment_logs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("experiment_*.json"))


def main() -> None:
    log_files = _collect_experiment_logs(OUTPUTS_ROOT)

    if not log_files:
        print("[WARN] No experiment logs found under outputs/**/experiment_*.json; benchmark summary was not generated.")
        if SUMMARY_CSV.exists():
            SUMMARY_CSV.unlink()
        return

    rows = []
    for path in log_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[WARN] Skipping unreadable log {path}: {exc}")
            continue

        rows.append({
            "log_path": str(path),
            "interval_id": data.get("benchmark_interval", "N/A"),
            "category": data.get("benchmark_category", "N/A"),
            "frame_count": int(data.get("frame_count", 0) or 0),
            "droplet_count": int(data.get("droplet_count", 0) or 0),
            "manual_count": int(data.get("benchmark_manual_count", -1) or -1),
            "total_runtime_sec": float(data.get("total_runtime", 0.0) or 0.0),
            "stage3_runtime_sec": float(data.get("stage3_runtime", 0.0) or 0.0),
            "stage3_cache_hits": int(data.get("stage3_cache_hits", 0) or 0),
            "stage3_cache_misses": int(data.get("stage3_cache_misses", 0) or 0),
            "stage3_propagation_calls": int(data.get("stage3_propagation_calls", 0) or 0),
            "memory_update_skip": int(data.get("memory_update_skip", 1) or 1),
        })

    if not rows:
        print("[WARN] No valid experiment logs were found; benchmark summary was not generated.")
        if SUMMARY_CSV.exists():
            SUMMARY_CSV.unlink()
        return

    df = pd.DataFrame(rows)
    baseline_runtime = float(df["total_runtime_sec"].min()) if not df.empty else 0.0
    df["speedup_vs_baseline"] = df["total_runtime_sec"].apply(
        lambda x: round(baseline_runtime / x, 3) if x > 0 else 0.0
    )

    # Keep the summary table easy to inspect in research runs.
    summary = df[[
        "interval_id",
        "category",
        "frame_count",
        "droplet_count",
        "manual_count",
        "total_runtime_sec",
        "stage3_runtime_sec",
        "stage3_cache_hits",
        "stage3_cache_misses",
        "stage3_propagation_calls",
        "memory_update_skip",
        "speedup_vs_baseline",
        "log_path",
    ]].copy()

    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"[BENCHMARK] Wrote final summary to {SUMMARY_CSV}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
