#!/usr/bin/env python3
# tools/count_droplets.py
# -*- coding: utf-8 -*-
# Counts droplets from a finalised final/ directory.
#
# Reads tracks_clean.parquet (preferred) or falls back to tracks_merged.parquet
# when the postprocess stage was skipped. Total count = number of unique IDs.
# Writes:
#   - droplet_count_summary.json   : total + per-ID first/last frame
#   - droplet_count_per_frame.csv  : frame_idx, active, cumulative_new

from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd


def _load_tracks(final_dir: Path) -> pd.DataFrame:
    # tracks_clean comes from the postprocess step; merged is the raw Stage-3 output.
    for name in ("tracks_clean.parquet", "tracks_merged.parquet"):
        p = final_dir / name
        if p.exists():
            df = pd.read_parquet(p)
            df["frame_idx"] = df["frame_idx"].astype("int64")
            df["id"]        = df["id"].astype("int64")
            print(f"[count] using {p.name} — rows={len(df)}")
            return df
    raise FileNotFoundError(f"No tracks_clean.parquet or tracks_merged.parquet in {final_dir}")


def _total_frames(final_dir: Path, df: pd.DataFrame) -> int:
    # Prefer the canonical frame count from frames_meta.json; fall back to the
    # highest frame_idx seen in tracks (which underestimates if the last frames
    # had no detections).
    meta_p = final_dir / "frames_meta.json"
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text())
            names = meta.get("frame_names") or []
            if names:
                return len(names)
        except Exception:
            pass
    return int(df["frame_idx"].max()) + 1 if not df.empty else 0


def main():
    ap = argparse.ArgumentParser("Count droplets from tracks parquet")
    ap.add_argument("--final-dir", type=str, required=True)
    args = ap.parse_args()

    final_dir = Path(args.final_dir).expanduser().resolve()
    df = _load_tracks(final_dir)

    if df.empty:
        summary = {"total_droplets": 0, "by_id": {}}
        (final_dir / "droplet_count_summary.json").write_text(json.dumps(summary, indent=2))
        pd.DataFrame(columns=["frame_idx", "active", "cumulative_new"]) \
          .to_csv(final_dir / "droplet_count_per_frame.csv", index=False)
        print("[count] no tracks; wrote empty summary")
        return

    n_frames = _total_frames(final_dir, df)

    # Track span (first/last frame, length) per ID.
    spans = df.groupby("id")["frame_idx"].agg(["min", "max", "count"]).reset_index()
    spans = spans.rename(columns={"min": "first_frame", "max": "last_frame", "count": "n_frames"})
    by_id = {
        int(r["id"]): {
            "first_frame": int(r["first_frame"]),
            "last_frame":  int(r["last_frame"]),
            "n_frames":    int(r["n_frames"]),
        }
        for _, r in spans.iterrows()
    }

    # Per-frame counts:
    #   active        = distinct IDs present at frame f
    #   cumulative_new = distinct IDs whose first_frame <= f (running total)
    starts_sorted = np.sort(spans["first_frame"].to_numpy(dtype=np.int64))
    rows = []
    for f in range(n_frames):
        active = int(df.loc[df["frame_idx"] == f, "id"].nunique())
        cum    = int(np.searchsorted(starts_sorted, f, side="right"))
        rows.append({"frame_idx": f, "active": active, "cumulative_new": cum})
    per_frame = pd.DataFrame(rows)
    per_frame.to_csv(final_dir / "droplet_count_per_frame.csv", index=False)

    summary = {
        "total_droplets": int(spans.shape[0]),
        "n_frames":       int(n_frames),
        "by_id":          by_id,
    }
    (final_dir / "droplet_count_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== DROPLET COUNT ===")
    print(f"  total unique droplets = {summary['total_droplets']}")
    print(f"  n frames              = {summary['n_frames']}")
    print(f"  summary  → {final_dir/'droplet_count_summary.json'}")
    print(f"  per-frame → {final_dir/'droplet_count_per_frame.csv'}")


if __name__ == "__main__":
    main()
