#!/usr/bin/env python3
# tools/postprocess_tracks_min.py
# -*- coding: utf-8 -*-
# Domain-specific post-processing for droplet TRACKING + COUNTING.
# Everything here is OPTIONAL — none of these filters appear in the paper.
# The set of filters that actually run is controlled by the YAML `postprocess`
# section (see --cfg). With postprocess.enabled=false this script becomes a
# pass-through that just computes centroids.
#
# Reads:  final/tracks_merged.parquet  (+ frames_meta.json)
# Writes: final/tracks_clean.parquet, final/centroids_clean.parquet,
#         final/postprocess_report.json

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# ========================== CONFIG ==========================
# Defaults for every tunable. The YAML can override the ones exposed in
# _load_pp_cfg(); the rest stay as constants used by the helper functions.
NUM_WORKERS: int | str = "auto"

# Drop tracks shorter than this many frames.
MIN_LEN_FRAMES = 6

# Yo-yo / stuck detection (frame-rate-agnostic — everything is pixels/frame).
ROLL_WIN              = 7      # trailing window length
MIN_CONSEC_STABLE     = 3
DROP_WIN_MIN_PX       = 6.0    # net downward progress inside the window
UP_FRAC_MAX           = 0.25   # max fraction of upward steps allowed
EXCESS_PATH_MAX       = 2.5    # path-length / net-drop ratio cutoff
AREA_SMOOTH_WIN       = 5
YOYO_AREA_SPIKE_FACTOR = 3.0   # mask area jumps over this -> yo-yo flag
YOYO_AREA_CRASH_FACTOR = 0.20  # mask area collapses under this -> yo-yo flag

# "Stuck" rule: too small a per-frame displacement for too many frames.
MIN_MOVE_PX_PER_FRAME = 0.8
STUCK_CONSEC_FRAMES   = 3

# Falling plausibility per ID (pure pixel metrics).
FALL_MIN_NET_DROP_PX        = 20.0  # max(cy) - min(cy)
FALL_MIN_DROP_PER_FRAME_PX  = 0.2   # median(|d cy / d frame|)

# Bottom-of-frame "stuck" filter (catches IDs glued near y=H).
BOTTOM_Y_FRAC               = 0.90
BOTTOM_MIN_NET_DROP_PX      = 8.0

# Soft start: tolerate jittery readings during the first few frames of a track.
KEEP_START_PREFIX     = True
START_PREFIX_FRAMES   = 6

# Segment selection (when a single ID is split into multiple stable runs).
CONTIGUITY_MAX_GAP_FRAMES   = 3
RUN_MIN_DROP_PX             = 12.0

# Cosmetic renumbering of IDs after filtering.
REINDEX_IDS_CONTIGUOUS = True
ID_RENUMBER_START      = 1
# ============================================================


def _parse_args():
    import argparse
    ap = argparse.ArgumentParser("Postprocess tracks (min) — counting only, no velocity")
    ap.add_argument("--final-dir", type=str, required=True)
    ap.add_argument("--cfg", type=str, default=None,
                    help="Pipeline YAML; if provided, the 'postprocess' section controls which filters run.")
    return ap.parse_args()


def _load_pp_cfg(cfg_path: str | None) -> dict:
    # Read the YAML's `postprocess` section. Flags default to True so that
    # omitting --cfg (or the section itself) preserves the legacy behaviour.
    defaults = {
        "enabled": True,
        "min_len_frames": int(MIN_LEN_FRAMES),
        "yoyo_trim": True,
        "falling_filter": True,
        "bottom_stuck_filter": True,
        "reindex_ids": True,
        "fall_min_net_drop_px": float(FALL_MIN_NET_DROP_PX),
        "fall_min_drop_per_frame_px": float(FALL_MIN_DROP_PER_FRAME_PX),
        "bottom_y_frac": float(BOTTOM_Y_FRAC),
        "bottom_min_net_drop_px": float(BOTTOM_MIN_NET_DROP_PX),
    }
    if not cfg_path:
        return defaults
    p = Path(cfg_path).expanduser().resolve()
    if not p.exists():
        print(f"[postprocess] WARN: cfg {p} not found, using defaults")
        return defaults
    with open(p, "r") as f:
        cfg = yaml.safe_load(f) or {}
    user_pp = (cfg.get("postprocess") or {})
    defaults.update({k: v for k, v in user_pp.items() if k in defaults})
    return defaults


def _series_to_np(a):
    # Coerce parquet list cells (which arrive as ndarray/list/tuple) into a
    # 1-D float array with NaNs stripped.
    if a is None:
        return np.empty(0, dtype=np.float64)
    try:
        arr = np.asarray(a, dtype=np.float64) if isinstance(a, (list, tuple, np.ndarray)) \
              else np.asarray(list(a), dtype=np.float64)
    except Exception:
        return np.empty(0, dtype=np.float64)
    if arr.size:
        arr = arr[~np.isnan(arr)]
    return arr


def _load_frames_meta(final_dir: Path) -> tuple[int, int]:
    p = final_dir / "frames_meta.json"
    if not p.exists():
        raise FileNotFoundError(f"frames_meta.json not found in {final_dir}")
    meta = json.loads(p.read_text())
    return int(meta["image_size"]["H"]), int(meta["image_size"]["W"])


# -------------------- centroids (no velocity) --------------------
# Centroids per (frame, id) are needed by every downstream filter.
def _centroids_one_id(df_id: pd.DataFrame) -> pd.DataFrame:
    g = df_id.sort_values("frame_idx").drop_duplicates("frame_idx", keep="first").copy()
    cxs, cys, areas = [], [], []
    for _, r in g.iterrows():
        ys = _series_to_np(r["ys"])
        xs = _series_to_np(r["xs"])
        n = min(xs.size, ys.size)
        if n == 0:
            cxs.append(np.nan); cys.append(np.nan); areas.append(0)
        else:
            cxs.append(float(xs[:n].mean())); cys.append(float(ys[:n].mean()))
            areas.append(int(n))
    g["cx"] = cxs
    g["cy"] = cys
    g["area_px"] = areas
    return g


def _centroids_parallel(tracks_df: pd.DataFrame) -> pd.DataFrame:
    # Fan out per-ID centroid computation; fall back to serial if the pool dies.
    groups = [(k, g.copy()) for k, g in tracks_df.groupby("id", sort=False)]
    if not groups:
        return pd.DataFrame(columns=list(tracks_df.columns) + ["cx", "cy", "area_px"])
    workers = (max(1, cpu_count()) if isinstance(NUM_WORKERS, str) and NUM_WORKERS.lower() == "auto"
               else max(1, int(NUM_WORKERS)))
    try:
        out = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_centroids_one_id, g): k for k, g in groups}
            for fut in as_completed(futs):
                out.append(fut.result())
        return pd.concat(out, ignore_index=True)
    except Exception:
        return pd.concat([_centroids_one_id(g) for _, g in groups], ignore_index=True)


# -------------------- stability / yo-yo (pixel-only) --------------------
# Per-frame boolean mask saying which samples of an ID are "stable enough".
# Flags yo-yo behaviour from two angles: erratic area (spike/crash) and
# erratic motion (too many up-steps or excessive path length vs net drop).
def _stable_mask_pixel(g: pd.DataFrame) -> np.ndarray:
    g = g.sort_values("frame_idx").copy()
    n = len(g)
    if n == 0:
        return np.zeros((0,), dtype=bool)

    fi = g["frame_idx"].astype(int).to_numpy()
    cy = g["cy"].astype(float).to_numpy()
    cx = g["cx"].astype(float).to_numpy() if "cx" in g.columns else np.full(n, np.nan, dtype=float)

    # Smooth area to absorb single-frame mask jitter before comparing.
    area = g.get("area_px", pd.Series([np.nan] * n)).astype(float)
    area_sm = (area.interpolate(limit_direction="both")
                   .rolling(window=max(1, int(AREA_SMOOTH_WIN)), center=False).median()
                   .bfill().ffill())
    Amed = (area_sm.rolling(window=max(1, int(ROLL_WIN)), center=False).median()
                   .bfill().ffill().to_numpy())
    Aref = np.maximum(Amed, 1.0)
    Acur = area_sm.to_numpy()
    area_spike = (Acur >= YOYO_AREA_SPIKE_FACTOR * Aref)
    area_crash = (Acur <= YOYO_AREA_CRASH_FACTOR * Aref)

    # Per-frame deltas in pixels (positive cy means downward).
    dcy = np.zeros(n, dtype=float)
    if n >= 2:
        dcy[1:] = np.diff(cy)

    win = int(max(1, ROLL_WIN))
    net_down = np.zeros(n, dtype=float)
    up_frac  = np.zeros(n, dtype=float)
    excess   = np.zeros(n, dtype=float)
    med_dcy  = np.zeros(n, dtype=float)
    for i in range(n):
        j = max(0, i - (win - 1))
        seg_d = dcy[j+1:i+1] if i > j else np.array([], dtype=float)
        net = float(cy[i] - cy[j])
        net_down[i] = net
        up_frac[i]  = float(np.mean(seg_d < 0)) if seg_d.size else 0.0
        path_len    = float(np.sum(np.abs(seg_d))) if seg_d.size else 0.0
        excess[i]   = (path_len / max(net, 1e-9)) if net > 0 else float("inf")
        med_dcy[i]  = float(np.nanmedian(seg_d)) if seg_d.size else 0.0

    # yo-yo only counts when the window has NOT already made enough downward progress.
    yoyo_area = (area_spike | area_crash) & (net_down < DROP_WIN_MIN_PX)
    yoyo_osc  = ((excess > EXCESS_PATH_MAX) | (up_frac > UP_FRAC_MAX)) & (net_down < DROP_WIN_MIN_PX)
    yoyo = yoyo_area | yoyo_osc

    # Stuck: total step magnitude below the threshold for STUCK_CONSEC_FRAMES in a row.
    dcx = np.zeros(n, dtype=float); dcy2 = np.zeros(n, dtype=float)
    if n >= 2:
        dcx[1:]  = np.diff(cx)
        dcy2[1:] = np.diff(cy)
    step  = np.hypot(np.nan_to_num(dcx, nan=0.0), np.nan_to_num(dcy2, nan=0.0))
    moved = (step >= float(MIN_MOVE_PX_PER_FRAME))
    stuck_run = np.zeros(n, dtype=int)
    c = 0
    for i in range(n):
        c = c + 1 if not moved[i] else 0
        stuck_run[i] = c
    stuck = (stuck_run >= int(STUCK_CONSEC_FRAMES))

    # "fall_ok" overrides the stuck rule once the droplet is clearly descending.
    fall_ok = (med_dcy >= float(FALL_MIN_DROP_PER_FRAME_PX)) & (net_down >= DROP_WIN_MIN_PX) \
              | (net_down >= 2.0 * DROP_WIN_MIN_PX)
    yoyo = yoyo | (stuck & (~fall_ok) & (net_down < DROP_WIN_MIN_PX))

    # Keep the first few frames even when noisy: tracks rarely look clean at birth.
    idx = np.arange(n, dtype=int)
    prefix_ok = (idx < int(START_PREFIX_FRAMES)) if KEEP_START_PREFIX else np.zeros(n, dtype=bool)
    keep = (~yoyo) & (fall_ok | moved | prefix_ok)
    return keep


def _trim_keep_best_segment(g: pd.DataFrame) -> pd.DataFrame:
    # Split a track at unstable samples and pick the best contiguous run
    # (longest, then largest drop). Falls back to the longest raw run if no
    # candidate clears MIN_LEN_FRAMES and RUN_MIN_DROP_PX.
    if g is None or g.empty:
        return g.iloc[0:0].copy()
    g = g.sort_values("frame_idx").copy()
    n = len(g)
    fi = g["frame_idx"].astype(int).to_numpy()
    cy = g["cy"].astype(float).to_numpy()

    keep = _stable_mask_pixel(g)
    if not keep.any():
        return g.iloc[0:0].copy()

    # Walk the keep mask, splitting whenever an unstable sample appears OR
    # the gap between consecutive frame_idx exceeds CONTIGUITY_MAX_GAP_FRAMES.
    runs: list[tuple[int, int]] = []
    s = None
    for i in range(n):
        if not keep[i]:
            if s is not None:
                runs.append((s, i - 1)); s = None
            continue
        if s is None:
            s = i
        elif fi[i] - fi[i - 1] > (1 + int(CONTIGUITY_MAX_GAP_FRAMES)):
            runs.append((s, i - 1)); s = i
    if s is not None:
        runs.append((s, n - 1))
    if not runs:
        return g.iloc[0:0].copy()

    min_len = int(max(MIN_LEN_FRAMES, 3))
    qualified = []
    for a, b in runs:
        L = b - a + 1
        if L < min_len:
            continue
        drop_run = float(np.nanmax(cy[a:b+1]) - float(cy[a]))
        if drop_run < float(RUN_MIN_DROP_PX):
            continue
        qualified.append((a, b, L, drop_run, int(fi[a])))

    if not qualified:
        runs.sort(key=lambda t: (t[1] - t[0] + 1, -fi[t[0]]), reverse=True)
        a, b = runs[0]
        return g.iloc[a:b+1].copy()

    qualified.sort(key=lambda t: (t[2], t[3], -t[4]), reverse=True)
    s_best, e_best, *_ = qualified[0]
    if KEEP_START_PREFIX and keep[0] and s_best > 0:
        s_best = 0
    return g.iloc[s_best:e_best + 1].copy()


# -------------------- per-ID coarse filters --------------------
def _filter_short_tracks(df: pd.DataFrame, min_len: int):
    counts = df.groupby("id")["frame_idx"].nunique()
    keep_ids = set(counts[counts >= int(min_len)].index)
    return df[df["id"].isin(keep_ids)].copy(), set(counts.index) - keep_ids


def _filter_falling_pixels(df: pd.DataFrame):
    # Drop IDs that never show a clear downward trajectory in pixel space.
    # Two criteria must hold: total drop max(cy)-min(cy) and median per-frame
    # drop both above their thresholds.
    if df is None or df.empty:
        return df, set()
    bad, kept = set(), []
    for oid, g in df.groupby("id", sort=False):
        g = g.sort_values("frame_idx")
        cy = g["cy"].astype(float).to_numpy()
        if cy.size == 0:
            bad.add(int(oid)); continue
        drop = float(np.nanmax(cy) - np.nanmin(cy)) if np.isfinite(cy).any() else 0.0
        dcy_med = float(np.nanmedian(np.diff(cy))) if cy.size >= 2 else 0.0
        if (drop >= float(FALL_MIN_NET_DROP_PX)) and (dcy_med >= float(FALL_MIN_DROP_PER_FRAME_PX)):
            kept.append(g)
        else:
            bad.add(int(oid))
    return (pd.concat(kept, ignore_index=True) if kept else df.iloc[0:0].copy()), bad


def _filter_stuck_bottom(df: pd.DataFrame, H: int):
    # Drop IDs whose centroid sits in the bottom band of the frame for the
    # whole track without enough vertical motion — typically the weld pool
    # or a static artefact misclassified as a droplet.
    if df is None or df.empty:
        return df, set()
    bad, keep = set(), []
    y_thr = float(BOTTOM_Y_FRAC) * float(H)
    for oid, g in df.groupby("id", sort=False):
        g = g.sort_values("frame_idx")
        cy = g["cy"].astype(float).to_numpy()
        if cy.size == 0:
            bad.add(int(oid)); continue
        cmin = float(np.nanmin(cy)); cmax = float(np.nanmax(cy))
        if (cmin >= y_thr) and ((cmax - cmin) < float(BOTTOM_MIN_NET_DROP_PX)):
            bad.add(int(oid))
        else:
            keep.append(g)
    return (pd.concat(keep, ignore_index=True) if keep else df.iloc[0:0].copy()), bad


# -------------------- renumber --------------------
def _build_temporal_id_map(df: pd.DataFrame, start_at: int = 1) -> dict[int, int]:
    # Renumber IDs by (first frame, then first cy, then first cx). Purely cosmetic
    # — makes report files and overlays nicer to read.
    if df is None or df.empty:
        return {}
    first_f = df.groupby("id")["frame_idx"].min()
    first_rows = df.sort_values(["id", "frame_idx"]).groupby("id", sort=False).head(1).set_index("id")
    order_tbl = pd.DataFrame({
        "id":    first_f.index.astype(int),
        "start": first_f.to_numpy(dtype=np.int64),
        "cy0":   first_rows.get("cy", pd.Series(np.inf, index=first_f.index)).to_numpy(dtype=float),
        "cx0":   first_rows.get("cx", pd.Series(np.inf, index=first_f.index)).to_numpy(dtype=float),
    })
    order_tbl[["cy0", "cx0"]] = order_tbl[["cy0", "cx0"]].fillna(np.inf)
    order_tbl = order_tbl.sort_values(["start", "cy0", "cx0"], kind="mergesort")
    new_ids = np.arange(start_at, start_at + len(order_tbl), dtype=int)
    return dict(zip(order_tbl["id"].tolist(), new_ids.tolist()))


# -------------------- main --------------------
def main():
    args = _parse_args()
    final_dir = Path(args.final_dir).expanduser().resolve()

    H, W = _load_frames_meta(final_dir)

    pp = _load_pp_cfg(args.cfg)

    # Filter helpers read the thresholds from module globals, so we patch them
    # in from the YAML before calling any of them.
    global FALL_MIN_NET_DROP_PX, FALL_MIN_DROP_PER_FRAME_PX
    global BOTTOM_Y_FRAC, BOTTOM_MIN_NET_DROP_PX
    FALL_MIN_NET_DROP_PX        = float(pp["fall_min_net_drop_px"])
    FALL_MIN_DROP_PER_FRAME_PX  = float(pp["fall_min_drop_per_frame_px"])
    BOTTOM_Y_FRAC               = float(pp["bottom_y_frac"])
    BOTTOM_MIN_NET_DROP_PX      = float(pp["bottom_min_net_drop_px"])

    p_tracks = final_dir / "tracks_merged.parquet"
    if not p_tracks.exists():
        raise FileNotFoundError(f"{p_tracks} not found. Run the pipeline first.")
    df = pd.read_parquet(p_tracks)
    for col in ("frame_idx", "id", "ys", "xs"):
        if col not in df.columns:
            raise ValueError(f"Parquet missing '{col}'. Columns: {list(df.columns)}")
    df["frame_idx"] = df["frame_idx"].astype("int64")
    df["id"] = df["id"].astype("int64")

    # 1) centroids (always — needed for outputs even when all filters are off)
    centro = _centroids_parallel(df)

    removed_short: set = set()
    removed_short_after: set = set()
    removed_not_falling: set = set()
    removed_bottom: set = set()
    id_map: dict = {}

    if not bool(pp["enabled"]):
        print("[postprocess] enabled=false → skipping all filters (paper-strict mode)")
    else:
        # 2) min length
        min_len = int(pp["min_len_frames"])
        if min_len > 0:
            centro, removed_short = _filter_short_tracks(centro, min_len)

        # 3) yo-yo / stuck per-ID trimming
        if bool(pp["yoyo_trim"]):
            kept = []
            for oid, g in centro.groupby("id", sort=False):
                t = _trim_keep_best_segment(g)
                if not t.empty:
                    kept.append(t)
            centro = pd.concat(kept, ignore_index=True) if kept else centro.iloc[0:0].copy()

            # 3b) re-apply min length
            if min_len > 0:
                centro, removed_short_after = _filter_short_tracks(centro, min_len)

        # 4) falling plausibility (pure pixels)
        if bool(pp["falling_filter"]):
            centro, removed_not_falling = _filter_falling_pixels(centro)

        # 5) stuck bottom
        if bool(pp["bottom_stuck_filter"]):
            centro, removed_bottom = _filter_stuck_bottom(centro, H=H)

        # 6) renumber in temporal order
        if bool(pp["reindex_ids"]) and not centro.empty:
            id_map = _build_temporal_id_map(centro[["frame_idx", "id", "cy", "cx"]],
                                            start_at=int(ID_RENUMBER_START))
            centro["id"] = centro["id"].map(id_map).astype("int64")

    # 7) write outputs
    # df has the raw ys/xs we need; centro has the filtered (frame, id) keys.
    # Apply the same id_map remap to df, then inner-join on (frame, id).
    df_in = df.copy()
    if id_map:
        inv = {int(k): int(v) for k, v in id_map.items()}
        df_in["id"] = df_in["id"].map(inv).astype("Int64")
        df_in = df_in[df_in["id"].notna()].copy()
        df_in["id"] = df_in["id"].astype("int64")

    keys = centro[["frame_idx", "id"]].drop_duplicates()
    tracks_clean = df_in.merge(keys, on=["frame_idx", "id"], how="inner") \
                       .sort_values(["id", "frame_idx"])
    out_tracks = final_dir / "tracks_clean.parquet"
    tracks_clean[["frame_idx", "id", "ys", "xs"]].to_parquet(out_tracks, index=False)

    out_centro = final_dir / "centroids_clean.parquet"
    centro[["frame_idx", "id", "cx", "cy", "area_px"]].sort_values(["id", "frame_idx"]) \
                                                     .to_parquet(out_centro, index=False)

    report = {
        "input_rows": int(len(df)),
        "rows_final": int(len(tracks_clean)),
        "n_ids_final": int(tracks_clean["id"].nunique()) if not tracks_clean.empty else 0,
        "postprocess_cfg": pp,
        "removed_short_initial": sorted(map(int, removed_short)),
        "removed_short_after_trim": sorted(map(int, removed_short_after)),
        "removed_not_falling": sorted(map(int, removed_not_falling)),
        "removed_stuck_bottom": sorted(map(int, removed_bottom)),
        "id_reindex_map": {int(k): int(v) for k, v in id_map.items()},
    }
    (final_dir / "postprocess_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("\n=== POSTPROCESS MIN (no velocity) ===")
    print(f"  tracks_clean    → {out_tracks}")
    print(f"  centroids_clean → {out_centro}")
    print(f"  total droplets  = {report['n_ids_final']}")


if __name__ == "__main__":
    main()
