import argparse
import copy
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

from src.benchmark.count_metrics import compute_count_metrics, save_count_metrics
from src.benchmark.mask_metrics import compute_mask_metrics, save_mask_metrics
from src.benchmark.ontology import explain_class_alignment
from src.benchmark.frame_alignment import remap_prediction_frames_to_reference
from src.benchmark.reference_loader import load_seg_masks, load_tracks_clean
from src.benchmark.track_metrics import compute_track_metrics, save_track_metrics
from src.benchmark.data_validation import validate_reference_dataset, validate_interval_bounds
from src.benchmark.aggregate_report import generate_report
from src.benchmark.manual_count_evaluator import evaluate_manual_counts
from src.benchmark.processed_dataset_loader import _find_reference_dir, discover_processed_dataset, discover_processed_datasets
from src.benchmark.video_id_matcher import normalize_video_id, find_matching_video_id
from src.benchmark.frame_discovery import find_video_frame_dir
from src.utils import load_frame_names
import subprocess


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _find_video_frame_dir(base_video_dir: Path | None, reference_root: str | None, video_id: str) -> Path | None:
    if not video_id:
        print("[FRAME] No video_id provided for frame discovery.")
        return None

    source_dir, diagnostics = find_video_frame_dir(base_video_dir, reference_root, video_id)
    if diagnostics:
        for entry in diagnostics:
            print(f"[FRAME] candidate={entry['candidate_path']} frame_count={entry['frame_count']} reason={entry['reason']}")

    if source_dir is None:
        print(f"[FRAME] No frame directory found for {video_id} under {base_video_dir} or {reference_root}")
        return None

    print(f"[FRAME] Selected frame directory for {video_id}: {source_dir}")
    return source_dir


def _extract_interval_frames(source_dir: Path, interval: dict[str, Any], temp_dir: Path) -> Path:
    frame_names = load_frame_names(str(source_dir))
    if not frame_names:
        raise FileNotFoundError(f"No frame files found in {source_dir}")

    start_frame = int(interval["start_frame"])
    end_frame = int(interval["end_frame"])
    pad = 2
    safe_start = max(0, start_frame - pad)
    safe_end = max(safe_start, end_frame + pad)

    selected = frame_names[safe_start:safe_end + 1]
    if not selected:
        raise ValueError(f"No frames selected for interval {interval.get('interval_id')}")

    temp_dir.mkdir(parents=True, exist_ok=True)
    for index, frame_name in enumerate(selected, start=1):
        src = source_dir / frame_name
        dst = temp_dir / f"{index:06d}{src.suffix}"
        shutil.copy2(src, dst)
    return temp_dir


def _filter_benchmarks(benchmarks: dict[str, list[dict[str, Any]]], category: str | None, video_id: str | None, interval_id: str | None) -> dict[str, list[dict[str, Any]]]:
    filtered: dict[str, list[dict[str, Any]]] = {}
    for vid, intervals in benchmarks.items():
        if video_id and vid.lower() != video_id.lower():
            continue
        kept = []
        for interval in intervals:
            if category and str(interval.get("category", "")).lower() != category.lower():
                continue
            if interval_id and str(interval.get("interval_id", "")).lower() != interval_id.lower():
                continue
            kept.append(interval)
        if kept:
            filtered[vid] = kept
    return filtered


def _write_summary(rows: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark_summary.csv"
    json_path = out_dir / "benchmark_summary.json"

    baseline = min((float(row.get("total_runtime", 0.0) or 0.0) for row in rows if (row.get("total_runtime", 0.0) or 0.0) > 0), default=0.0)
    for row in rows:
        runtime = float(row.get("total_runtime", 0.0) or 0.0)
        row["speedup_vs_baseline"] = round(baseline / runtime, 3) if runtime > 0 and baseline > 0 else 0.0

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        headers = [
            "interval_id", "video_id", "category", "memory_update_skip", "total_runtime", "runtime_per_frame",
            "stage1_runtime", "stage2_runtime", "stage3_runtime",
            "cache_hits", "cache_misses", "propagation_calls",
            "droplet_count", "manual_count", "count_error",
            "mean_iou", "mean_dice", "avg_centroid_distance", "track_continuity", "speedup_vs_baseline",
        ]
        import csv

        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in headers})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"benchmark_summary": rows}, f, indent=2, ensure_ascii=False)

    return csv_path, json_path


def _resolve_reference_dir(reference_root: str | None, video_id: str) -> Path | None:
    if reference_root is None:
        return None

    candidate = Path(reference_root).expanduser().resolve()
    if not candidate.exists():
        print(f"[REF] Reference root not found: {candidate}")
        return None

    discovered = discover_processed_dataset(candidate, video_id=video_id)
    if discovered.get("reference_dir"):
        resolved = Path(discovered["reference_dir"])
        if resolved.exists():
            return resolved
        print(f"[REF] Discovered reference_dir for {video_id}, but path does not exist: {resolved}")

    normalized_video_id = normalize_video_id(video_id)
    candidates: list[Path] = []
    if candidate.is_dir():
        candidates.extend([
            candidate / f"{video_id}_data",
            candidate / f"{normalized_video_id}_data",
            candidate / video_id,
            candidate / normalized_video_id,
            candidate / video_id / "final",
            candidate / normalized_video_id / "final",
            candidate / "final",
        ])

    for path in candidates:
        if path.exists() and path.is_dir():
            if (path / "tracks_clean.parquet").exists():
                return path
            if (path / "final" / "tracks_clean.parquet").exists():
                return path / "final"

    if candidate.is_dir():
        for child in candidate.iterdir():
            if not child.is_dir():
                continue
            if normalize_video_id(child.name) == normalized_video_id:
                if (child / "tracks_clean.parquet").exists():
                    return child
                if (child / "final" / "tracks_clean.parquet").exists():
                    return child / "final"
            if normalize_video_id(child.name).startswith(normalized_video_id) or normalized_video_id.startswith(normalize_video_id(child.name)):
                if (child / "tracks_clean.parquet").exists():
                    return child
                if (child / "final" / "tracks_clean.parquet").exists():
                    return child / "final"

    return None


def _load_run_tracks(run_dir: Path) -> pd.DataFrame:
    for name in ("tracks_clean.parquet", "tracks.parquet", "tracks_merged.parquet"):
        path = run_dir / name
        if path.exists():
            df = pd.read_parquet(path)
            return df
    raise FileNotFoundError(f"No track parquet file found in {run_dir}")


def _load_run_seg_masks(run_dir: Path) -> pd.DataFrame:
    for name in ("segonly.parquet", "segonly_merged.parquet"):
        path = run_dir / name
        if path.exists():
            return pd.read_parquet(path)
    raise FileNotFoundError(f"No segmentation parquet file found in {run_dir}")


def _load_frames_meta(run_dir: Path) -> dict[str, Any] | None:
    meta_path = run_dir / "frames_meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _validate_benchmark_outputs(output_root: Path, run_dir: Path | None, report_json: Path, dry_run: bool = False) -> None:
    required_paths = [
        output_root / "benchmark_summary" / "benchmark_summary.csv",
        output_root / "benchmark_summary" / "benchmark_summary.json",
        output_root / "benchmark_summary" / "benchmark_report.json",
        output_root / "benchmark_summary" / "benchmark_report.html",
        output_root / "manual_count_metrics.json",
    ]
    if not dry_run and run_dir is not None:
        required_paths.extend([run_dir / "tracks.parquet", run_dir / "trajectories.csv"])
    for path in required_paths:
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"Required benchmark output missing or empty: {path}")
    html_text = (output_root / "benchmark_summary" / "benchmark_report.html").read_text(encoding="utf-8")
    if "<html" not in html_text.lower() or "</html>" not in html_text.lower():
        raise RuntimeError(f"Invalid HTML report: {output_root / 'benchmark_summary' / 'benchmark_report.html'}")
    if not report_json.exists() or report_json.stat().st_size <= 0:
        raise RuntimeError(f"Benchmark report JSON missing or empty: {report_json}")


def execute_benchmarks(
    pipeline_cfg_path: str,
    benchmark_yaml_path: str,
    category: str | None = None,
    video_id: str | None = None,
    interval_id: str | None = None,
    memory_update_skips: list[int] | None = None,
    reuse_existing_outputs: bool = True,
    reference_root: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    base_cfg = _load_yaml(Path(pipeline_cfg_path))
    benchmark_cfg = _load_yaml(Path(benchmark_yaml_path))
    
    # Auto-detect reference_root from config if not provided via CLI
    if reference_root is None:
        reference_root = base_cfg.get("data", {}).get("reference_root")

    benchmark_root = Path(base_cfg.get("data", {}).get("output_root", "./outputs")) / "benchmark_summary"
    benchmark_runs_root = Path(base_cfg.get("data", {}).get("output_root", "./outputs")) / "benchmark_runs"
    benchmark_runs_root.mkdir(parents=True, exist_ok=True)

    benchmarks = _filter_benchmarks(benchmark_cfg.get("benchmark_cases", {}), category, video_id, interval_id)

    if not benchmarks:
        print("[WARN] No benchmark intervals matched the requested filters.")
        return {"rows": [], "summary_csv": None, "summary_json": None}

    output_root = Path(base_cfg.get("data", {}).get("output_root", "./outputs"))
    benchmark_root = output_root / "benchmark_summary"
    benchmark_runs_root = output_root / "benchmark_runs"
    benchmark_runs_root.mkdir(parents=True, exist_ok=True)

    memory_update_skips = memory_update_skips or [1, 3, 5]
    rows: list[dict[str, Any]] = []

    single_video_reference_dir: Path | None = None
    single_video_mode = False
    if video_id and reference_root:
        candidate = Path(reference_root).expanduser()
        resolved_candidate = _find_reference_dir(candidate)
        if resolved_candidate is not None:
            single_video_reference_dir = resolved_candidate
            single_video_mode = True

    discovered_datasets = []
    processed_ids: list[str] = []
    benchmark_ids = sorted(benchmarks.keys())
    matched_video_map: dict[str, str | None] = {}
    matched_video_ids: list[str] = []
    matched_processed_ids: list[str] = []
    missing_benchmark_ids: list[str] = []
    unused_processed_ids: list[str] = []
    discovered_map: dict[str, dict[str, Any]] = {}

    if single_video_mode:
        print("[Benchmark] Benchmark mode: SINGLE_VIDEO")
        print(f"[Benchmark] Reference dataset detected: {single_video_reference_dir}")
        print("[Benchmark] Skipping global processed dataset discovery.")
    else:
        discovered_datasets = discover_processed_datasets(reference_root) if reference_root else []
        processed_ids = sorted(item["video_id"] for item in discovered_datasets if item.get("video_id"))
        matched_video_map = {
            vid: find_matching_video_id(vid, processed_ids)
            for vid in benchmark_ids
        }
        matched_video_ids = [vid for vid, matched in matched_video_map.items() if matched is not None]
        matched_processed_ids = sorted({matched for matched in matched_video_map.values() if matched})
        missing_benchmark_ids = [vid for vid, matched in matched_video_map.items() if matched is None]
        unused_processed_ids = [vid for vid in processed_ids if vid not in matched_processed_ids]
        discovered_map = {
            normalize_video_id(item["video_id"]): item
            for item in discovered_datasets
            if item.get("video_id")
        }
        discovered_map |= {
            normalize_video_id(Path(item.get("reference_dir", "")).name): item
            for item in discovered_datasets
            if item.get("reference_dir")
        }

        print(f"[Benchmark] Evaluating {sum(len(v) for v in benchmarks.values())} interval(s) across {len(benchmarks)} video(s).")
        print(f"[Benchmark] Discovered {len(discovered_datasets)} processed datasets under {reference_root}")
        print("[Benchmark] Video ID comparison:")
        print(f"  benchmark_ids={benchmark_ids}")
        print(f"  processed_ids={processed_ids}")
        print(f"  matched_ids={matched_video_ids}")
        print(f"  benchmark_ids_missing_from_processed={missing_benchmark_ids}")
        print(f"  processed_ids_not_referenced_by_benchmark={unused_processed_ids}")

        if not matched_video_ids:
            print("[ERROR] No benchmark video IDs match discovered processed dataset IDs. This is a DATASET COMPATIBILITY ISSUE.")
            return {"rows": [], "summary_csv": None, "summary_json": None}

    for video_id_name, intervals in tqdm(benchmarks.items(), desc="Benchmark sweeps"):
        source_dir = _find_video_frame_dir(
            Path(base_cfg.get("data", {}).get("video_dir", "")),
            reference_root if not single_video_mode else single_video_reference_dir,
            video_id_name,
        )
        if source_dir is None or not source_dir.exists():
            print(f"[WARN] No frame directory found for {video_id_name}; trying fallback discovery from processed datasets.")
            matched_id = find_matching_video_id(video_id_name, processed_ids)
            if matched_id:
                print(f"[INFO] Found processed dataset id for {video_id_name} -> {matched_id} via normalization.")
                source_dir = _find_video_frame_dir(
                    Path(base_cfg.get("data", {}).get("video_dir", "")),
                    reference_root,
                    matched_id,
                )
            if source_dir is None or not source_dir.exists():
                print(f"[ERROR] No frame directory found for {video_id_name}. Searched legacy video_dir and normalized candidates.")
                continue

        # Resolve and validate optional reference dataset for this video
        resolved_reference_dir = None
        if reference_root or single_video_reference_dir:
            if single_video_mode and single_video_reference_dir is not None:
                resolved = single_video_reference_dir
            else:
                resolved = _resolve_reference_dir(reference_root, video_id_name)
                if resolved is None:
                    matched_id = find_matching_video_id(video_id_name, processed_ids)
                    dataset_info = None
                    if matched_id:
                        dataset_info = discovered_map.get(normalize_video_id(matched_id))
                    if dataset_info and dataset_info.get("reference_dir"):
                        print(f"[WARN] Reference directory for {video_id_name} not found under {reference_root}; using discovered dataset {dataset_info.get('reference_dir')}")
                        resolved = Path(dataset_info["reference_dir"])
                    else:
                        print(f"[WARN] Reference directory for {video_id_name} not found under {reference_root}")
            if resolved is None:
                print(f"[WARN] No reference directory available for {video_id_name}")
            else:
                vres = validate_reference_dataset(resolved)
                if not vres.ok:
                    print(f"[WARN] Reference dataset validation failed for {video_id_name}: {vres.issues}")
                    print("[INFO] Continuing with evaluation and reporting the compatibility issues instead of blocking the run")
                else:
                    print(f"[OK] Reference dataset validated for {video_id_name}")
            resolved_reference_dir = resolved

        for interval in intervals:
            interval_id_name = interval.get("interval_id", "unknown")
            print(f"\n[Benchmark] Interval {interval_id_name} ({video_id_name}, {interval.get('category', 'unknown')})")

            for skip_value in memory_update_skips:
                temp_dir = Path(tempfile.mkdtemp(prefix=f"benchmark_{video_id_name}_{interval_id_name}_", dir=str(benchmark_runs_root)))
                try:
                    frame_dir = _extract_interval_frames(source_dir, interval, temp_dir / "frames")
                    cfg = copy.deepcopy(base_cfg)
                    cfg["data"]["video_dir"] = str(frame_dir)
                    cfg["data"]["output_root"] = str(output_root)
                    cfg["stage2"]["memory_update_skip"] = int(skip_value)
                    cfg["stage3"]["reuse_existing_outputs"] = bool(reuse_existing_outputs)
                    cfg["benchmark_meta"] = dict(interval)

                    if dry_run:
                        print(f"[DRY-RUN] would evaluate {video_id_name}/{interval_id_name} with memory_update_skip={skip_value}")
                        rows.append({
                            "interval_id": interval_id_name,
                            "video_id": video_id_name,
                            "category": interval.get("category", "unknown"),
                            "memory_update_skip": skip_value,
                            "total_runtime": 0.0,
                            "stage3_runtime": 0.0,
                            "droplet_count": 0,
                            "manual_count": interval.get("manual_count", -1),
                            "propagation_calls": 0,
                            "cache_hits": 0,
                            "cache_misses": 0,
                        })
                        continue

                    from .pipeline import run_pipeline

                    run_dir = benchmark_runs_root / video_id_name / interval_id_name / f"skip_{skip_value}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    out_dir = run_pipeline(cfg=cfg, frame_start=0, frame_end=None, force_run_dir=run_dir)
                    visual_dir = Path(output_root) / "benchmark_visualizations" / video_id_name / interval_id_name / f"skip_{skip_value}"
                    visual_dir.mkdir(parents=True, exist_ok=True)
                    overlays_dir = out_dir / "overlays"
                    if overlays_dir.exists():
                        for overlay_file in overlays_dir.glob("*.mp4"):
                            shutil.copy2(overlay_file, visual_dir / overlay_file.name)
                    log_dir = out_dir / cfg.get("stage3", {}).get("experiment_log_dir", "experiment_logs")
                    log_files = sorted(log_dir.glob("experiment_*.json"))
                    if not log_files:
                        print(f"[WARN] No experiment log found for {interval_id_name}, skip={skip_value}")
                        continue

                    with open(log_files[-1], "r", encoding="utf-8") as handle:
                        metrics = json.load(handle)

                    reference_metrics = {
                        "mean_iou": None,
                        "mean_dice": None,
                        "avg_centroid_distance": None,
                        "track_continuity": None,
                    }
                    if resolved_reference_dir:
                        try:
                            predicted_tracks = _load_run_tracks(out_dir)
                            ref_tracks = load_tracks_clean(resolved_reference_dir, video_id_name)
                            ref_masks = load_seg_masks(resolved_reference_dir, video_id_name)
                            reference_frame_values = set(int(v) for v in ref_tracks["abs_frame"].dropna().astype(int).tolist()) if "abs_frame" in ref_tracks.columns else None
                            predicted_tracks = remap_prediction_frames_to_reference(
                                predicted_tracks,
                                frame_offset=None,
                                reference_frames=reference_frame_values,
                            )
                            frames_meta = _load_frames_meta(out_dir)
                            width = 0
                            if frames_meta and isinstance(frames_meta.get("image_size"), dict):
                                width = int(frames_meta["image_size"].get("W", 0) or 0)

                            print(f"[EVAL] reference class alignment: {explain_class_alignment(ref_tracks)}")
                            print(f"[EVAL] predicted class alignment: {explain_class_alignment(predicted_tracks)}")

                            if "frame_idx" in ref_tracks.columns and "frame_idx" in predicted_tracks.columns:
                                ref_frames = set(int(v) for v in ref_tracks["frame_idx"].dropna().astype(int).tolist())
                                pred_frames = set(int(v) for v in predicted_tracks["frame_idx"].dropna().astype(int).tolist())
                                print(f"[EVAL] frame index overlap: reference={len(ref_frames)} predicted={len(pred_frames)} common={len(ref_frames & pred_frames)}")
                                print(f"[EVAL] frame index range reference={min(ref_frames)}..{max(ref_frames)} predicted={min(pred_frames)}..{max(pred_frames)}")
                            else:
                                print("[EVAL] frame index compatibility could not be checked because one or both dataframes lacked a frame_idx column")

                            def _describe_mask_geometry(df: pd.DataFrame) -> dict[str, Any]:
                                info: dict[str, Any] = {}
                                for field in ("mask_px", "ys", "xs", "centroid_x", "centroid_y", "area_px"):
                                    info[field] = field in df.columns
                                if "width" in df.columns:
                                    info["width_values"] = sorted({int(v) for v in df["width"].dropna().astype(int).tolist()})
                                if "height" in df.columns:
                                    info["height_values"] = sorted({int(v) for v in df["height"].dropna().astype(int).tolist()})
                                return info

                            print(f"[EVAL] reference mask geometry: {_describe_mask_geometry(ref_tracks)}")
                            print(f"[EVAL] predicted mask geometry: {_describe_mask_geometry(predicted_tracks)}")

                            interval_frames = set(int(v) for v in predicted_tracks["abs_frame"].dropna().astype(int).tolist()) if "abs_frame" in predicted_tracks.columns else None
                            if interval_frames is None and "rel_frame" in predicted_tracks.columns:
                                interval_frames = set(int(v) for v in predicted_tracks["rel_frame"].dropna().astype(int).tolist())
                            if interval_frames is None and "frame_idx" in predicted_tracks.columns:
                                interval_frames = set(int(v) for v in predicted_tracks["frame_idx"].dropna().astype(int).tolist())

                            count_metrics = compute_count_metrics(ref_tracks, predicted_tracks, interval_frames=interval_frames)
                            save_count_metrics(count_metrics, out_dir / "count_metrics.json")

                            mask_metrics = compute_mask_metrics(ref_masks, predicted_tracks, width=width, interval_frames=interval_frames)
                            save_mask_metrics(mask_metrics, out_dir / "mask_metrics.json")
                            reference_metrics["mean_iou"] = mask_metrics.get("mean_iou")
                            reference_metrics["mean_dice"] = mask_metrics.get("mean_dice")
                            reference_metrics["avg_centroid_distance"] = mask_metrics.get("mean_centroid_distance")

                            if mask_metrics.get("reference_mask_count", 0) <= 0 or mask_metrics.get("predicted_mask_count", 0) <= 0:
                                print("[EVAL] mask metrics are null or zero because either the reference or predicted droplet masks were empty after semantic filtering")
                            if mask_metrics.get("frame_alignment", {}).get("issues"):
                                print(f"[EVAL] frame alignment issues: {mask_metrics['frame_alignment']['issues']}")
                            if interval_frames is not None and "abs_frame" in ref_tracks.columns:
                                ref_frame_values = set(int(v) for v in ref_tracks["abs_frame"].dropna().astype(int).tolist())
                                print(f"[EVAL] reference interval frames available: {sorted(ref_frame_values)[:10]}... count={len(ref_frame_values)}")
                                print(f"[EVAL] predicted interval frames: {sorted(interval_frames)}")

                            track_metrics = compute_track_metrics(ref_tracks, predicted_tracks, interval_frames=interval_frames)
                            save_track_metrics(track_metrics, out_dir / "track_metrics.json")
                            # track_metrics uses 'avg_centroid_deviation' naming
                            reference_metrics["avg_centroid_distance"] = reference_metrics.get("avg_centroid_distance") or track_metrics.get("avg_centroid_deviation")
                            reference_metrics["track_continuity"] = track_metrics.get("track_continuity")

                            try:
                                from src.benchmark.count_metrics import compute_interval_reference_count
                                interval_reference_count = compute_interval_reference_count(ref_tracks, int(interval.get("start_frame", 0)), int(interval.get("end_frame", 0)))
                            except Exception:
                                interval_reference_count = None
                            metrics["benchmark_interval_reference_count"] = interval_reference_count
                            metrics["benchmark_count_error"] = None if interval_reference_count is None else int(metrics.get("droplet_count", 0)) - int(interval_reference_count)
                        except Exception as exc:
                            print(f"[WARN] Reference benchmark evaluation failed for {video_id_name}: {exc}")

                    row = {
                        "interval_id": metrics.get("benchmark_interval", interval_id_name),
                        "video_id": video_id_name,
                        "category": metrics.get("benchmark_category", interval.get("category", "unknown")),
                        "memory_update_skip": int(metrics.get("memory_update_skip", skip_value)),
                        "total_runtime": float(metrics.get("total_runtime", 0.0) or 0.0),
                        "stage3_runtime": float(metrics.get("stage3_runtime", 0.0) or 0.0),
                        "droplet_count": int(metrics.get("droplet_count", 0) or 0),
                        "manual_count": int(metrics.get("benchmark_manual_count", interval.get("manual_count", -1)) or -1),
                        "propagation_calls": int(metrics.get("stage3_propagation_calls", 0) or 0),
                        "cache_hits": int(metrics.get("stage3_cache_hits", 0) or 0),
                        "cache_misses": int(metrics.get("stage3_cache_misses", 0) or 0),
                        "mean_iou": reference_metrics["mean_iou"],
                        "mean_dice": reference_metrics["mean_dice"],
                        "avg_centroid_distance": reference_metrics["avg_centroid_distance"],
                        "track_continuity": reference_metrics["track_continuity"],
                    }
                    rows.append(row)
                    # augment with runtime-per-frame and stage runtimes when available
                    try:
                        frames_count = max(1, int(interval.get("end_frame", 0)) - int(interval.get("start_frame", 0)) + 1)
                    except Exception:
                        frames_count = 1
                    rows[-1]["runtime_per_frame"] = round(rows[-1]["total_runtime"] / frames_count, 4) if frames_count else None
                    rows[-1]["stage1_runtime"] = metrics.get("stage1_runtime")
                    rows[-1]["stage2_runtime"] = metrics.get("stage2_runtime")
                    rows[-1]["stage3_runtime"] = metrics.get("stage3_runtime") or rows[-1].get("stage3_runtime")
                    rows[-1]["count_error"] = metrics.get("benchmark_count_error")
                    print(f"[OK] Recorded skip={skip_value}, runtime={rows[-1]['total_runtime']:.2f}s")
                except Exception as exc:
                    print(f"[ERROR] Failed interval {interval_id_name} with memory_update_skip={skip_value}: {exc}")
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)

    csv_path, json_path = _write_summary(rows, benchmark_root)
    print(f"[Benchmark] Summary written to:\n  - {csv_path}\n  - {json_path}")

    # Prepare manual counts: use configs/video_manual_counts.json if present, else try to parse CSV/Excel
    manual_counts_cfg = Path('configs') / 'video_manual_counts.json'
    if not manual_counts_cfg.exists():
        print('[INFO] video_manual_counts.json not found, attempting to parse annotation files...')
        try:
            subprocess.run(["python", "tools/parse_manual_counts.py"], check=False)
        except Exception:
            print('[WARN] Failed to auto-run parse_manual_counts.py')

    manual_metrics_path = None
    if manual_counts_cfg.exists():
        try:
            manual_metrics_path = evaluate_manual_counts(manual_counts_cfg, output_root=Path(base_cfg.get('data',{}).get('output_root','./outputs')))
            print(f"[OK] Manual count metrics written to {manual_metrics_path}")
        except Exception as exc:
            print(f"[WARN] Manual count evaluation failed: {exc}")

    # Sanity checks and aggregate report
    try:
        report_json = generate_report(json_path, benchmark_runs_root, benchmark_root, generate_visuals=True, manual_metrics=manual_metrics_path)
        print(f"[Benchmark] Aggregated report written to: {report_json}")
    except Exception as exc:
        raise RuntimeError(f"Failed to generate benchmark report: {exc}") from exc

    first_success_row = next((row for row in rows if row.get("interval_id") and row.get("video_id")), None)
    if first_success_row is not None:
        first_run_dir = benchmark_runs_root / str(first_success_row["video_id"]) / str(first_success_row["interval_id"]) / f"skip_{first_success_row.get('memory_update_skip', memory_update_skips[0])}"
        _validate_benchmark_outputs(output_root, first_run_dir, report_json, dry_run=dry_run)

    return {"rows": rows, "summary_csv": csv_path, "summary_json": json_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated benchmark execution for difficult welding intervals.")
    parser.add_argument("--category", default=None, help="Filter by benchmark category, e.g. id_switches")
    parser.add_argument("--video-id", default=None, help="Filter by video identifier")
    parser.add_argument("--interval-id", default=None, help="Filter by interval identifier")
    parser.add_argument("--memory-update-skip", nargs="+", type=int, default=[1, 3, 5], help="Sweep values for memory_update_skip")
    parser.add_argument("--reuse-existing-outputs", action="store_true", help="Reuse cached outputs where available")
    parser.add_argument("--reference-dir", default=None, help="Path to reference dataset root containing *_data directories")
    parser.add_argument("--dry-run", action="store_true", help="Preview the sweep without executing the pipeline")
    args = parser.parse_args()

    pipeline_cfg = "configs/pipeline.yaml"
    benchmark_cfg = "configs/benchmark_cases.yaml"

    if not Path(benchmark_cfg).exists():
        print(f"[ERROR] {benchmark_cfg} not found. Run tools/benchmark_parser.py first.")
        return

    print("\nCLI examples:")
    print("  python -m src.run_benchmark --category id_switches")
    print("  python -m src.run_benchmark --interval-id AIS26T1_8245_8330")
    print("  python -m src.run_benchmark --reference-dir /path/to/ALS29T7_data")
    print("  python -m src.run_benchmark")

    execute_benchmarks(
        pipeline_cfg,
        benchmark_cfg,
        category=args.category,
        video_id=args.video_id,
        interval_id=args.interval_id,
        memory_update_skips=args.memory_update_skip,
        reuse_existing_outputs=args.reuse_existing_outputs,
        reference_root=args.reference_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()