# End-to-end orchestrator:
#   run_chunks (or run_pipeline) -> postprocess -> count -> render -> mp4
from __future__ import annotations
import argparse, yaml, sys, subprocess, copy, os, tempfile
from pathlib import Path
from typing import Optional
from .pipeline import run_pipeline

def parse_args():
    ap = argparse.ArgumentParser("YOLO + EfficientTAM Orchestrator (sequential)")
    ap.add_argument("--cfg", type=str, default="./configs/pipeline.yaml", help="Path to pipeline YAML")
    ap.add_argument("--frame-start", type=int, default=None)
    ap.add_argument("--frame-end", type=int, default=None)
    ap.add_argument("--force-run-dir", type=str, default=None)
    ap.add_argument("--no-chunks", action="store_true")
    ap.add_argument("--video-out", type=str, default=None)
    return ap.parse_args()

def _resolve_rel(p: str | Path, anchor: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (anchor / p)

def _find_latest_final_dir(output_root: Path) -> Optional[Path]:
    # run_chunks writes outputs/<timestamp>/final/. Pick the most recent one.
    if not output_root.exists():
        return None
    candidates = []
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        final = child / "final"
        if final.is_dir():
            candidates.append(final)
    if not candidates:
        return None
    candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return candidates[0]

def _run(cmd: list[str]) -> None:
    # -u: unbuffered so subprocess logs stream live.
    if cmd and cmd[0] == sys.executable and (len(cmd) == 1 or cmd[1] != "-u"):
        cmd = [cmd[0], "-u"] + cmd[1:]
    print(f"[exec] {' '.join(map(str, cmd))}")
    subprocess.run(list(map(str, cmd)), check=True)

def main():
    args = parse_args()

    # Repo root = parent of src/
    repo_root = Path(__file__).resolve().parents[1]
    tools_dir = repo_root / "tools"
    file2mp4_py = repo_root / "file2mp4.py"

    cfg_path = Path(args.cfg).expanduser().resolve()
    assert cfg_path.exists(), f"Config not found: {cfg_path}"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    print("[DEBUG] main cfg memory_update_skip =", cfg.get("stage2", {}).get("memory_update_skip", 1))

    # YAML paths are resolved against the repo root, not the configs/ folder.
    data = cfg.get("data", {}) or {}
    video_dir = _resolve_rel(data.get("video_dir", "./videos"), repo_root).expanduser().resolve()
    output_root = _resolve_rel(data.get("output_root", "./outputs"), repo_root).expanduser().resolve()

    par = cfg.get("parallel", {}) or {}
    strategy = str(par.get("strategy", "")).lower().strip()
    use_chunks = (strategy in {"fixed_len", "per_gpu"}) and (not args.no_chunks)

    # 0) core pipeline (chunked when parallel.strategy is set)
    if use_chunks:
        runtime_cfg_path = Path(tempfile.gettempdir()) / f"etam_runtime_cfg_{os.getpid()}.yaml"
        with open(runtime_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(copy.deepcopy(cfg), f, sort_keys=False)
        print("[DEBUG] chunk runner cfg path =", runtime_cfg_path)
        print("[DEBUG] chunk runner memory_update_skip =", cfg.get("stage2", {}).get("memory_update_skip", 1))
        cfg_path = runtime_cfg_path
        _run([sys.executable, "-m", "src.run_chunks", "--cfg", str(cfg_path)])
        final_dir = _find_latest_final_dir(output_root)
        assert final_dir and final_dir.exists(), f"Could not locate final/ under {output_root}"
    else:
        run_dir = run_pipeline(
            cfg,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            force_run_dir=Path(args.force_run_dir).expanduser().resolve() if args.force_run_dir else None,
        )
        final_dir = (Path(run_dir) / "final").expanduser().resolve()

    print(f"[orchestrator] final_dir = {final_dir}")
    print(f"[orchestrator] video_dir = {video_dir}")
    assert video_dir.is_dir(), f"Frames directory not found: {video_dir}"

    overlays_dir = (final_dir / "overlays_post").resolve()
    overlays_dir.mkdir(parents=True, exist_ok=True)

    video_out = Path(args.video_out).expanduser().resolve() if args.video_out else (output_root / "video_out.mp4")

    post_py   = (tools_dir / "postprocess_tracks_min.py").resolve()
    render_py = (tools_dir / "render_overlays_min.py").resolve()
    count_py  = (tools_dir / "count_droplets.py").resolve()

    out_cfg = (cfg.get("output") or {})
    do_render = bool(out_cfg.get("render_overlays", True))
    do_mp4    = bool(out_cfg.get("make_mp4", True))
    fps       = int(out_cfg.get("fps", 30))

    # 1) postprocess (flags read from --cfg)
    _run([sys.executable, str(post_py), "--final-dir", str(final_dir), "--cfg", str(cfg_path)])

    # 2) count droplets
    _run([sys.executable, str(count_py), "--final-dir", str(final_dir)])

    # 3) render overlays (optional)
    if do_render:
        _run([
            sys.executable, str(render_py),
            "--images-dir", str(video_dir),
            "--final-dir",  str(final_dir),
            "--out-dir",    str(overlays_dir),
        ])
    else:
        print("[orchestrator] output.render_overlays=false → skipping overlay rendering")

    # 4) file2mp4 (optional; requires overlays)
    if do_mp4:
        if not do_render:
            print("[orchestrator] WARN: make_mp4=true but render_overlays=false; mp4 will use existing overlays_post if any")
        _run([
            sys.executable, str(file2mp4_py),
            "--img-dir", str(overlays_dir),
            "--out",     str(video_out),
            "--fps",     str(fps),
        ])
    else:
        print("[orchestrator] output.make_mp4=false → skipping mp4 generation")

    steps = ["postprocess", "count"]
    if do_render: steps.append("render")
    if do_mp4:    steps.append("mp4")
    print(f"[DONE] {' -> '.join(steps)}")

if __name__ == "__main__":
    main()
