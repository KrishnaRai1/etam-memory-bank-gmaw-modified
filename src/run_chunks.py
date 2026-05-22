# Multi-GPU chunked runner. Splits the sequence into overlapping chunks, runs
# run_pipeline on each chunk in its own process, then stitches per-chunk IDs
# into a single global ID space via mutual-best IoU on the overlap frames.
# Final outputs are appended to one parquet file per artifact.
from __future__ import annotations
import os
# These flags must be set before importing torch in child processes.
os.environ.setdefault("PYTORCH_DISABLE_TRITON", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import json, math, shutil, argparse, yaml
from pathlib import Path
from multiprocessing import get_context
import numpy as np
from PIL import Image

# ---------------------------- utils ----------------------------
def _detect_gpus(cfg_gpus):
    # Accepts None / list / "auto" / "0,1,3" — returns a list of GPU indices.
    if cfg_gpus is None: return []
    if isinstance(cfg_gpus, list): return [int(x) for x in cfg_gpus]
    s = str(cfg_gpus).strip().lower()
    if s == "auto":
        try:
            import torch
            if not torch.cuda.is_available(): return []
            return list(range(torch.cuda.device_count()))
        except Exception:
            return []
    if not s: return []
    return [int(x) for x in s.split(",") if x.strip().isdigit()]

def _downsample_names(names: list[str], factor: int) -> list[str]:
    if factor <= 1: return names
    idx = np.round(np.linspace(0, len(names) - 1, math.ceil(len(names) / factor))).astype(int)
    return [names[i] for i in idx]

def _make_chunks_by_len(N: int, m: int, overlap: int) -> list[tuple[int, int]]:
    # Split [0, N) into chunks of length m. Each next chunk starts `overlap`
    # frames before the previous chunk's end so we have shared frames for ID stitching.
    assert m > overlap, f"frames_per_part (= {m}) must be > overlap (= {overlap})"
    chunks, start = [], 0
    while start < N:
        end = min(start + m - 1, N - 1)
        chunks.append((start, end))
        if end == N - 1: break
        start = end + 1 - overlap
    return chunks

def _coords_to_bool(coords, H, W):
    m = np.zeros((H, W), dtype=bool)
    ys, xs = coords
    if xs.size: m[ys, xs] = True
    return m

def _load_T(path: Path):
    with open(path, "r") as f:
        raw = json.load(f)
    out = {}
    for oid, frames in raw.items():
        oid_i = int(oid)
        out[oid_i] = {}
        for fi, lists in frames.items():
            ys, xs = lists
            out[oid_i][int(fi)] = (np.array(ys, dtype=np.int64), np.array(xs, dtype=np.int64))
    return out

def _load_segonly_json(path: Path):
    if not path.exists(): return {}
    with open(path, "r") as f:
        raw = json.load(f)  # {"0":[[ys,xs],...]}
    out = {}
    for k, lst in raw.items():
        fi = int(k)
        out[fi] = [(np.array(ys, dtype=np.int64), np.array(xs, dtype=np.int64)) for ys, xs in lst]
    return out

def _image_size(video_dir: Path) -> tuple[int, int]:
    first = sorted(list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png")))
    if not first: raise FileNotFoundError(f"No frames found in {video_dir}")
    W, H = Image.open(first[0]).size
    return H, W

def _mutual_best(matches_ab: dict[tuple[int, int], float]):
    # Keep only pairs that are each other's top match on both sides. This avoids
    # collapsing two tracks of chunk A onto one track of chunk B.
    best_for_a, best_for_b = {}, {}
    for (aid, bid), iou in matches_ab.items():
        if (aid not in best_for_a) or (iou > best_for_a[aid][1]): best_for_a[aid] = (bid, iou)
        if (bid not in best_for_b) or (iou > best_for_b[bid][1]): best_for_b[bid] = (aid, iou)
    out = {}
    for aid, (bid, iou) in best_for_a.items():
        baid, iou_b = best_for_b.get(bid, (None, -1))
        if baid == aid and iou == iou_b: out[(aid, bid)] = iou
    return out

def _build_mapping_for_boundary(TA, TB, startA, endA, startB, endB, overlap, H, W, iou_thr=0.5):
    # On the overlap frames between two adjacent chunks, score every (idA, idB)
    # pair by max-over-frames IoU, then keep mutual-best pairs above iou_thr.
    lastA = list(range(endA - overlap + 1, endA + 1))
    firstB = list(range(startB, startB + overlap))
    F = [f for f in lastA if f in firstB]
    if not F: return {}
    scores = {}
    for aid, framesA in TA.items():
        for bid, framesB in TB.items():
            ious = []
            for f in F:
                # Each chunk uses chunk-local frame indices, so shift by startA/startB.
                la, lb = f - startA, f - startB
                if (la in framesA) and (lb in framesB):
                    mA = _coords_to_bool(framesA[la], H, W)
                    mB = _coords_to_bool(framesB[lb], H, W)
                    inter = np.logical_and(mA, mB).sum(dtype=np.float64)
                    union = np.logical_or(mA, mB).sum(dtype=np.float64)
                    iou = float(inter / max(union, 1.0))
                    ious.append(iou)
            if ious:
                mx = max(ious)
                if mx >= iou_thr:
                    scores[(aid, bid)] = mx
    return _mutual_best(scores)

def _create_chunk_view(video_dir: Path, names_ds: list[str], st: int, en: int, dest_root: Path) -> Path:
    # EfficientTAM consumes a folder of frames, so we expose this chunk's slice
    # via symlinks (or copy as fallback on filesystems without symlink support).
    frames_dir = dest_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for f in range(st, en + 1):
        src, dst = video_dir / names_ds[f], frames_dir / (video_dir / names_ds[f]).name
        if dst.exists(): continue
        try: os.symlink(src, dst)
        except Exception: shutil.copy2(src, dst)
    return frames_dir

# ---------------------------- worker ----------------------------
def _worker_run(cfg: dict, chunk_idx: int, gpu_visible: str | None,
                start_idx: int, end_idx: int, base_run_dir: Path,
                video_dir: Path, names_ds: list[str]):
    # Runs in a fresh process: set env flags, pin the visible GPU, then import torch.
    os.environ["PYTORCH_DISABLE_TRITON"] = "1"
    os.environ["TORCHINDUCTOR_DISABLE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    # Drop noisy logging env vars that may have leaked from the parent.
    for _k in ("TORCH_LOGS", "TORCH_LOGS_CONFIG"): os.environ.pop(_k, None)
    if gpu_visible is not None: os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_visible)
    else: os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    # Importing torch / pipeline must happen AFTER the env flags above.
    from .pipeline import run_pipeline

    ch_dir = base_run_dir / "chunks" / f"ch_{chunk_idx:02d}"
    ch_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = _create_chunk_view(video_dir, names_ds, start_idx, end_idx, ch_dir)

    # Deep-copy cfg via json so we can mutate it locally without affecting the parent.
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["data"]["output_root"] = str(ch_dir)
    cfg2["data"]["video_dir"] = str(frames_dir)
    cfg2.setdefault("run", {})
    cfg2["run"]["save_overlays"] = False

    # chunk_meta.json is the parent's signal that this worker finished. Write
    # an error file too if run_pipeline raises so the parent can surface it.
    meta_path = ch_dir / "chunk_meta.json"
    try:
        run_dir = run_pipeline(cfg2, frame_start=0, frame_end=(end_idx - start_idx), force_run_dir=ch_dir)
        meta = {"chunk_idx": chunk_idx, "start": start_idx, "end": end_idx, "run_dir": str(run_dir), "error": None}
        with open(meta_path, "w") as f: json.dump(meta, f, indent=2)
        return meta
    except Exception as e:
        import traceback
        err_txt = traceback.format_exc()
        with open(ch_dir / "chunk_error.txt", "w") as ef: ef.write(err_txt)
        meta = {"chunk_idx": chunk_idx, "start": start_idx, "end": end_idx, "run_dir": str(ch_dir), "error": str(e)}
        with open(meta_path, "w") as f: json.dump(meta, f, indent=2)
        raise

# ---------------------------- parquet builders (read T.json / segonly.json) ----------------------------
# Each chunk emits T.json / segonly_by_frame.json; the parent appends them into
# a single merged parquet, remapping local chunk IDs to global IDs via b2g.
def _append_tracks_from_Tjson(tbl_writer, out_path: Path, Tloc: dict, b2g: dict[int,int], start_idx: int):
    import pyarrow as pa, pyarrow.parquet as pq
    rec_f, rec_id, rec_ys, rec_xs = [], [], [], []
    rows = 0
    for bid, frames in Tloc.items():
        gid = int(b2g[int(bid)])
        for floc, (ys, xs) in frames.items():
            if xs.size == 0: continue
            rec_f.append(int(floc) + int(start_idx))
            rec_id.append(gid)
            rec_ys.append(pa.array(ys, type=pa.int64()))
            rec_xs.append(pa.array(xs, type=pa.int64()))
            rows += 1
    if rows == 0: return tbl_writer, 0
    col_f  = pa.array(rec_f,  type=pa.int64())
    col_id = pa.array(rec_id, type=pa.int64())
    col_ys = pa.array(rec_ys, type=pa.list_(pa.int64()))
    col_xs = pa.array(rec_xs, type=pa.list_(pa.int64()))
    tbl = pa.Table.from_arrays([col_f, col_id, col_ys, col_xs], names=["frame_idx","id","ys","xs"])
    if tbl_writer is None:
        tbl_writer = pq.ParquetWriter(out_path, tbl.schema)
    tbl_writer.write_table(tbl)
    return tbl_writer, rows

def _append_seg_from_json(tbl_writer, out_path: Path, SEG: dict[int, list[tuple]], start_idx: int, class_id_default: int):
    import pyarrow as pa, pyarrow.parquet as pq
    rec_f, rec_cls, rec_ys, rec_xs = [], [], [], []
    rows = 0
    for floc, lst in SEG.items():
        for (ys, xs) in lst:
            if xs.size == 0: continue
            rec_f.append(int(floc) + int(start_idx))
            rec_cls.append(int(class_id_default))
            rec_ys.append(pa.array(ys, type=pa.int64()))
            rec_xs.append(pa.array(xs, type=pa.int64()))
            rows += 1
    if rows == 0: return tbl_writer, 0
    col_f   = pa.array(rec_f,   type=pa.int64())
    col_cls = pa.array(rec_cls, type=pa.int64())
    col_ys  = pa.array(rec_ys,  type=pa.list_(pa.int64()))
    col_xs  = pa.array(rec_xs,  type=pa.list_(pa.int64()))
    tbl = pa.Table.from_arrays([col_f, col_cls, col_ys, col_xs], names=["frame_idx","class_id","ys","xs"])
    if tbl_writer is None:
        tbl_writer = pq.ParquetWriter(out_path, tbl.schema)
    tbl_writer.write_table(tbl)
    return tbl_writer, rows

# ---------------------------- scheduler + merge ----------------------------
# Launches up to k workers in parallel, then consumes results in order so that
# ID stitching is done sequentially against the previous chunk.
def main():
    from .utils import load_frame_names, ensure_run_dir

    ap = argparse.ArgumentParser("Multi-GPU chunked runner (stitch + Parquet merge from T.json)")
    ap.add_argument("--cfg", type=str, default="./configs/pipeline.yaml")
    ap.add_argument("--frames-per-part", type=int, default=None)
    ap.add_argument("--overlap", type=int, default=None)
    ap.add_argument("--strategy", choices=["fixed_len", "per_gpu"], default=None)
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--max-workers-per-gpu", type=int, default=None)
    args = ap.parse_args()

    cfg_path = Path(args.cfg)
    with open(cfg_path, "r") as f: cfg = yaml.safe_load(f)

    video_dir  = Path(cfg["data"]["video_dir"]).resolve()
    output_root= Path(cfg["data"]["output_root"]).resolve()

    names_all = load_frame_names(str(video_dir))
    ds_factor = int(cfg.get("run", {}).get("downsample", 1))
    names_ds  = _downsample_names(names_all, ds_factor)
    N = len(names_ds); assert N > 0, "No .jpg/.png frames after downsample"

    par = cfg.get("parallel", {}) or {}
    # CLI flags override the YAML's parallel.* settings.
    if args.frames_per_part is not None: par["frames_per_part"] = int(args.frames_per_part)
    if args.overlap is not None:        par["overlap"] = int(args.overlap)
    if args.strategy is not None:       par["strategy"] = str(args.strategy)
    if args.gpus is not None:           par["gpus"] = args.gpus
    if args.max_workers_per_gpu is not None: par["max_workers_per_gpu"] = int(args.max_workers_per_gpu)
    cfg["parallel"] = par

    overlap   = int(cfg.get("parallel", {}).get("overlap", 5))
    strategy  = str(cfg.get("parallel", {}).get("strategy", "fixed_len"))
    gpus      = _detect_gpus(cfg.get("parallel", {}).get("gpus", "auto"))
    mwpg      = int(cfg.get("parallel", {}).get("max_workers_per_gpu", 1))
    base_devices = len(gpus) if len(gpus) > 0 else 1
    k = max(1, base_devices * max(1, mwpg))

    # fixed_len: chunk size from YAML. per_gpu: one chunk per visible GPU.
    if strategy == "fixed_len":
        m = int(cfg.get("parallel", {}).get("frames_per_part", 0))
        chunks = _make_chunks_by_len(N, m, overlap)
    else:
        m = math.ceil(N / max(1, base_devices))
        chunks = _make_chunks_by_len(N, m, overlap)

    print(f"[plan] N={N}, chunks={chunks}, gpus={gpus}, concurrency={k}")
    # Sanity check: every chunk boundary must contain exactly `overlap` shared frames.
    for i in range(1, len(chunks)):
        a_start, a_end = chunks[i-1]; b_start, _ = chunks[i]
        expected_b_start = a_end - overlap + 1
        assert b_start == expected_b_start, f"Inconsistent boundary {i-1}->{i}: expected={expected_b_start}, got={b_start}"

    base_run_dir = ensure_run_dir(str(output_root))
    final_dir = base_run_dir / "final"; final_dir.mkdir(parents=True, exist_ok=True)

    H, W = _image_size(video_dir)

    import pyarrow.parquet as pq
    tracks_writer = None
    seg_writer    = None
    tracks_out = final_dir / "tracks_merged.parquet"
    seg_out    = final_dir / "segonly_merged.parquet"

    # Default class_id stored on seg-only rows: use the single segment_only class
    # if there is exactly one, otherwise -1 (mixed / unknown).
    classes_cfg = (cfg.get("yolo", {}).get("classes") or {})
    seg_classes = list(map(int, classes_cfg.get("segment_only", []))) if isinstance(classes_cfg, dict) else []
    seg_cls_default = seg_classes[0] if len(seg_classes) == 1 else -1

    max_gid = 0
    prev = None
    gpu_idx = 0

    # spawn context (not fork): CUDA contexts cannot survive fork().
    ctx = get_context("spawn")
    running: dict[int, any] = {}
    waiting = 0
    next_to_consume = 0

    def launch(i: int):
        # Round-robin GPU assignment so workers spread across visible devices.
        nonlocal gpu_idx
        st, en = chunks[i]
        gpu = gpus[gpu_idx % max(1, len(gpus))] if len(gpus) > 0 else None
        gpu_idx += 1
        p = ctx.Process(target=_worker_run,
                        args=(cfg, i, str(gpu) if gpu is not None else None, st, en, base_run_dir, video_dir, names_ds))
        p.start(); running[i] = p
        print(f"[launch] chunk {i:02d} -> GPU {gpu}  range={chunks[i]}")
        print(f"[running] active={sorted(running.keys())}")

    # Prime the pool up to the concurrency limit k.
    while waiting < len(chunks) and len(running) < k:
        launch(waiting); waiting += 1

    # Consume chunks in order. Out-of-order completions are fine: we just block
    # on next_to_consume's process and stitch sequentially against the previous one.
    while next_to_consume < len(chunks):
        p = running.get(next_to_consume)
        if p is not None:
            p.join()
            print(f"[join]   chunk {next_to_consume:02d} finished; active={sorted(set(running.keys()) - {next_to_consume})}")
            del running[next_to_consume]

        st, en = chunks[next_to_consume]
        ch_dir   = base_run_dir / "chunks" / f"ch_{next_to_consume:02d}"
        meta_path= ch_dir / "chunk_meta.json"
        if not meta_path.exists():
            err_hint = (ch_dir / "chunk_error.txt").read_text() if (ch_dir / "chunk_error.txt").exists() else "(no error file)"
            raise RuntimeError(f"Chunk {next_to_consume:02d} finished without meta.\nPath: {ch_dir}\nError:\n{err_hint}")

        meta = json.loads(Path(meta_path).read_text())
        if meta.get("error"):
            err_hint = (ch_dir / "chunk_error.txt").read_text() if (ch_dir / "chunk_error.txt").exists() else meta["error"]
            raise RuntimeError(f"Chunk {next_to_consume:02d} failed.\nPath: {ch_dir}\nError:\n{err_hint}")

        # Load this chunk's local tracks and seg-only masks.
        Tloc = _load_T(Path(meta["run_dir"]) / "T.json")
        SEG  = _load_segonly_json(Path(meta["run_dir"]) / "segonly_by_frame.json")

        curr = {"idx": next_to_consume, "start": st, "end": en, "T": Tloc}

        # b2g: chunk-local ID -> global ID. First chunk just enumerates new IDs.
        # Later chunks inherit IDs through mutual-best IoU on the overlap frames,
        # and any unmatched local ID becomes a brand-new global ID.
        if prev is None:
            b2g = {}
            for bid in sorted(Tloc.keys()):
                max_gid += 1; b2g[bid] = max_gid
            curr["b2g"] = b2g
        else:
            pairs = _build_mapping_for_boundary(prev["T"], Tloc, prev["start"], prev["end"], st, en, overlap, H, W, iou_thr=0.5)
            b2g, used = {}, set()
            for (aid, bid), _ in pairs.items():
                gid = prev["b2g"][aid]; b2g[bid] = gid; used.add(gid)
            for bid in Tloc.keys():
                if bid not in b2g:
                    max_gid += 1; b2g[bid] = max_gid
            curr["b2g"] = b2g

        # Append this chunk's rows (already remapped to global IDs) to the merged parquet.
        tracks_writer, ntr = _append_tracks_from_Tjson(tracks_writer, tracks_out, Tloc, curr["b2g"], start_idx=st)
        print(f"[merge] appended tracks rows={ntr} (chunk {next_to_consume:02d})")

        if SEG:
            seg_writer, nsg = _append_seg_from_json(seg_writer, seg_out, SEG, start_idx=st, class_id_default=seg_cls_default)
            print(f"[merge] appended seg rows={nsg} (chunk {next_to_consume:02d})")
        else:
            print(f"[merge] no seg rows (chunk {next_to_consume:02d})")

        prev = curr
        next_to_consume += 1

        while waiting < len(chunks) and len(running) < k:
            launch(waiting); waiting += 1

    if tracks_writer is not None: tracks_writer.close(); print(f"[final] wrote merged tracks → {tracks_out}")
    else: print("[final] no tracks written (empty)")

    if seg_writer is not None: seg_writer.close(); print(f"[final] wrote merged seg-only → {seg_out}")
    else: print("[final] no seg-only written (empty)")

    # frames_meta.json
    frames_meta = {
        "image_size": {"W": int(W), "H": int(H)},
        "offset": 0,
        "downsample": int(ds_factor),
        "frame_names": names_ds,
        "classes": cfg.get("yolo", {}).get("classes", {}),
    }
    with open(final_dir / "frames_meta.json", "w") as f: json.dump(frames_meta, f, indent=2)
    print(f"[OK] Final output at: {final_dir}")
    print(f" - Chunk artifacts: {base_run_dir / 'chunks'}")

if __name__ == "__main__":
    main()
