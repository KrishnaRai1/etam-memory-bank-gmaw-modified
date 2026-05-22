# Concatenates a folder of PNG/JPG frames into a single H.264 mp4 via ffmpeg.
# Defaults below act as fallback when the script is run with no CLI arguments;
# the CLI block at the bottom overrides them via --img-dir / --out / --fps.
import os
import re
import math
import numpy as np
from PIL import Image
from tqdm import tqdm
import imageio.v3 as iio
from imageio_ffmpeg import write_frames

# ====== fallback defaults ======
IMG_DIR      = "../EfficientTAM/pool_depression"        # input folder of frames
OUTPUT_MP4   = "../EfficientTAM/pool_depression.mp4"    # output mp4 path
FPS          = 30                                       # output frame rate
CRF          = 15                                       # libx264 quality (lower = better; 18~22 typical)
PRESET       = "slow"                                   # encode preset (ultrafast..veryslow)
PIX_FMT      = "yuv420p"                                # broadly compatible pixel format
IMG_EXTS     = {".jpg", ".jpeg", ".png"}                # accepted input extensions
# ===============================

def _numeric_sort_key(name: str):
    # Sort by the LAST integer in the basename so e.g. "frame_000123.png"
    # orders by 123, not by lexicographic suffix.
    root, _ = os.path.splitext(name)
    nums = re.findall(r"\d+", root)
    return (int(nums[-1]) if nums else math.inf, name)

def list_images(img_dir, exts):
    files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in exts]
    files.sort(key=_numeric_sort_key)
    return [os.path.join(img_dir, f) for f in files]

def ensure_even_dims(arr):
    # H.264 requires even width and height; pad by one pixel on the right/bottom
    # when needed instead of resizing, to keep pixel content unchanged.
    h, w = arr.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return arr
    if arr.ndim == 2:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="edge")
    else:
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return arr

def main():
    imgs = list_images(IMG_DIR, IMG_EXTS)
    if not imgs:
        raise FileNotFoundError(f"No images found in: {IMG_DIR}")

    os.makedirs(os.path.dirname(OUTPUT_MP4) or ".", exist_ok=True)

    # Use the first frame's (even-padded) size as the output size; any frame
    # that differs later will be resampled to match.
    first = np.array(Image.open(imgs[0]).convert("RGB"))
    first = ensure_even_dims(first)
    h, w = first.shape[:2]

    # ffmpeg encoder params: quality (CRF) over fixed bitrate.
    output_params = [
        "-crf", str(CRF),
        "-preset", PRESET,
        "-pix_fmt", PIX_FMT,
        # To target a bitrate instead of CRF:
        # "-maxrate", "10M", "-bufsize", "20M",
        # For progressive playback in browsers:
        # "-movflags", "+faststart",
    ]

    writer = write_frames(
        OUTPUT_MP4,
        size=(w, h),
        fps=FPS,
        codec="libx264",
        pix_fmt_in="rgb24",      # frames are sent as raw RGB bytes
        output_params=output_params,
    )
    next(writer)  # prime the generator (imageio-ffmpeg API)

    pbar = tqdm(total=len(imgs), desc="Encoding MP4", unit="frame")
    try:
        # First frame is already loaded and even-padded.
        writer.send(first.tobytes())
        pbar.update(1)

        for path in imgs[1:]:
            frame = np.array(Image.open(path).convert("RGB"))
            frame = ensure_even_dims(frame)
            # Defensive resize: keeps the encoder happy if a frame slipped through
            # with a different resolution than the first one.
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = np.array(Image.fromarray(frame).resize((w, h), Image.BICUBIC))
            writer.send(frame.tobytes())
            pbar.update(1)
    finally:
        # write_frames raises StopIteration on close() — swallow it.
        try:
            writer.close()
        except StopIteration:
            pass
        pbar.close()

    print(f"[OK] mp4 written: {OUTPUT_MP4}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Images -> MP4")
    ap.add_argument("--img-dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    # main() reads these as module globals, so override before calling.
    IMG_DIR = args.img_dir
    OUTPUT_MP4 = args.out
    FPS = args.fps

    if "main" in globals() and callable(globals()["main"]):
        globals()["main"]()
