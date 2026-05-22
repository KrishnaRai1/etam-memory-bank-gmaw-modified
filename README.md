# HS-WeldCount — Droplet Tracking + Counting

Detect, segment, track and **count** droplets in high-speed GMAW welding video. Three stages:
**YOLO** per-frame detection → **EfficientTAM** segmentation/propagation → temporal filter + long-term tracking with new-object discovery.

## Layout

```
tracking_only/
├── src/                # pipeline core (Stages 1-2-3 + chunked multi-GPU runner)
├── tools/              # postprocess, counting, overlay rendering
├── configs/            # pipeline.yaml + EfficientTAM cfg
├── efficient_track_anything/   # segmenter package (checkpoints/ is empty by default)
├── yolov11x/           # detector weights folder (empty by default)
├── file2mp4.py         # frames -> mp4
└── outputs/            # created at runtime (one subfolder per run)
```

## 1. Install

```bash
pip install -r requirements.txt
```

For GPU acceleration, install the torch wheel that matches your CUDA version (see the comment in `requirements.txt`). A CUDA-capable GPU is strongly recommended.

## 2. Download checkpoints

Weights are **not** in this repo. Get them from:

**https://drive.google.com/drive/folders/1iY52wo8ZthAMKxm3gh3Hr5jdWVaijZyH?usp=sharing**

Place the files exactly as follows:

| File from Drive | Where to put it |
|---|---|
| `efficienttam_s.pt` | `efficient_track_anything/checkpoints/efficienttam_s.pt` |
| `best.pt` | `yolov11x/best.pt` |

## 3. Run

Point `data.video_dir` in `configs/pipeline.yaml` at a folder of `.jpg`/`.png` frames, then:

```bash
python -m src.main --cfg configs/pipeline.yaml
```

This runs the chunked pipeline, post-processing, counting, overlay rendering and MP4 encoding in one go. Final outputs land in `outputs/<timestamp>/final/`:

- `tracks_clean.parquet`, `centroids_clean.parquet`
- `droplet_count_summary.json`, `droplet_count_per_frame.csv`
- `overlays_post/*.png`, `video_out.mp4`

## 4. Paper-strict mode

The repo includes welding-specific filters and visual outputs that are not part of the original methodology. To disable them, set in `configs/pipeline.yaml`:

```yaml
postprocess:
  enabled: false
output:
  render_overlays: false
  make_mp4: false
```

The total droplet count is then just the number of unique masklets produced by Stage 3.

## References

- Bondi, E. et al. **CountVid: Open-World Object Counting in Videos.** [arXiv:2506.15368](https://arxiv.org/abs/2506.15368)
- Xiong, Y. et al. **Efficient Track Anything.** [arXiv:2411.18933](https://arxiv.org/abs/2411.18933)
- Amini, N. et al. **CountGD: Multi-Modal Open-World Counting.** [arXiv:2408.00714](https://arxiv.org/abs/2408.00714)
