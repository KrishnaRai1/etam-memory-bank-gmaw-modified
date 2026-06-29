from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import math
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from tools.export_benchmark_visualization import render_visualization


def _finite(x):
    try:
        return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
    except Exception:
        return False


def _safe_mean(values):
    vals = [v for v in values if _finite(v)]
    return float(statistics.mean(vals)) if vals else None


def generate_report(summary_json: Path, runs_root: Path, out_root: Path, generate_visuals: bool = True, manual_metrics: Path | None = None) -> Path:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    data = json.loads(summary_json.read_text(encoding='utf-8'))
    rows = data.get('benchmark_summary', [])
    if not rows:
        raise RuntimeError('No benchmark rows found in summary json')

    df = pd.DataFrame(rows)
    # sanity checks: drop invalid rows
    df = df[df['interval_id'].notna()]
    df = df[df['video_id'].notna()]

    # ensure numeric conversion
    numeric_cols = ['total_runtime','stage1_runtime','stage2_runtime','stage3_runtime','runtime_per_frame','mean_iou','mean_dice','avg_centroid_distance','track_continuity','droplet_count','manual_count','count_error']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # runtime summary
    runtime_summary = {
        'best_runtime': float(df['total_runtime'].min()) if 'total_runtime' in df.columns else None,
        'worst_runtime': float(df['total_runtime'].max()) if 'total_runtime' in df.columns else None,
        'mean_runtime': float(df['total_runtime'].mean()) if 'total_runtime' in df.columns else None,
    }

    # memory-bank summary by skip
    skips = {}
    for skip in sorted(df['memory_update_skip'].dropna().unique().tolist()):
        grp = df[df['memory_update_skip'] == skip]
        skips[int(skip)] = {
            'count': int(len(grp)),
            'mean_total_runtime': _safe_mean(grp['total_runtime'].tolist()) if 'total_runtime' in grp else None,
            'mean_iou': _safe_mean(grp['mean_iou'].tolist()) if 'mean_iou' in grp else None,
            'mean_dice': _safe_mean(grp['mean_dice'].tolist()) if 'mean_dice' in grp else None,
            'mean_track_continuity': _safe_mean(grp['track_continuity'].tolist()) if 'track_continuity' in grp else None,
        }

    # counts
    count_errs = df['count_error'].dropna().tolist() if 'count_error' in df.columns else []
    count_summary = {
        'mean_count_error': float(statistics.mean(count_errs)) if count_errs else None,
        'best_count_error': int(min(count_errs)) if count_errs else None,
        'worst_count_error': int(max(count_errs)) if count_errs else None,
    }

    mask_summary = {
        'mean_iou': _safe_mean(df['mean_iou'].tolist()) if 'mean_iou' in df.columns else None,
        'mean_dice': _safe_mean(df['mean_dice'].tolist()) if 'mean_dice' in df.columns else None,
    }

    track_summary = {
        'mean_track_continuity': _safe_mean(df['track_continuity'].tolist()) if 'track_continuity' in df.columns else None,
        'mean_centroid_deviation': _safe_mean(df['avg_centroid_distance'].tolist()) if 'avg_centroid_distance' in df.columns else None,
    }

    # identify best overall configuration by simple scoring (lower runtime better, higher IoU/Dice/continuity better)
    best = None
    scores = []
    for skip, info in skips.items():
        score = 0.0
        if info.get('mean_total_runtime'):
            score += info['mean_total_runtime']
        # prefer higher metrics -> subtract
        if info.get('mean_iou'):
            score -= (info['mean_iou'] * 0.1)
        if info.get('mean_dice'):
            score -= (info['mean_dice'] * 0.1)
        if info.get('mean_track_continuity'):
            score -= (info['mean_track_continuity'] * 0.1)
        scores.append((skip, score))
    if scores:
        best = sorted(scores, key=lambda x: x[1])[0][0]

    per_video_metrics = {}
    for video_id in sorted(df['video_id'].dropna().unique().tolist()):
        grp = df[df['video_id'] == video_id]
        per_video_metrics[str(video_id)] = {
            'intervals_evaluated': int(len(grp)),
            'mean_total_runtime': _safe_mean(grp['total_runtime'].tolist()) if 'total_runtime' in grp.columns else None,
            'mean_iou': _safe_mean(grp['mean_iou'].tolist()) if 'mean_iou' in grp.columns else None,
            'mean_dice': _safe_mean(grp['mean_dice'].tolist()) if 'mean_dice' in grp.columns else None,
            'mean_track_continuity': _safe_mean(grp['track_continuity'].tolist()) if 'track_continuity' in grp.columns else None,
            'mean_count_error': _safe_mean(grp['count_error'].tolist()) if 'count_error' in grp.columns else None,
        }

    cross_video_metrics = {
        'videos_evaluated': int(df['video_id'].nunique()) if 'video_id' in df.columns else 0,
        'intervals_evaluated': int(len(df)),
        'mean_total_runtime': _safe_mean(df['total_runtime'].tolist()) if 'total_runtime' in df.columns else None,
        'mean_iou': _safe_mean(df['mean_iou'].tolist()) if 'mean_iou' in df.columns else None,
        'mean_dice': _safe_mean(df['mean_dice'].tolist()) if 'mean_dice' in df.columns else None,
        'mean_track_continuity': _safe_mean(df['track_continuity'].tolist()) if 'track_continuity' in df.columns else None,
        'mean_count_error': _safe_mean(df['count_error'].tolist()) if 'count_error' in df.columns else None,
    }

    report = {
        'dataset_summary': {
            'videos_evaluated': int(df['video_id'].nunique()),
            'intervals_evaluated': int(len(df)),
        },
        'runtime_summary': runtime_summary,
        'memory_bank_summary': skips,
        'count_summary': count_summary,
        'mask_summary': mask_summary,
        'track_summary': track_summary,
        'per_video_metrics': per_video_metrics,
        'cross_video_metrics': cross_video_metrics,
        'best_overall_memory_update_skip': int(best) if best is not None else None,
    }

    # include manual count metrics if available
    if manual_metrics and manual_metrics.exists():
        try:
            manual_data = json.loads(manual_metrics.read_text(encoding='utf-8'))
            report['manual_count_metrics'] = manual_data
            # aggregate totals
            ref_total = 0
            pred_total = 0
            ref_count = 0
            pred_count = 0
            for vid, m in manual_data.items():
                if isinstance(m, dict) and m.get('reference_count') is not None:
                    ref_total += int(m['reference_count'])
                    ref_count += 1
                if isinstance(m, dict) and m.get('predicted_count') is not None:
                    pred_total += int(m['predicted_count'])
                    pred_count += 1
            report['count_overall'] = {'total_reference_count': ref_total if ref_count else None, 'total_predicted_count': pred_total if pred_count else None}
        except Exception:
            pass

    out_json = out_root / 'benchmark_report.json'
    out_json.write_text(json.dumps(report, indent=2), encoding='utf-8')

    def _render_table(title: str, values: dict[str, Any]) -> str:
        html = f"<h3>{title}</h3>"
        html += "<table border='1' cellspacing='0' cellpadding='4' style='border-collapse:collapse'>"
        html += "<tr><th>Metric</th><th>Value</th></tr>"
        for key, value in sorted(values.items()):
            html += f"<tr><td>{key}</td><td>{value if value is not None else 'N/A'}</td></tr>"
        html += "</table>"
        return html

    html_path = out_root / 'benchmark_report.html'
    generated_assets: list[Path] = []
    with open(html_path, 'w', encoding='utf-8') as fh:
        fh.write('<html><head><title>Benchmark Report</title></head><body>')
        fh.write(f"<h1>Benchmark Report</h1>")
        fh.write(f"<p><a href='benchmark_report.json'>View raw benchmark_report.json</a></p>")
        fh.write(f"<h2>Dataset</h2><p>Videos: {report['dataset_summary']['videos_evaluated']}, Intervals: {report['dataset_summary']['intervals_evaluated']}</p>")
        fh.write(_render_table('Dataset Summary', report['dataset_summary']))
        fh.write(_render_table('Runtime Summary', runtime_summary))
        fh.write(_render_table('Count Summary', count_summary))
        fh.write(_render_table('Mask Summary', mask_summary))
        fh.write(_render_table('Track Summary', track_summary))

        # runtime plot
        if 'total_runtime' in df.columns:
            plt.figure(figsize=(6,3))
            df.boxplot(column='total_runtime', by='memory_update_skip')
            plt.title('Total runtime by memory_update_skip')
            plt.suptitle('')
            plt.xlabel('memory_update_skip')
            plt.ylabel('seconds')
            rp = out_root / 'runtime_box.png'
            plt.savefig(rp, bbox_inches='tight')
            plt.close()
            generated_assets.append(rp)
            fh.write(f"<h3>Runtime</h3><img src='{rp.name}' alt='runtime'>")

        # IoU plot
        if 'mean_iou' in df.columns:
            plt.figure(figsize=(6,3))
            df.boxplot(column='mean_iou', by='memory_update_skip')
            plt.title('Mean IoU by memory_update_skip')
            plt.suptitle('')
            rp = out_root / 'iou_box.png'
            plt.savefig(rp, bbox_inches='tight')
            plt.close()
            generated_assets.append(rp)
            fh.write(f"<h3>IoU</h3><img src='{rp.name}' alt='iou'>")

        # Count error plot
        if 'count_error' in df.columns:
            plt.figure(figsize=(6,3))
            df.boxplot(column='count_error', by='memory_update_skip')
            plt.title('Count error by memory_update_skip')
            plt.suptitle('')
            rp = out_root / 'count_box.png'
            plt.savefig(rp, bbox_inches='tight')
            plt.close()
            generated_assets.append(rp)
            fh.write(f"<h3>Count Error</h3><img src='{rp.name}' alt='count'>")

        fh.write('</body></html>')

    if not html_path.exists() or html_path.stat().st_size <= 0:
        raise RuntimeError(f'Failed to generate benchmark HTML report: {html_path}')
    html_text = html_path.read_text(encoding='utf-8')
    if '<html' not in html_text.lower() or '</html>' not in html_text.lower():
        raise RuntimeError(f'Benchmark HTML report is invalid: {html_path}')
    for asset in generated_assets:
        if not asset.exists() or asset.stat().st_size <= 0:
            raise RuntimeError(f'Benchmark report asset missing or empty: {asset}')

    # optional visualizations per run
    if generate_visuals:
        for _, row in df.iterrows():
            vid = row.get('video_id')
            interval = row.get('interval_id')
            skip = row.get('memory_update_skip')
            if not vid or not interval:
                continue
            run_dir = runs_root / str(vid) / str(interval) / f"skip_{skip}"
            images_dir = run_dir / 'frames'
            pred_dir = run_dir
            # try to find reference dir sibling under runs_root? skip if not present
            ref_dir = None
            # we cannot reliably guess reference location here; caller can pass explicit reference root
            try:
                if images_dir.exists() and pred_dir.exists():
                    vis_out = out_root / 'benchmark_visualizations' / str(vid) / str(interval) / f"skip_{skip}"
                    try:
                        render_visualization(images_dir, pred_dir.parent / 'reference' if False else pred_dir, pred_dir, vis_out, mp4_path=None)
                    except Exception:
                        # best-effort only
                        pass
            except Exception:
                continue

    return out_json
