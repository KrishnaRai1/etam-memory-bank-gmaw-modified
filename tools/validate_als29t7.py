"""Run end-to-end validation for ALS29T7 reference data.
Generates outputs/validation_reports/ALS29T7_validation_report.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.benchmark.data_validation import validate_reference_dataset, validate_processed_dataset_root
from src.benchmark.processed_dataset_loader import discover_processed_datasets


def main(reference_dir: Path, manual_counts_cfg: Path | None = None, dataset_root: Path | None = None) -> Path:
    out_root = Path('outputs') / 'validation_reports'
    out_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}

    if dataset_root is None:
        pipeline_cfg = REPO_ROOT / 'configs' / 'pipeline.yaml'
        if pipeline_cfg.exists():
            cfg = json.loads(json.dumps(yaml.safe_load(pipeline_cfg.read_text(encoding='utf-8')))) if False else None
        else:
            cfg = None
        if cfg is None:
            try:
                import yaml
                cfg = yaml.safe_load(pipeline_cfg.read_text(encoding='utf-8')) or {}
            except Exception:
                cfg = {}
        dataset_root = Path(cfg.get('data', {}).get('reference_root', 'New_experiments_v3_final'))
    dataset_root = dataset_root or reference_dir
    discovered = discover_processed_datasets(dataset_root)
    report['dataset_root'] = str(dataset_root)
    report['discovered_videos'] = len(discovered)
    report['videos'] = discovered

    if reference_dir.exists():
        vres = validate_reference_dataset(reference_dir)
        report['reference_validation'] = {'ok': vres.ok, 'issues': vres.issues, 'details': vres.details}
    else:
        report['reference_validation'] = {'ok': False, 'issues': [f'Reference directory not found: {reference_dir}'], 'details': {}}

    if dataset_root.exists():
        root_validation = validate_processed_dataset_root(dataset_root)
        report['dataset_validation'] = {'ok': root_validation.ok, 'issues': root_validation.issues, 'details': root_validation.details}
    else:
        report['dataset_validation'] = {'ok': False, 'issues': [f'Dataset root not found: {dataset_root}'], 'details': {}}

    # manual counts availability
    manual_ok = False
    if manual_counts_cfg and manual_counts_cfg.exists():
        try:
            m = json.loads(manual_counts_cfg.read_text(encoding='utf-8'))
            video_manuals = m.get('video_manual_counts', {})
            manual_ok = bool(video_manuals.get('AIS29T7') or video_manuals.get('ALS29T7') or any(k.lower().startswith('ais29t7') or k.lower().startswith('als29t7') for k in video_manuals.keys()))
            report['manual_counts_available'] = manual_ok
            report['manual_counts_sources'] = m.get('sources', [])
        except Exception as exc:
            report['manual_counts_available'] = False
            report['manual_counts_error'] = str(exc)
    else:
        report['manual_counts_available'] = False

    # run alignment checker (best-effort) for each discovered video
    try:
        ar_out = out_root / 'alignment_check'
        ar_out.mkdir(parents=True, exist_ok=True)
        alignment_reports = []
        for video in discovered:
            video_id = video.get('video_id')
            ref_dir = Path(video['reference_dir']) if video.get('reference_dir') else reference_dir
            if not video_id or not ref_dir.exists():
                continue
            cmd = ["python", "tools/check_frame_alignment.py", "--reference-dir", str(ref_dir), "--frames-dir", str(ref_dir), "--video-id", str(video_id), "--samples", "10", "--outdir", str(ar_out / video_id)]
            subprocess.run(cmd, check=False)
            rep = ar_out / video_id / 'alignment_report.json'
            if rep.exists():
                alignment_reports.append({video_id: json.loads(rep.read_text(encoding='utf-8'))})
        report['alignment_report'] = alignment_reports if alignment_reports else {'present': False}
    except Exception as exc:
        report['alignment_report_error'] = str(exc)

    out_path = out_root / 'dataset_validation_report.json'
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return out_path


if __name__ == '__main__':
    ref = Path('ALS29T7_data') if Path('ALS29T7_data').exists() else Path('als29t7_data') if Path('als29t7_data').exists() else Path('.')
    pipeline_cfg = REPO_ROOT / 'configs' / 'pipeline.yaml'
    dataset_root = Path('New_experiments_v3_final') if Path('New_experiments_v3_final').exists() else ref
    if pipeline_cfg.exists():
        try:
            cfg = yaml.safe_load(pipeline_cfg.read_text(encoding='utf-8')) or {}
            dataset_root = Path(cfg.get('data', {}).get('reference_root', 'New_experiments_v3_final'))
        except Exception:
            pass
    manual_cfg = Path('configs') / 'video_manual_counts.json'
    p = main(ref, manual_cfg if manual_cfg.exists() else None, dataset_root)
    print(f'Wrote {p}')
