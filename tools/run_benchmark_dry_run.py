"""Phase 6: Benchmark dry run.
Runs one difficult interval with memory_update_skip=1 only.
Collects outputs and reports success/issues.

Generates outputs/benchmark_dry_run_report.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_dry_run(
    interval_id: str = "AIS26T1_8245_8330",
    category: str = "id_switches",
    reference_dir: Path | None = None
) -> dict[str, Any]:
    """Run benchmark on one interval and collect outputs."""
    reference_dir = reference_dir or Path("New_experiments_v3_final")
    report: dict[str, Any] = {
        'interval_id': interval_id,
        'category': category,
        'reference_dir': str(reference_dir),
        'command': '',
        'exit_code': None,
        'stdout': '',
        'stderr': '',
        'outputs_found': {},
        'issues': []
    }

    # Build benchmark command using the current Python runtime
    cmd = [
        sys.executable, "-m", "src.run_benchmark",
        "--category", category,
        "--interval-id", interval_id,
        "--memory-update-skip", "1",
        "--reference-dir", str(reference_dir.resolve())
    ]
    report['command'] = ' '.join(cmd)

    print(f"Running dry-run: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        report['exit_code'] = result.returncode
        report['stdout'] = result.stdout
        report['stderr'] = result.stderr if result.stderr else ""
    except subprocess.TimeoutExpired:
        report['issues'].append('Benchmark timed out after 600 seconds')
        return report
    except Exception as exc:
        report['issues'].append(f'Failed to run benchmark: {exc}')
        return report

    if report['exit_code'] != 0:
        report['issues'].append(f'Non-zero exit code: {report["exit_code"]}')

    # Check for expected outputs
    outputs_root = Path('outputs')
    
    # Check benchmark summary
    csv_path = outputs_root / 'benchmark_summary' / 'benchmark_summary.csv'
    json_path = outputs_root / 'benchmark_summary' / 'benchmark_summary.json'
    report['outputs_found']['benchmark_summary_csv'] = csv_path.exists()
    report['outputs_found']['benchmark_summary_json'] = json_path.exists()

    # Check aggregated report
    report_json = outputs_root / 'benchmark_summary' / 'benchmark_report.json'
    report['outputs_found']['benchmark_report_json'] = report_json.exists()

    # Check manual count metrics
    manual_path = outputs_root / 'manual_count_metrics.json'
    report['outputs_found']['manual_count_metrics_json'] = manual_path.exists()

    # Check per-run outputs based on interval prefix
    interval_prefix = interval_id.split('_', 1)[0] if '_' in interval_id else interval_id
    run_output_dir = outputs_root / 'benchmark_runs' / interval_prefix / interval_id / 'skip_1'
    report['outputs_found']['run_output_dir'] = run_output_dir.exists()
    
    if run_output_dir.exists():
        report['outputs_found']['count_metrics_json'] = (run_output_dir / 'count_metrics.json').exists()
        report['outputs_found']['mask_metrics_json'] = (run_output_dir / 'mask_metrics.json').exists()
        report['outputs_found']['track_metrics_json'] = (run_output_dir / 'track_metrics.json').exists()
        report['outputs_found']['experiment_log'] = bool(list(run_output_dir.glob('experiment_*.json')))

    # Overall success
    all_found = all(report['outputs_found'].values())
    report['success'] = report['exit_code'] == 0 and all_found

    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval-id', default='AIS26T1_8245_8330')
    parser.add_argument('--category', default='id_switches')
    parser.add_argument('--reference-dir', default='New_experiments_v3_final')
    parser.add_argument('--out', default='outputs/benchmark_dry_run_report.json')
    args = parser.parse_args()

    rep = run_dry_run(
        interval_id=args.interval_id,
        category=args.category,
        reference_dir=Path(args.reference_dir)
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2), encoding='utf-8')
    print(f'Dry-run report written to {args.out}')
