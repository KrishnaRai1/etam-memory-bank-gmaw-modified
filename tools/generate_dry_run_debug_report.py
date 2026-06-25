"""Dry run debug report generator.
Writes outputs/dry_run_debug_report.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def load_existing_dry_run_report() -> dict[str, Any]:
    path = Path('outputs/benchmark_dry_run_report.json')
    if not path.exists():
        return {'exists': False, 'path': str(path), 'content': None}
    try:
        content = json.loads(path.read_text(encoding='utf-8'))
        return {'exists': True, 'path': str(path), 'content': content}
    except Exception as exc:
        return {'exists': True, 'path': str(path), 'content': None, 'error': str(exc)}


def get_environment_diagnostics() -> dict[str, Any]:
    result = {
        'python_executable': sys.executable,
        'python_version': sys.version,
        'matplotlib_importable': False,
        'matplotlib_version': None,
        'src_run_benchmark_importable': False,
        'src_run_benchmark_error': None,
    }
    try:
        import matplotlib
        result['matplotlib_importable'] = True
        result['matplotlib_version'] = getattr(matplotlib, '__version__', None)
    except Exception as exc:
        result['matplotlib_importable'] = False
        result['matplotlib_version'] = str(exc)
    try:
        from src.run_benchmark import execute_benchmarks  # type: ignore
        result['src_run_benchmark_importable'] = True
    except Exception as exc:
        result['src_run_benchmark_importable'] = False
        result['src_run_benchmark_error'] = str(exc)
    return result


def collect_output_inspection() -> dict[str, Any]:
    root = Path('outputs')
    summary = {'paths': {}}
    for p in [root / 'benchmark_summary' / 'benchmark_summary.csv', root / 'benchmark_summary' / 'benchmark_summary.json', root / 'benchmark_summary' / 'benchmark_report.json', root / 'manual_count_metrics.json']:
        summary['paths'][str(p)] = p.exists()
    run_dir = root / 'benchmark_runs' / 'AIS29T7' / 'AIS29T7_385_395' / 'skip_1'
    summary['run_dir'] = {'path': str(run_dir), 'exists': run_dir.exists(), 'contents': []}
    if run_dir.exists():
        summary['run_dir']['contents'] = sorted([str(p.name) for p in run_dir.iterdir()])
    return summary


def analyze_failure(report: dict[str, Any]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    if not report.get('exists'):
        details['reason'] = 'benchmark dry run report missing'
        return details

    content = report.get('content')
    if content is None:
        details['reason'] = 'could not parse existing dry run report'
        return details

    details['command'] = content.get('command')
    details['exit_code'] = content.get('exit_code')
    details['stdout_present'] = bool(content.get('stdout'))
    details['stderr_present'] = bool(content.get('stderr'))
    details['stderr'] = content.get('stderr')
    if content.get('exit_code') != 0:
        details['failure'] = 'non_zero_exit_code'
    if 'ModuleNotFoundError' in (content.get('stderr') or ''):
        details['failure_detail'] = 'missing_module'
    return details


def generate_report(out_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        'dry_run_debug': {
            'existing_report': load_existing_dry_run_report(),
            'environment': get_environment_diagnostics(),
            'output_inspection': collect_output_inspection(),
        }
    }
    report['dry_run_debug']['analysis'] = analyze_failure(report['dry_run_debug']['existing_report'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate dry run debug report.')
    parser.add_argument('--out', default='outputs/dry_run_debug_report.json')
    args = parser.parse_args()

    rep = generate_report(Path(args.out))
    print(f'Dry run debug report written to {args.out}')
