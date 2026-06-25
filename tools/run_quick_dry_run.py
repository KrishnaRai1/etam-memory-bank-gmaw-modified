"""Run a single benchmark interval in a controlled output root with timeout protection.
Writes outputs/dry_run_test/quick_dry_run_report.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml


def _run_benchmark_subprocess(output_root: str, interval_id: str, category: str, reference_root: str, video_id: str | None, timeout_seconds: int) -> dict[str, Any]:
    import os as os_module
    
    repo_root = Path(__file__).resolve().parents[1]
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    pipeline_cfg = repo_root / 'configs' / 'pipeline.yaml'
    data = yaml.safe_load(pipeline_cfg.read_text(encoding='utf-8'))
    data.setdefault('data', {})['output_root'] = str(output_root_path.resolve())
    override_cfg = output_root_path / 'pipeline_override.yaml'
    override_cfg.write_text(yaml.safe_dump(data), encoding='utf-8')

    child_script = output_root_path / 'run_quick_dry_run_child.py'
    child_code = f"""from pathlib import Path
import sys
import json
import traceback
import yaml
import os
import site

repo_root = Path({json.dumps(str(repo_root))})
sys.path.insert(0, str(repo_root))

# Add venv site-packages to path
# This dynamically finds site-packages for the current Python
site_packages = site.getsitepackages()
for sp in site_packages:
    if sp not in sys.path:
        sys.path.insert(0, sp)

# Also try user site-packages
user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)

os.chdir(repo_root)
from src.run_benchmark import execute_benchmarks

try:
    result = execute_benchmarks(
        pipeline_cfg_path={json.dumps(str(override_cfg))},
        benchmark_yaml_path={json.dumps(str(repo_root / 'configs' / 'benchmark_cases.yaml'))},
        category={json.dumps(category)},
        video_id={json.dumps(video_id)},
        interval_id={json.dumps(interval_id)},
        memory_update_skips=[1],
        reuse_existing_outputs=False,
        reference_root={json.dumps(reference_root)},
        dry_run=False,
    )
    out = {{'success': True, 'result': result}}
except Exception as exc:
    out = {{'success': False, 'error': traceback.format_exc()}}
Path({json.dumps(str(output_root_path / 'quick_dry_run_process_result.json'))}).write_text(json.dumps(out, default=str), encoding='utf-8')
"""
    child_script.write_text(child_code, encoding='utf-8')

    # Create environment with proper paths
    env = os_module.environ.copy()
    env['PYTHONPATH'] = str(repo_root) + os_module.pathsep + env.get('PYTHONPATH', '')

    proc = subprocess.run([
        sys.executable,
        str(child_script),
    ], capture_output=True, text=True, timeout=timeout_seconds, cwd=str(repo_root), env=env)

    result_path = output_root_path / 'quick_dry_run_process_result.json'
    if not result_path.exists():
        return {
            'success': False,
            'error': 'Child process did not produce result file.',
            'stdout': proc.stdout,
            'stderr': proc.stderr,
            'returncode': proc.returncode,
        }

    content = json.loads(result_path.read_text(encoding='utf-8'))
    content['stdout'] = proc.stdout
    content['stderr'] = proc.stderr
    content['returncode'] = proc.returncode
    return content


def run_quick_dry_run(interval_id: str = 'AIS29T7_385_395', category: str = 'segmentation_errors', reference_root: str = 'New_experiments_v3_final', output_root: str = 'outputs/dry_run_test', timeout_seconds: int = 300, video_id: str | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        'interval_id': interval_id,
        'category': category,
        'reference_root': reference_root,
        'output_root': output_root,
        'timeout_seconds': timeout_seconds,
        'status': 'UNKNOWN',
        'result': None,
    }

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    # Setup pipeline config override
    repo_root = Path(__file__).resolve().parents[1]
    pipeline_cfg = repo_root / 'configs' / 'pipeline.yaml'
    data = yaml.safe_load(pipeline_cfg.read_text(encoding='utf-8'))
    data.setdefault('data', {})['output_root'] = str(output_root_path.resolve())
    override_cfg = output_root_path / 'pipeline_override.yaml'
    override_cfg.write_text(yaml.safe_dump(data), encoding='utf-8')

    try:
        # Import and run benchmark directly in parent process
        import sys
        sys.path.insert(0, str(repo_root))
        from src.run_benchmark import execute_benchmarks

        result = execute_benchmarks(
            pipeline_cfg_path=str(override_cfg),
            benchmark_yaml_path=str(repo_root / 'configs' / 'benchmark_cases.yaml'),
            category=category,
            video_id=video_id,
            interval_id=interval_id,
            memory_update_skips=[1],
            reuse_existing_outputs=False,
            reference_root=reference_root,
            dry_run=False,
        )
        report['result'] = {'success': True, 'result': result}
        report['status'] = 'SUCCESS'
    except Exception:
        report['result'] = {'success': False, 'error': traceback.format_exc()}
        report['status'] = 'FAILED'

    report['artifacts'] = _inspect_artifacts(output_root_path, interval_id, video_id)
    output_file = output_root_path / 'quick_dry_run_report.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def _inspect_artifacts(output_root: Path, interval_id: str, video_id: str | None = None) -> dict[str, Any]:
    data = {
        'benchmark_summary_csv': False,
        'benchmark_summary_json': False,
        'benchmark_report_json': False,
        'manual_count_metrics_json': False,
        'run_output_dir': False,
        'run_output_contents': [],
    }
    benchmark_summary = output_root / 'benchmark_summary'
    data['benchmark_summary_csv'] = (benchmark_summary / 'benchmark_summary.csv').exists()
    data['benchmark_summary_json'] = (benchmark_summary / 'benchmark_summary.json').exists()
    data['benchmark_report_json'] = (benchmark_summary / 'benchmark_report.json').exists()
    data['manual_count_metrics_json'] = (output_root / 'manual_count_metrics.json').exists()
    run_dir = output_root / 'benchmark_runs' / (video_id or 'unknown') / interval_id / 'skip_1'
    data['run_output_dir'] = run_dir.exists()
    if run_dir.exists():
        data['run_output_contents'] = sorted([str(p.name) for p in run_dir.iterdir()])
    return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run a controlled quick benchmark dry run for AIS29T7.')
    parser.add_argument('--interval-id', default='AIS29T7_385_395')
    parser.add_argument('--category', default='segmentation_errors')
    parser.add_argument('--reference-dir', default='New_experiments_v3_final')
    parser.add_argument('--video-id', default=None)
    parser.add_argument('--output-root', default='outputs/dry_run_test')
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()

    rep = run_quick_dry_run(
        interval_id=args.interval_id,
        category=args.category,
        reference_root=args.reference_dir,
        output_root=args.output_root,
        timeout_seconds=args.timeout,
        video_id=args.video_id,
    )
    print(f'Quick dry run report written to {Path(args.output_root) / "quick_dry_run_report.json"}')
    print(f'Status: {rep["status"]}')
    if rep.get('result', {}).get('error'):
        print(f'Error: {rep["result"]["error"]}')
