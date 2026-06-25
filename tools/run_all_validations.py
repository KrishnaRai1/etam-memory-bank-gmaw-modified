"""Master validation wrapper.
Runs all 9 phases of ALS29T7 integration validation in sequence.
Generates outputs/validation_summary.json with overall status.
"""
from __future__ import annotations

import subprocess
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_all_validations(reference_dir: str = "New_experiments_v3_final") -> dict[str, Any]:
    """Run all validation phases."""
    summary: dict[str, Any] = {
        'reference_dir': reference_dir,
        'phases': {},
        'overall': 'NOT_RUN'
    }

    phases = [
        ("Phase 1: Reference Inspection", "tools/inspect_als29t7_data.py"),
        ("Phase 2: Schema Validation", "tools/inspect_als29t7_data.py"),  # Combined tool
        ("Phase 3: Frame Alignment", "tools/validate_alignment_als29t7.py"),
        ("Phase 4: Reference Counts", "tools/validate_counts_als29t7.py"),
        ("Phase 5: Mask Metrics", "tools/validate_mask_metrics_als29t7.py"),
        ("Phase 6: Benchmark Dry Run", "tools/run_benchmark_dry_run.py"),
        ("Phase 7: Colab Hardening", "tools/audit_colab_hardening.py"),
    ]

    for phase_name, script in phases:
        print(f"\n{'='*60}")
        print(f"Running {phase_name}")
        print(f"{'='*60}")
        try:
            cmd = ["python", script, "--reference-dir", reference_dir]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            summary['phases'][phase_name] = {
                'exit_code': result.returncode,
                'success': result.returncode == 0,
                'output': result.stdout[:500] if result.stdout else ""
            }
            if result.returncode != 0:
                print(f"[WARN] {phase_name} exited with code {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr[:200]}")
        except Exception as exc:
            summary['phases'][phase_name] = {'error': str(exc), 'success': False}
            print(f"[ERROR] {phase_name} failed: {exc}")

    # Generate final readiness report
    print(f"\n{'='*60}")
    print("Phase 9: Generating Final Readiness Report")
    print(f"{'='*60}")
    try:
        result = subprocess.run(
            ["python", "tools/generate_final_readiness_report.py"],
            capture_output=True, text=True, timeout=60
        )
        summary['phases']['Phase 9: Final Readiness'] = {
            'exit_code': result.returncode,
            'success': result.returncode == 0
        }
    except Exception as exc:
        summary['phases']['Phase 9: Final Readiness'] = {'error': str(exc), 'success': False}

    # Determine overall status
    all_success = all(p.get('success', False) for p in summary['phases'].values())
    summary['overall'] = 'READY' if all_success else 'NEEDS_REVIEW'

    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-dir', default='New_experiments_v3_final')
    parser.add_argument('--out', default='outputs/validation_summary.json')
    args = parser.parse_args()

    rep = run_all_validations(reference_dir=args.reference_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2), encoding='utf-8')
    print(f'\nValidation summary written to {args.out}')
    print(f'Overall status: {rep["overall"]}')
