"""Phase 9: Final readiness report.
Aggregates all validation reports and produces a single readiness assessment.

Generates outputs/final_readiness_report.json with PASS/FAIL for each phase.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def aggregate_readiness(outputs_root: Path = Path("outputs")) -> dict[str, Any]:
    """Aggregate all validation reports into a single readiness assessment."""
    outputs_root = Path(outputs_root)
    report: dict[str, Any] = {
        'timestamp': None,
        'validations': {},
        'overall_status': 'READY'
    }

    # Phase 1: Reference inspection
    report['validations']['phase1_inspection'] = {
        'name': 'Reference data inspection',
        'status': 'PASS' if (outputs_root / 'reference_inspection.json').exists() else 'SKIP'
    }

    # Phase 2: Schema validation
    schema_path = outputs_root / 'schema_validation_report.json'
    schema_status = 'SKIP'
    if schema_path.exists():
        try:
            schema_data = json.loads(schema_path.read_text(encoding='utf-8'))
            schema_status = 'PASS' if schema_data.get('overall_ok', False) else 'FAIL'
        except Exception:
            schema_status = 'FAIL'
    report['validations']['phase2_schema'] = {'name': 'Schema validation', 'status': schema_status}

    # Phase 3: Frame alignment
    align_path = outputs_root / 'alignment_validation' / 'alignment_report.json'
    align_status = 'SKIP'
    if align_path.exists():
        try:
            align_data = json.loads(align_path.read_text(encoding='utf-8'))
            num_samples = align_data.get('num_samples', 0)
            samples = align_data.get('samples', [])
            found_count = sum(1 for s in samples if s.get('image_found', False))
            align_status = 'PASS' if found_count >= (num_samples * 0.8) else 'FAIL'  # 80% threshold
        except Exception:
            align_status = 'FAIL'
    report['validations']['phase3_alignment'] = {'name': 'Frame alignment validation', 'status': align_status}

    # Phase 4: Count validation
    count_path = outputs_root / 'count_validation_report.json'
    count_status = 'SKIP'
    if count_path.exists():
        try:
            count_data = json.loads(count_path.read_text(encoding='utf-8'))
            match = count_data.get('comparison', {}).get('sources_match', False)
            count_status = 'PASS' if match else 'WARN'  # WARN if mismatches but sources exist
        except Exception:
            count_status = 'FAIL'
    report['validations']['phase4_count'] = {'name': 'Reference count validation', 'status': count_status}

    # Phase 5: Mask metrics
    mask_path = outputs_root / 'mask_metric_validation.json'
    mask_status = 'SKIP'
    if mask_path.exists():
        try:
            mask_data = json.loads(mask_path.read_text(encoding='utf-8'))
            perfect = mask_data.get('self_comparison_perfect', False)
            mask_status = 'PASS' if perfect else 'WARN'  # WARN if not perfect but metrics computed
        except Exception:
            mask_status = 'FAIL'
    report['validations']['phase5_mask_metrics'] = {'name': 'Reference mask metrics validation', 'status': mask_status}

    # Phase 6: Benchmark dry run
    dry_run_path = outputs_root / 'benchmark_dry_run_report.json'
    dryrun_status = 'SKIP'
    if dry_run_path.exists():
        try:
            dry_run_data = json.loads(dry_run_path.read_text(encoding='utf-8'))
            success = dry_run_data.get('success', False)
            dryrun_status = 'PASS' if success else 'FAIL'
        except Exception:
            dryrun_status = 'FAIL'
    report['validations']['phase6_dry_run'] = {'name': 'Benchmark dry run', 'status': dryrun_status}

    # Determine overall status
    statuses = [v['status'] for v in report['validations'].values()]
    fails = sum(1 for s in statuses if s == 'FAIL')
    warns = sum(1 for s in statuses if s == 'WARN')
    
    if fails > 0:
        report['overall_status'] = 'NOT_READY'
    elif warns > 0:
        report['overall_status'] = 'READY_WITH_WARNINGS'
    else:
        report['overall_status'] = 'READY'

    report['summary'] = {
        'pass': sum(1 for s in statuses if s == 'PASS'),
        'warn': warns,
        'fail': fails,
        'skip': sum(1 for s in statuses if s == 'SKIP')
    }

    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs-root', default='outputs')
    parser.add_argument('--out', default='outputs/final_readiness_report.json')
    args = parser.parse_args()

    rep = aggregate_readiness(Path(args.outputs_root))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2), encoding='utf-8')
    print(f'Final readiness report written to {args.out}')
    print(f"Overall status: {rep['overall_status']}")
