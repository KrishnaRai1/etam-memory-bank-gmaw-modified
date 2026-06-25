"""Phase 7: Colab hardening audit.
Scans repository for hardcoded paths, non-pathlib usage, and error handling issues.
Generates outputs/colab_hardening_audit.json with findings.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any
import json


def audit_hardcoded_paths() -> list[dict[str, Any]]:
    """Find hardcoded Windows/Linux absolute paths."""
    findings = []
    patterns = [
        r"['\"]C:\\",  # Windows absolute
        r"['\"]D:\\",  # Windows drive
        r"['\"]E:\\",  # Windows drive
        r"['\"]\/home\/",  # Linux absolute
        r"['\"]\/root\/",  # Linux root
        r"['\"]\/tmp\/",  # Linux tmp
    ]
    
    for py_file in Path('.').rglob('*.py'):
        try:
            content = py_file.read_text(encoding='utf-8')
        except Exception:
            continue
        
        for pattern in patterns:
            for match in re.finditer(pattern, content):
                line_no = content[:match.start()].count('\n') + 1
                line_text = content.split('\n')[line_no - 1].strip()
                findings.append({
                    'file': str(py_file),
                    'line': line_no,
                    'pattern': pattern,
                    'context': line_text[:100]
                })
    
    return findings


def audit_error_handling() -> list[dict[str, Any]]:
    """Find bare excepts and silent failures."""
    findings = []
    
    for py_file in Path('.').rglob('*.py'):
        try:
            content = py_file.read_text(encoding='utf-8')
        except Exception:
            continue
        
        # Bare except
        for match in re.finditer(r"^\s*except\s*:\s*$", content, flags=re.MULTILINE):
            line_no = content[:match.start()].count('\n') + 1
            findings.append({
                'file': str(py_file),
                'line': line_no,
                'issue': 'bare_except',
                'severity': 'high'
            })
        
        # except Exception: pass (silent)
        for match in re.finditer(r"except\s+Exception\s*:\s*pass", content):
            line_no = content[:match.start()].count('\n') + 1
            findings.append({
                'file': str(py_file),
                'line': line_no,
                'issue': 'silent_exception_pass',
                'severity': 'medium'
            })
    
    return findings


def audit_pathlib_usage() -> list[dict[str, Any]]:
    """Find string-based path operations instead of pathlib."""
    findings = []
    patterns = [
        (r"open\(['\"][^'\"]*['\"]", "open() with string path"),  # open("path")
        (r"os\.path\.", "os.path usage"),
        (r"from os\.path import", "os.path import"),
    ]
    
    for py_file in Path('.').rglob('*.py'):
        try:
            content = py_file.read_text(encoding='utf-8')
        except Exception:
            continue
        
        for pattern, issue in patterns:
            for match in re.finditer(pattern, content):
                line_no = content[:match.start()].count('\n') + 1
                findings.append({
                    'file': str(py_file),
                    'line': line_no,
                    'issue': issue,
                    'severity': 'low'
                })
    
    return findings


def main() -> dict[str, Any]:
    """Run full audit."""
    report = {
        'hardcoded_paths': audit_hardcoded_paths(),
        'error_handling_issues': audit_error_handling(),
        'pathlib_issues': audit_pathlib_usage(),
    }
    
    total_issues = sum(len(v) for v in report.values())
    critical = sum(1 for v in report['hardcoded_paths'] if v)
    high_severity = sum(1 for v in report['error_handling_issues'] if v.get('severity') == 'high')
    
    report['summary'] = {
        'total_issues': total_issues,
        'critical_hardcoded_paths': len(report['hardcoded_paths']),
        'high_severity_errors': high_severity,
        'colab_ready': len(report['hardcoded_paths']) == 0 and high_severity == 0
    }
    
    return report


if __name__ == '__main__':
    rep = main()
    out = Path('outputs') / 'colab_hardening_audit.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2), encoding='utf-8')
    print(f'Colab hardening audit written to {out}')
    print(f"Colab ready: {rep['summary']['colab_ready']}")
