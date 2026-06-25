"""Scan repository for bare excepts and silent failures and emit a report.
This tool does not modify code; it reports locations to review.
"""
from __future__ import annotations

import re
from pathlib import Path
import json


def scan(root: Path):
    findings = []
    for p in root.rglob('*.py'):
        try:
            text = p.read_text(encoding='utf-8')
        except Exception:
            continue
        for m in re.finditer(r"^\s*except\s*:\s*$", text, flags=re.MULTILINE):
            line_no = text[:m.start()].count('\n') + 1
            findings.append({'file': str(p), 'line': line_no, 'type': 'bare_except'})
        for m in re.finditer(r"except\s+Exception\s*:\s*pass", text):
            line_no = text[:m.start()].count('\n') + 1
            findings.append({'file': str(p), 'line': line_no, 'type': 'except_exception_pass'})
    return findings

if __name__ == '__main__':
    root = Path('.')
    rep = scan(root)
    print(json.dumps(rep, indent=2))
