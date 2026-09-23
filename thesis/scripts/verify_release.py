"""Check public artifact hashes and selected aggregate arithmetic without research data."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def verify():
    rows = list(csv.DictReader((ROOT / 'SOURCE_MANIFEST.csv').read_text(encoding='utf-8').splitlines()))
    for row in rows:
        p = (REPO / row['path']).resolve()
        assert p.is_relative_to(ROOT.resolve()), row['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row['released_sha256'], row['path']
    cases = list(csv.DictReader((ROOT / 'results/verified_prefix_and_outer/prefix_quality_verified.csv').read_text(encoding='utf-8').splitlines()))
    checks = 0
    for row in cases:
        r = {k: float(v) for k, v in row.items() if k != 'method'}
        tn, fp, fn, tp = (r[k] for k in ('tn','fp','fn','tp'))
        expected = {
            'supported': tn + fp + fn + tp,
            'coverage': r['supported'] / r['requests'],
            'all_malicious_recall': tp / r['malicious_requests'],
            'macro_f1': (2*tp/(2*tp+fp+fn) + 2*tn/(2*tn+fp+fn))/2,
            'scores_per_supported': r['model_scores']/r['supported'],
            'malicious_abstentions': r['malicious_requests']-r['supported_malicious'],
            'benign_abstentions': r['requests']-r['malicious_requests']-r['supported_benign'],
        }
        assert r['short'] + r['censored'] + r['supported'] == r['requests']
        checks += 1
        for key, value in expected.items():
            assert math.isclose(r[key], value, rel_tol=1e-9, abs_tol=1e-9), (row['method'],key)
            checks += 1
    py_count = 0
    for p in ROOT.rglob('*.py'):
        ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
        py_count += 1
    result = dict(status='PASS', hashed_copied_files=len(rows), prefix_configurations=len(cases),
                  saved_arithmetic_checks=checks, python_files_parsed=py_count,
                  new_scientific_experiments=False, raw_labels_reverified=False)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    verify()
