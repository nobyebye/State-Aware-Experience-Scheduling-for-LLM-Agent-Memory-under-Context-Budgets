"""Build the task-type-stratified PRISM-JS policy (js2s) from the js2 policy family. No API calls.

Usage:
    python analysis/js2s_table.py --family results/js2/js_table.json --query-lambda 2 --modify-lambda 16 \
        --out results/js2s/js_table.json

The policy takes every "<state>|query" entry from λ=--query-lambda and every "<state>|modify" entry from
λ=--modify-lambda. The λ values come from calibrating each task type separately on the build calibration half
(results/js2/calib.jsonl): the largest λ in the fixed sequence that passes the 5pp non-inferiority test within that
task type. The result is stored as a family with the single level "1", so it runs as condition js@1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def pick(fam, lam):
    key = next((k for k in fam if float(k) == float(lam)), None)
    if key is None:
        raise SystemExit(f'λ={lam} not in family {sorted(fam, key=float)}')
    return fam[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True)
    ap.add_argument('--query-lambda', type=float, required=True)
    ap.add_argument('--modify-lambda', type=float, required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    src = Path(a.family).read_bytes()
    fam = json.loads(src)['family']
    q, m = pick(fam, a.query_lambda), pick(fam, a.modify_lambda)
    policy = {k: v for k, v in q.items() if k.endswith('|query')}
    policy.update({k: v for k, v in m.items() if k.endswith('|modify')})
    out = {'source': a.family, 'source_sha256': hashlib.sha256(src).hexdigest(),
           'query_lambda': a.query_lambda, 'modify_lambda': a.modify_lambda, 'family': {'1': policy}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True), encoding='utf-8')
    for k in sorted(policy):
        print(f'{k:26s} omit {policy[k]}')


if __name__ == '__main__':
    main()
