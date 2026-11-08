"""Supplement PRISM-bundle: non-inferiority of prism-bundle@8 against full, and token use against prism@8.

Usage (construction set, gate before the test run):
    python analysis/bundle_check.py --treatment results/calibrate/bundle_calib.jsonl \
        --reference results/calibrate/calib_v2.jsonl
Usage (test set):
    python analysis/bundle_check.py --treatment results/compare/bundle_test.jsonl \
        --reference results/compare/test.jsonl

`--reference` must contain `full` and `prism@8` runs on the same split. Uses the same estimator as the main
analysis: task-weighted paired difference, task-group cluster bootstrap (B=10000, seed 20261022), delta = 5pp.
Analysis only, no API calls.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prism_thesis import stats  # noqa: E402

TREAT = 'prism-bundle@8'


def load(path):
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r.get('status') == 'infrastructure_error':
                continue
            out[r['run_key']] = r
    return list(out.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--treatment', required=True)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--delta', type=float, default=5.0)
    a = ap.parse_args()
    T = [r for r in load(a.treatment) if r['condition'] == TREAT]
    R = [r for r in load(a.reference) if r['condition'] in ('full', 'prism@8', 'phase_switch')]
    splits = {r['split'] for r in T} | {r['split'] for r in R}
    if len(splits) != 1:
        raise SystemExit(f'splits differ: {sorted(splits)}')
    recs = T + R
    group_of = {r['task_id']: r['group_id'] for r in recs}
    by = collections.defaultdict(list)
    for r in recs:
        by[r['condition']].append(r)
    print(f'split: {splits.pop()}')
    print(f"{'condition':16s} {'runs':>5s} {'success':>8s} {'tokens':>7s}   per-turn tokens by phase")
    for c in ('full', 'prism@8', 'phase_switch', TREAT):
        if c not in by:
            continue
        rs = by[c]
        ph = collections.defaultdict(list)
        for r in rs:
            for t in r['turns']:
                ph[t['phase']].append(t['tokens']['input'])
        print(f"{c:16s} {len(rs):5d} {sum(r['success'] for r in rs) / len(rs):8.3f} "
              f"{sum(r['input_tokens'] for r in rs) / len(rs):7.0f}   "
              + '  '.join(f'{p} {sum(v) / len(v):.0f}' for p, v in sorted(ph.items())))
    for ref in ('full', 'prism@8'):
        d = stats.paired_difference(recs, TREAT, ref, group_of, b=10000, seed=20261022)
        verdict = ''
        if ref == 'full':
            verdict = 'non-inferior (pass)' if 100 * d.low > -a.delta else 'NOT non-inferior (fail)'
        print(f'{TREAT} - {ref}: {100 * d.estimate:+.2f}pp  95% CI [{100 * d.low:+.2f}, {100 * d.high:+.2f}]  '
              f'({d.n_tasks} tasks / {d.n_groups} groups)  {verdict}')
    tok = {c: sum(r['input_tokens'] for r in by[c]) / len(by[c]) for c in ('full', 'prism@8', TREAT)}
    print(f"tokens: {TREAT} {tok[TREAT]:.0f} vs full {tok['full']:.0f} ({100 * (tok[TREAT] / tok['full'] - 1):+.1f}%), "
          f"vs prism@8 {tok['prism@8']:.0f} ({100 * (tok[TREAT] / tok['prism@8'] - 1):+.1f}%)")
    picks = collections.Counter()
    for r in by[TREAT]:
        for t in r['turns']:
            if t['phase'] == 'after_success':
                pi = t.get('plan_info', {})
                picks[(r['task_type_pred'], 'FALLBACK' if pi.get('fallback') else '+'.join(pi.get('bundles', [])) or '-')] += 1
    print('after-success choices (task type, bundle):', dict(picks.most_common()))


if __name__ == '__main__':
    main()
