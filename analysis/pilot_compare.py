"""Pilot comparison of all conditions on the pilot half of the build set (quick look, not a formal test).

Usage:
    python analysis/pilot_compare.py --runs results/js/pilot.jsonl

Prints success, input tokens per task and per state, saving against full, and the task-weighted paired
success difference against full with a task-group cluster bootstrap 95% interval. Also splits by task type.
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

ORDER = ['full', 'nomem', 'js-ref', 'prism@8', 'prism-bundle@8', 'phase_switch', 'static@8', 'random_drop']


def load(paths):
    out = {}
    for p in paths:
        with open(p, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                if r.get('status') == 'infrastructure_error':
                    continue
                out[r['run_key']] = r
    return list(out.values())


def key(c):
    return (ORDER.index(c), 0) if c in ORDER else (len(ORDER), float(c.split('@')[1]) if c.startswith('js@') else 0)


def report(recs, title, b):
    group_of = {r['task_id']: r['group_id'] for r in recs}
    by = collections.defaultdict(list)
    for r in recs:
        by[r['condition']].append(r)
    if 'full' not in by:
        raise SystemExit('pilot runs must include full')
    full_tok = sum(r['input_tokens'] for r in by['full']) / len(by['full'])
    print(f'\n== {title}: {len({r["task_id"] for r in recs})} tasks')
    print(f"{'condition':16s} {'runs':>5s} {'succ':>6s} {'tok':>6s} {'save':>6s} {'diff vs full (95% CI)':>26s}"
          f"   tokens per request: initial / after_success / after_error")
    for c in sorted(by, key=key):
        rs = by[c]
        ph = collections.defaultdict(list)
        for r in rs:
            for t in r['turns']:
                ph[t['phase']].append(t['tokens']['input'])
        tok = sum(r['input_tokens'] for r in rs) / len(rs)
        diff = ''
        if c != 'full':
            d = stats.paired_difference(recs, c, 'full', group_of, b=b, seed=20261022)
            diff = f'{100 * d.estimate:+6.2f} [{100 * d.low:+6.2f},{100 * d.high:+6.2f}]'
        per = ' / '.join(f"{sum(ph[p]) / len(ph[p]):.0f}" if ph[p] else '-'
                         for p in ('initial', 'after_success', 'after_error'))
        print(f"{c:16s} {len(rs):5d} {100 * sum(r['success'] for r in rs) / len(rs):5.1f}% {tok:6.0f} "
              f"{100 * (1 - tok / full_tok):5.1f}% {diff:>26s}   {per}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--b', type=int, default=5000)
    a = ap.parse_args()
    recs = load(a.runs)
    report(recs, 'all tasks', a.b)
    for t in ('query', 'modify'):
        report([r for r in recs if r['task_type_pred'] == t], f'{t} tasks', a.b)


if __name__ == '__main__':
    main()
