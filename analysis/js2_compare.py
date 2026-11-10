"""Test-set report for the improved PRISM-JS (js2) next to the original version.

Usage:
    python analysis/js2_compare.py --new results/js2/test.jsonl --old results/js/formal_test.jsonl

--new: the js2 test batch (run with configs/js2.json): full, phase_switch and js@<λ*> (relabelled js2@<λ*>).
--old: the original formal test batch; only its js@8 runs are used (relabelled js1@8).
Prints per condition success, mean input tokens per run and per state, then task-weighted paired success
differences (task-group cluster bootstrap B=10000, seed 20261022, non-inferiority margin 5pp) and paired
input-token differences, for all tasks and by predicted task type. Analysis only, no API calls.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prism_thesis import stats  # noqa: E402


def load(path, keep=None, rename=None):
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r.get('status') == 'infrastructure_error':
                continue
            if keep is not None and r['condition'] not in keep:
                continue
            out[r['run_key']] = r
    recs = list(out.values())
    for r in recs:
        r['condition'] = (rename or {}).get(r['condition'], r['condition'])
    return recs


def wrong_after_success(r):
    turns = r.get('turns') or []
    return (not r['success']) and r.get('status') == 'answered' and bool(turns) and turns[-1]['phase'] == 'after_success'


def report(recs, title, delta, pairs):
    group_of = {r['task_id']: r['group_id'] for r in recs}
    by = collections.defaultdict(list)
    for r in recs:
        by[r['condition']].append(r)
    print(f"\n== {title}: {len({r['task_id'] for r in recs})} tasks")
    print(f"{'condition':14s} {'runs':>5s} {'succ':>6s} {'tok':>6s}  per turn initial / after_error / after_success"
          f"   wrong answer after successful SQL")
    for c in sorted(by):
        rs = by[c]
        ph = collections.defaultdict(list)
        for r in rs:
            for t in r['turns']:
                ph[t['phase']].append(t['tokens']['input'])
        per = ' / '.join(f"{sum(ph[p]) / len(ph[p]):.0f}" if ph[p] else '-'
                         for p in ('initial', 'after_error', 'after_success'))
        was = sum(wrong_after_success(r) for r in rs)
        print(f"{c:14s} {len(rs):5d} {100 * sum(r['success'] for r in rs) / len(rs):5.1f}% "
              f"{sum(r['input_tokens'] for r in rs) / len(rs):6.0f}  {per:>34s}   {was} ({100 * was / len(rs):.1f}%)")
    for a, b in pairs:
        if a not in by or b not in by:
            continue
        d = stats.paired_difference(recs, a, b, group_of, b=10000, seed=20261022)
        t = stats.paired_difference(recs, a, b, group_of, key='input_tokens', b=2000, seed=20261022)
        ni = 'non-inferior' if stats.noninferior(d, delta / 100) else 'not shown non-inferior'
        print(f"  {a} - {b}: success {100 * d.estimate:+.2f} [{100 * d.low:+.2f}, {100 * d.high:+.2f}] pp ({ni}); "
              f"input tokens per run {t.estimate:+.0f} [{t.low:+.0f}, {t.high:+.0f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', required=True)
    ap.add_argument('--old', required=True)
    ap.add_argument('--old-js', default='js@8')
    ap.add_argument('--delta', type=float, default=5.0)
    a = ap.parse_args()
    new = load(a.new)
    js_new = sorted(c for c in {r['condition'] for r in new} if c.startswith('js@'))
    if len(js_new) != 1:
        raise SystemExit(f'expected one js@<λ> condition in --new, found {js_new}')
    lab_new = 'js2@' + js_new[0][3:]
    lab_old = 'js1@' + a.old_js[3:]
    for r in new:
        if r['condition'] == js_new[0]:
            r['condition'] = lab_new
    old = load(a.old, keep={a.old_js}, rename={a.old_js: lab_old})
    recs = new + old
    counts = collections.Counter(r['condition'] for r in recs)
    print('runs per condition:', dict(counts))
    pairs = [(lab_new, 'full'), (lab_new, 'phase_switch'), (lab_new, lab_old), ('phase_switch', 'full'),
             (lab_old, 'full')]
    report(recs, 'all tasks', a.delta, pairs)
    for t in ('query', 'modify'):
        report([r for r in recs if r['task_type_pred'] == t], f'{t} tasks', a.delta, pairs)


if __name__ == '__main__':
    main()
