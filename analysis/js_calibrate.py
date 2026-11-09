"""PRISM-JS formal calibration on the calibration half of the build set (configs/pilot_split.json).

Usage:
    python analysis/js_calibrate.py --runs results/js/formal_calib.jsonl --lambda-out results/js/lambda_star.txt

Fixed-sequence non-inferiority test against full (delta = 5pp, task-weighted paired difference, task-group
cluster bootstrap B=10000, seed 20261022) over the distinct policies of the frozen family in increasing order
of lambda; stops at the first failure. Also reports, without entering the sequence, the same test for
phase_switch (calibrated rule) and js-ref (state-aware retrieval only). Writes lambda* (or 'none').
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

SEQUENCE = [2, 4, 8, 16]          # distinct policies of the frozen family (6, 10, 12 equal 8)
OTHERS = ['js-ref', 'phase_switch']


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
    ap.add_argument('--runs', required=True)
    ap.add_argument('--lambda-out', required=True)
    ap.add_argument('--delta', type=float, default=5.0)
    a = ap.parse_args()
    recs = load(a.runs)
    calib = set(json.loads(Path('configs/pilot_split.json').read_text())['calibration_task_ids'])
    if {r['task_id'] for r in recs} - calib:
        raise SystemExit('records contain tasks outside the calibration half')
    group_of = {r['task_id']: r['group_id'] for r in recs}
    by = collections.defaultdict(list)
    for r in recs:
        by[r['condition']].append(r)
    tok_full = sum(r['input_tokens'] for r in by['full']) / len(by['full'])
    print(f"calibration half: {len({r['task_id'] for r in recs})} tasks")
    print(f"{'condition':14s} {'runs':>5s} {'succ':>6s} {'tok':>6s} {'save':>6s}  diff vs full (95% CI)")
    for c in ['full'] + OTHERS + [f'js@{l}' for l in SEQUENCE]:
        rs = by.get(c)
        if not rs:
            print(f'{c:14s} missing')
            continue
        tok = sum(r['input_tokens'] for r in rs) / len(rs)
        d = '' if c == 'full' else stats.paired_difference(recs, c, 'full', group_of, b=10000, seed=20261022)
        ds = '' if c == 'full' else f'{100 * d.estimate:+.2f} [{100 * d.low:+.2f}, {100 * d.high:+.2f}]'
        print(f"{c:14s} {len(rs):5d} {100 * sum(r['success'] for r in rs) / len(rs):5.1f}% {tok:6.0f} "
              f"{100 * (1 - tok / tok_full):5.1f}%  {ds}")
    best, steps = stats.fixed_sequence(recs, [(l, f'js@{l}') for l in SEQUENCE], 'full', group_of,
                                       a.delta / 100, b=10000, seed=20261022)
    print('\nfixed sequence (delta = %.1fpp):' % a.delta)
    for s in steps:
        print(f'  λ={s.epsilon:g}: {100 * s.diff.estimate:+.2f} [{100 * s.diff.low:+.2f}, {100 * s.diff.high:+.2f}] '
              f"{'pass' if s.passed else 'FAIL (stop)'}")
    for c in OTHERS:
        d = stats.paired_difference(recs, c, 'full', group_of, b=10000, seed=20261022)
        print(f"  {c} (reported, not in sequence): {'pass' if stats.noninferior(d, a.delta / 100) else 'fail'}")
    print(f'λ* = {best if best is not None else "none (fall back to full)"}')
    Path(a.lambda_out).write_text('none' if best is None else f'{best:g}')


if __name__ == '__main__':
    main()
