"""Power analysis for the task-averaged paired non-inferiority test (appendix derivation). Analysis only.

Usage:
    python analysis/power_appendix.py --runs reports/js_formal_test.csv --a js@8 --b full [--delta 5]

Steps:
1. Per task t, d_t = mean over reps of success(a) - mean over reps of success(b); D = mean_t d_t.
2. SE(D) from the task-group cluster bootstrap (B = 10000, seed 20261022), as in the main analysis.
3. Run-level (within-task) part of Var(D): sum_t [s2_a,t / R_a + s2_b,t / R_b] / T^2, with s2 the unbiased
   within-task variance of the 0/1 outcomes. The rest is the between-task part, which more reps cannot reduce.
4. Power for a true difference mu: P(D_hat - 1.96 SE > -delta) = Phi((mu + delta) / SE - 1.96), normal approximation.
   SE for R reps: sqrt(between + run_part * R_obs / R); for T' tasks: scaled by T / T'.
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prism_thesis import stats  # noqa: E402


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--delta', type=float, default=5.0)
    x = ap.parse_args()
    recs = []
    for r in csv.DictReader(open(x.runs, encoding='utf-8')):
        if r['status'] == 'infrastructure_error' or r['condition'] not in (x.a, x.b):
            continue
        recs.append({'condition': r['condition'], 'task_id': int(r['task_id']), 'group_id': r['group_id'],
                     'success': r['success'] in ('True', 'true', '1')})
    g = {r['task_id']: r['group_id'] for r in recs}
    d = stats.paired_difference(recs, x.a, x.b, g, b=10000, seed=20261022)
    se = (d.high - d.low) / (2 * 1.96)
    by = collections.defaultdict(list)
    for r in recs:
        by[(r['condition'], r['task_id'])].append(r['success'])
    tasks = sorted({t for (_, t) in by if (x.a, t) in by and (x.b, t) in by})
    T = len(tasks)
    run_part, reps = 0.0, []
    for t in tasks:
        for c in (x.a, x.b):
            v = by[(c, t)]
            n = len(v)
            reps.append(n)
            p = sum(v) / n
            run_part += (p * (1 - p) * n / (n - 1)) / n if n > 1 else 0.0
    run_part /= T ** 2
    total = se ** 2
    between = max(total - run_part, 0.0)
    R = round(sum(reps) / len(reps))
    print(f'{x.a} - {x.b}: D = {100 * d.estimate:+.2f} pp, 95% CI [{100 * d.low:+.2f}, {100 * d.high:+.2f}], '
          f'T = {T} tasks, {len(set(g[t] for t in tasks))} groups, R = {R} reps')
    print(f'SE = {100 * se:.2f} pp; run-level share of variance = {100 * run_part / total:.1f}%, '
          f'between-task share = {100 * between / total:.1f}%')
    print('\nCI half-width and power (true difference 0) by reps, 114 tasks:')
    for r in (5, 10, 20, 50, 10 ** 9):
        s = math.sqrt(between + run_part * R / r)
        print(f'  R = {"inf" if r > 10 ** 6 else r:>4}: half-width {100 * 1.96 * s:.2f} pp, '
              f'power {phi(x.delta / 100 / s - 1.96):.2f}')
    print('\nPower by true difference mu, R as observed:')
    for mu in (-2, 0, 2, 3, 5):
        print(f'  mu = {mu:+d} pp: {phi((mu + x.delta) / 100 / se - 1.96):.2f}')
    print('\nTasks needed (R as observed, scaled SE ~ 1/sqrt(T)):')
    for target in (0.5, 0.8):
        for mu in (0, 2):
            z = (mu + x.delta) / 100
            # Phi(z/s - 1.96) = target -> s = z / (1.96 + z_target)
            zt = {0.5: 0.0, 0.8: 0.8416}[target]
            s_needed = z / (1.96 + zt)
            print(f'  power {target:.0%} at mu = {mu:+d} pp: about {math.ceil(T * (se / s_needed) ** 2)} tasks')


if __name__ == '__main__':
    main()
