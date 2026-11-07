"""Diagnostic experiment on the dev set: why does removing whole blocks after a successful SQL help?

Usage:
    python analysis/after_success_diagnosis.py results/diag/after_success.jsonl \
        --obs results/diag/obs_examples.jsonl

The main file holds `full` and the after-success ablations; the --obs file holds `full` run with
configs/obs_examples.json (examples show each SQL's MySQL result), reported as `full+obs-examples`.
Reports per condition: success, paired difference to `full` (group-cluster bootstrap 95% CI),
first-SQL error rate, "correct SQL but wrong answer" rate on query tasks, and after-success prompt tokens.
Analysis only, no API calls.
"""
from __future__ import annotations

import argparse
import collections
import json
import random

SHORT = {
    'full': 'full',
    'ablate:exp_top1+exp_rest@after_success': '(a) drop all examples',
    'ablate:one_stmt+scope+fn_predicted+fn_other+schema@after_success': '(b) drop schema + rules',
    'ablate:schema+exp_top1+exp_rest@after_success': '(c) drop schema + examples',
    'ablate:one_stmt+scope+fn_predicted+fn_other+schema+exp_top1+exp_rest+repair@after_success':
        '(d) phase_switch bundle',
    'full+obs-examples': '(e) examples show MySQL result',
}


def load(path, rename=None):
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r.get('status') == 'infrastructure_error':
                continue
            if rename:
                r['condition'] = rename.get(r['condition'], r['condition'])
                r['run_key'] = r['condition'] + '|' + r['run_key'].split('|', 1)[1]
            out[r['run_key']] = r
    return list(out.values())


def first_ok(r):
    return bool(r['turns'][0].get('exec', {}).get('ok'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs')
    ap.add_argument('--obs')
    ap.add_argument('--b', type=int, default=10000)
    a = ap.parse_args()
    R = load(a.runs)
    if a.obs:
        R += load(a.obs, {'full': 'full+obs-examples'})
    good = collections.defaultdict(set)      # first-SQL results that led straight to a correct answer
    for r in R:
        ts = r['turns']
        if r['success'] and len(ts) == 2 and first_ok(r) and ts[1]['action'] == 'answer':
            good[r['task_id']].add(str(ts[0]['observation']))
    C = collections.defaultdict(list)
    for r in R:
        C[r['condition']].append(r)
    S = {c: collections.defaultdict(list) for c in C}
    G = {}
    for r in R:
        S[r['condition']][r['task_id']].append(r['success'])
        G[r['task_id']] = r['group_id']
    mean = lambda c, t: sum(S[c][t]) / len(S[c][t])

    def paired(c):
        tasks = sorted(set(S[c]) & set(S['full']))
        d = {t: mean(c, t) - mean('full', t) for t in tasks}
        groups = collections.defaultdict(list)
        for t in tasks:
            groups[G[t]].append(t)
        gs, rng, bs = list(groups), random.Random(20261022), []
        for _ in range(a.b):
            ts = [t for g in (rng.choice(gs) for _ in gs) for t in groups[g]]
            bs.append(sum(d[t] for t in ts) / len(ts))
        bs.sort()
        return 100 * sum(d.values()) / len(d), 100 * bs[int(.025 * a.b)], 100 * bs[int(.975 * a.b) - 1]

    def copy_errors(rs):
        n = e = 0
        for r in rs:
            ts = r['turns']
            if r['task_type_pred'] != 'query' or len(ts) < 2 or not first_ok(r):
                continue
            if str(ts[0]['observation']) not in good[r['task_id']]:
                continue
            n += 1
            e += not r['success']
        return e, n

    order = [c for c in SHORT if c in C] + sorted(set(C) - set(SHORT))
    print(f"{'condition':34s} {'runs':>5s} {'success':>8s} {'vs full (pp) [95% CI]':>24s} "
          f"{'1st SQL err':>11s} {'right SQL, wrong answer':>24s} {'after-success tok':>18s}")
    for c in order:
        rs = C[c]
        succ = sum(r['success'] for r in rs) / len(rs)
        diff = '' if c == 'full' else '{:+.1f} [{:+.1f}, {:+.1f}]'.format(*paired(c))
        fe = sum(not first_ok(r) for r in rs) / len(rs)
        e, n = copy_errors(rs)
        tok = [t['tokens']['input'] for r in rs for t in r['turns'] if t['phase'] == 'after_success']
        print(f"{SHORT.get(c, c):34s} {len(rs):5d} {succ:8.3f} {diff:>24s} {fe:11.3f} "
              f"{f'{e}/{n} = {e / n:.3f}' if n else '-':>24s} {sum(tok) / max(1, len(tok)):18.0f}")
    print('\nBy task type (success):')
    for c in order:
        row = []
        for ty in ('query', 'modify'):
            rs = [r for r in C[c] if r['task_type_pred'] == ty]
            row.append(f'{ty} {sum(r["success"] for r in rs) / len(rs):.3f} (n={len(rs)})')
        print(f'{SHORT.get(c, c):34s}', '  '.join(row))


if __name__ == '__main__':
    main()
