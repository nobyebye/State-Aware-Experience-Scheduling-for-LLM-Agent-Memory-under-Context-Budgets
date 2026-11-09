"""PRISM-JS search: from the joint-omission runs on the development set, build the λ-indexed policy family.

Usage:
    python analysis/js_search.py --runs results/js/search.jsonl --out results/js/js_table.json

For every stratum (state, task type) and every step j of the pre-registered nested path
(docs/预实验_PRISM-JS_预先登记_20261109.md), the joint loss of omitting the whole step-j set is estimated as
js-ref minus js-abl (task-weighted paired success difference, task-group cluster bootstrap), with its 97.5%
upper bound U and the mean input saving V per task. Step 0 (no omission) has U = V = 0.

For a budget λ and each task type, the policy picks one step for the initial state and one for the
after-success state, maximising V_init + V_as subject to max(U_init, 0) + max(U_as, 0) <= λ (a two-item
multiple-choice knapsack, solved by enumeration). Only two upper bounds are added, one per state, instead of
one per group. Analysis only, no API calls.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prism_thesis import stats  # noqa: E402

# Pre-registered nested paths (fixed before any search run).
PATHS = {
    'initial': [
        ['fn_other'],
        ['answer_query', 'empty_scalar'],
        ['insight'],
        ['observe', 'rounds', 'modify_answer'],
        ['exp_rest'],
    ],
    'after_success': [
        ['exp_rest'],
        ['exp_top1'],
        ['fn_other', 'fn_predicted'],
        ['schema', 'one_stmt', 'scope'],
        ['insight'],
        ['observe', 'rounds', 'modify_answer'],
        ['answer_query', 'empty_scalar'],
    ],
}
LAMBDAS = [2, 4, 6, 8, 10, 12, 16]
TYPES = ('query', 'modify')


def cumulative(phase: str) -> list[list[str]]:
    out, acc = [], []
    for step in PATHS[phase]:
        acc = acc + step
        out.append(list(acc))
    return out


def conditions() -> list[str]:
    """All search conditions, in run order."""
    return ['js-ref'] + [f"js-abl:{'+'.join(s)}@{ph}" for ph in PATHS for s in cumulative(ph)]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--b', type=int, default=10000)
    a = ap.parse_args()
    recs = load(a.runs)
    group_of = {r['task_id']: r['group_id'] for r in recs}
    have = {r['condition'] for r in recs}
    missing = [c for c in conditions() if c not in have]
    if missing:
        raise SystemExit(f'missing search conditions: {missing}')
    strata = {}
    print(f"{'stratum':26s} {'step':>4s} {'loss':>7s} {'U97.5':>7s} {'save':>6s}  omitted")
    for ttype in TYPES:
        rs = [r for r in recs if r['task_type_pred'] == ttype]
        for phase in PATHS:
            rows = [{'step': 0, 'groups': [], 'loss': 0.0, 'upper': 0.0, 'save': 0.0}]
            for j, groups in enumerate(cumulative(phase), 1):
                cond = f"js-abl:{'+'.join(groups)}@{phase}"
                d = stats.paired_difference(rs, 'js-ref', cond, group_of, b=a.b, seed=20261022)
                t = stats.paired_difference(rs, 'js-ref', cond, group_of, key='input_tokens', b=200, seed=20261022)
                rows.append({'step': j, 'groups': groups, 'loss': 100 * d.estimate, 'upper': 100 * d.high,
                             'save': t.estimate, 'n_tasks': d.n_tasks, 'n_groups': d.n_groups})
            strata[f'{phase}|{ttype}'] = rows
            for row in rows[1:]:
                print(f"{phase + '|' + ttype:26s} {row['step']:4d} {row['loss']:+7.2f} {row['upper']:+7.2f} "
                      f"{row['save']:6.0f}  {'+'.join(row['groups'])}")
    family, chosen = {}, {}
    for lam in LAMBDAS:
        spec, pick = {}, {}
        for ttype in TYPES:
            ini, suc = strata[f'initial|{ttype}'], strata[f'after_success|{ttype}']
            best = None
            for x in ini:
                for y in suc:
                    if max(x['upper'], 0) + max(y['upper'], 0) > lam:
                        continue
                    key = (x['save'] + y['save'], -(x['step'] + y['step']))
                    if best is None or key > best[0]:
                        best = (key, x, y)
            _, x, y = best
            spec[f'initial|{ttype}'] = x['groups']
            spec[f'after_success|{ttype}'] = y['groups']
            pick[ttype] = {'initial_step': x['step'], 'after_success_step': y['step'],
                           'upper_sum': max(x['upper'], 0) + max(y['upper'], 0), 'save': x['save'] + y['save']}
        family[str(lam)] = spec
        chosen[str(lam)] = pick
    print('\nfamily (task type: initial step / after-success step, summed upper bound, est. saving per task):')
    for lam, pick in chosen.items():
        print(f'  λ={lam:>2s}  ' + '   '.join(
            f"{t}: {p['initial_step']}/{p['after_success_step']} U={p['upper_sum']:.1f} save≈{p['save']:.0f}"
            for t, p in pick.items()))
    out = {'meta': {'paths': PATHS, 'lambdas': LAMBDAS, 'b': a.b, 'seed': 20261022,
                    'runs': [str(p) for p in a.runs], 'n_records': len(recs)},
           'strata': strata, 'chosen': chosen, 'family': family}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'\nwritten {a.out}')


if __name__ == '__main__':
    main()
