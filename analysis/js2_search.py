"""PRISM-JS improved version (js2): rebuild the λ policy family with a new nested path for after_success|query.

Usage:
    python analysis/js2_search.py --runs-v1 results/js/search.jsonl --runs-v2 results/js2/search.jsonl \
        --out results/js2/js_table.json

Everything is as in analysis/js_search.py except one stratum. For after_success|query the pre-registered path of
the improved version (docs/PRISM-JS改进版_预先登记_20261110.md) removes rules and schema first and the retrieved
experience last, and its steps are measured in a new joint-omission batch on the development set (js-ref plus the
new js-abl conditions, same batch). The other three strata keep the v1 path and the v1 search runs. The λ family
is then built by the same two-item multiple-choice knapsack. Analysis only, no API calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import js_search  # noqa: E402
from prism_thesis import stats  # noqa: E402

# Pre-registered before any v2 search run (2026-11-10).
PATH_V2_AS_QUERY = [
    ['schema', 'one_stmt', 'scope'],
    ['fn_other', 'fn_predicted'],
    ['insight'],
    ['observe', 'rounds', 'modify_answer'],
    ['exp_rest'],
    ['exp_top1'],
]


# Calibration sequence, as in v1: λ = 2, 4, 8, 16, keeping only the first λ of each distinct policy.
SEQUENCE = [2, 4, 8, 16]


def cumulative_v2() -> list[list[str]]:
    out, acc = [], []
    for step in PATH_V2_AS_QUERY:
        acc = acc + step
        out.append(list(acc))
    return out


def conditions_v2() -> list[str]:
    """New search conditions, in run order (js-ref is re-run in the same batch)."""
    return ['js-ref'] + [f"js-abl:{'+'.join(s)}@after_success" for s in cumulative_v2()]


def stratum(recs, paths, phase, b):
    group_of = {r['task_id']: r['group_id'] for r in recs}
    rows = [{'step': 0, 'groups': [], 'loss': 0.0, 'upper': 0.0, 'save': 0.0}]
    for j, groups in enumerate(paths, 1):
        cond = f"js-abl:{'+'.join(groups)}@{phase}"
        d = stats.paired_difference(recs, 'js-ref', cond, group_of, b=b, seed=20261022)
        t = stats.paired_difference(recs, 'js-ref', cond, group_of, key='input_tokens', b=200, seed=20261022)
        rows.append({'step': j, 'groups': groups, 'loss': 100 * d.estimate, 'upper': 100 * d.high,
                     'save': t.estimate, 'n_tasks': d.n_tasks, 'n_groups': d.n_groups})
    return rows


def family(strata):
    fam, chosen = {}, {}
    for lam in js_search.LAMBDAS:
        spec, pick = {}, {}
        for ttype in js_search.TYPES:
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
        fam[str(lam)] = spec
        chosen[str(lam)] = pick
    return fam, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-v1', nargs='+', required=True)
    ap.add_argument('--runs-v2', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--b', type=int, default=10000)
    a = ap.parse_args()
    v1, v2 = js_search.load(a.runs_v1), js_search.load(a.runs_v2)
    for name, recs, conds in (('v1', v1, js_search.conditions()), ('v2', v2, conditions_v2())):
        missing = [c for c in conds if c not in {r['condition'] for r in recs}]
        if missing:
            raise SystemExit(f'missing {name} search conditions: {missing}')
    strata = {}
    for ttype in js_search.TYPES:
        r1 = [r for r in v1 if r['task_type_pred'] == ttype]
        strata[f'initial|{ttype}'] = stratum(r1, js_search.cumulative('initial'), 'initial', a.b)
        if ttype == 'query':
            r2 = [r for r in v2 if r['task_type_pred'] == ttype]
            strata['after_success|query'] = stratum(r2, cumulative_v2(), 'after_success', a.b)
        else:
            strata[f'after_success|{ttype}'] = stratum(r1, js_search.cumulative('after_success'), 'after_success', a.b)
    print(f"{'stratum':26s} {'step':>4s} {'loss':>7s} {'U97.5':>7s} {'save':>6s}  omitted")
    for name, rows in strata.items():
        tag = '  (v2 path)' if name == 'after_success|query' else ''
        for row in rows[1:]:
            print(f"{name:26s} {row['step']:4d} {row['loss']:+7.2f} {row['upper']:+7.2f} {row['save']:6.0f}  "
                  f"{'+'.join(row['groups'])}{tag}")
    fam, chosen = family(strata)
    print('\nfamily (task type: initial step / after-success step, summed upper bound, est. saving per task):')
    for lam, pick in chosen.items():
        print(f'  λ={lam:>2s}  ' + '   '.join(
            f"{t}: {p['initial_step']}/{p['after_success_step']} U={p['upper_sum']:.1f} save≈{p['save']:.0f}"
            for t, p in pick.items()))
    distinct, seen = [], set()
    for lam in SEQUENCE:
        k = json.dumps(fam[str(lam)], sort_keys=True)
        if k not in seen:
            seen.add(k)
            distinct.append(lam)
    print(f'calibration sequence (distinct policies among λ = {SEQUENCE}): {distinct}')
    paths = {**js_search.PATHS, 'after_success|query (v2)': PATH_V2_AS_QUERY}
    out = {'meta': {'version': 'js2', 'paths': paths, 'lambdas': js_search.LAMBDAS, 'distinct': distinct,
                    'b': a.b, 'seed': 20261022, 'runs_v1': a.runs_v1, 'runs_v2': a.runs_v2,
                    'n_records': [len(v1), len(v2)]},
           'strata': strata, 'chosen': chosen, 'family': fam}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    Path(a.out).with_name('distinct_lambdas.txt').write_text(' '.join(map(str, distinct)))
    print(f'\nwritten {a.out}')


if __name__ == '__main__':
    main()
