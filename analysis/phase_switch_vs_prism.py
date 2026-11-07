"""Diagnose why phase_switch beats prism@8 on the test set (analysis only, no API calls).

Usage:
    python analysis/phase_switch_vs_prism.py results/compare/test.jsonl [results/risk/ablation.jsonl]

Prints:
  1. success / tokens per condition and per interaction phase;
  2. outcome decomposition (first SQL error, SQL right but answer wrong, SQL wrong, extra operations);
  3. per-task paired difference phase_switch - prism@8 with a group-cluster bootstrap CI;
  4. what PRISM dropped in each phase;
  5. answer-transcription error rate after a correct SQL, against after_success prompt size;
  6. (optional) point estimates of every dev-set ablation (full - ablated, pp).
"""
from __future__ import annotations

import collections
import json
import random
import sys


def load(path):
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r.get('status') == 'infrastructure_error':
                continue
            out[r['run_key']] = r
    return list(out.values())


def first_ok(r):
    return bool(r['turns'][0].get('exec', {}).get('ok'))


def correct_observations(records):
    """Observations of a first SQL that led straight to a correct answer, per task (any condition)."""
    good = collections.defaultdict(set)
    for r in records:
        ts = r['turns']
        if r['success'] and len(ts) == 2 and first_ok(r) and ts[1]['action'] == 'answer':
            good[r['task_id']].add(str(ts[0]['observation']))
    return good


def outcome(r, good):
    ts = r['turns']
    s = 'S' if r['success'] else 'F'
    if not first_ok(r):
        return f'first_error_{s}'
    if len(ts) > 1 and ts[1]['action'] == 'operation':
        return f'extra_operation_{s}'
    if r['task_type_pred'] == 'modify':
        return f'modify_answered_{s}'
    if str(ts[0]['observation']) in good[r['task_id']]:
        return f'sql_right_{s}'
    return f'sql_wrong_{s}'


def main(test_path, ablation_path=None):
    R = load(test_path)
    C = collections.defaultdict(list)
    for r in R:
        C[r['condition']].append(r)
    good = correct_observations(R)
    main_conds = [c for c in ('full', 'nomem', 'phase_switch', 'prism@8', 'static@8') if c in C]

    print('== 1. success / tokens ==')
    for c, rs in C.items():
        n = len(rs)
        print(f"{c:15s} n={n:5d} success={sum(r['success'] for r in rs) / n:.3f} "
              f"tokens={sum(r['input_tokens'] for r in rs) / n:6.0f}")
    print('\nmean input tokens per turn, by phase (turns, tokens):')
    for c in main_conds:
        ph = collections.defaultdict(list)
        for r in C[c]:
            for t in r['turns']:
                ph[t['phase']].append(t['tokens']['input'])
        print(f'{c:13s}', {p: (len(v), round(sum(v) / len(v))) for p, v in sorted(ph.items())})

    print('\n== 2. outcome decomposition ==')
    for ty in ('query', 'modify'):
        print('--', ty)
        for c in main_conds:
            k = collections.Counter(outcome(r, good) for r in C[c] if r['task_type_pred'] == ty)
            print(f'{c:13s}', dict(sorted(k.items())))

    print('\n== 3. per-task difference phase_switch - prism@8 ==')
    S = collections.defaultdict(lambda: collections.defaultdict(list))
    G, T = {}, {}
    for r in R:
        S[r['condition']][r['task_id']].append(r['success'])
        G[r['task_id']], T[r['task_id']] = r['group_id'], r['task_type_pred']
    m = lambda c, t: sum(S[c][t]) / len(S[c][t])
    tasks = sorted(set(S['phase_switch']) & set(S['prism@8']))
    d = {t: m('phase_switch', t) - m('prism@8', t) for t in tasks}
    groups = collections.defaultdict(list)
    for t in tasks:
        groups[G[t]].append(t)
    gs, rng, bs = list(groups), random.Random(20261022), []
    for _ in range(10000):
        ts = [t for g in (rng.choice(gs) for _ in gs) for t in groups[g]]
        bs.append(sum(d[t] for t in ts) / len(ts))
    bs.sort()
    print(f'mean {100 * sum(d.values()) / len(d):+.1f} pp, 95% CI [{100 * bs[250]:+.1f}, {100 * bs[-251]:+.1f}]')
    for ty in ('query', 'modify'):
        ts = [t for t in tasks if T[t] == ty]
        print(f'  {ty}: {len(ts)} tasks, {100 * sum(d[t] for t in ts) / len(ts):+.1f} pp')
    same = sum(1 for v in d.values() if abs(v) < 0.05)
    print(f'  tasks with no difference: {same}/{len(tasks)}; '
          f'share of total from |diff|>=0.3: {sum(v for v in d.values() if abs(v) >= .3) / sum(d.values()):.0%}')
    for t, v in sorted(d.items(), key=lambda kv: -abs(kv[1]))[:12]:
        print(f'  task {t:4d} {T[t]:6s} ' + ' '.join(f'{c}={m(c, t):.1f}' for c in main_conds))

    print('\n== 4. what prism@8 dropped, per phase ==')
    dropped = collections.defaultdict(collections.Counter)
    for r in C['prism@8']:
        for t in r['turns']:
            pi = t.get('plan_info', {})
            key = 'FALLBACK(full prompt)' if pi.get('fallback') else '+'.join(pi.get('dropped', [])) or '(none)'
            dropped[(t['phase'], r['task_type_pred'])][key] += 1
    for k in sorted(dropped):
        print(k, dropped[k].most_common(5))

    print('\n== 5. answer wrong after a correct SQL (query tasks) ==')
    for c, rs in C.items():
        if 'llmlingua' in c:
            continue
        n = e = 0
        tok, mem = [], []
        for r in rs:
            ts = r['turns']
            if r['task_type_pred'] != 'query' or len(ts) < 2 or not first_ok(r):
                continue
            if str(ts[0]['observation']) not in good[r['task_id']]:
                continue
            n += 1
            e += not r['success']
            tok.append(ts[1]['tokens']['input'])
            mem.append(ts[1]['tokens'].get('memory', 0))
        if n:
            print(f'{c:13s} after_success tokens={sum(tok) / n:5.0f} (memory {sum(mem) / n:4.0f}) '
                  f'wrong answer rate={e / n:.3f} (n={n})')

    if ablation_path:
        print('\n== 6. dev-set ablation point estimates (full - ablated, pp; positive = worse) ==')
        A = load(ablation_path)
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        tt = {}
        for r in A:
            by[r['condition']][r['task_id']].append(r['success'])
            tt[r['task_id']] = r['task_type_pred']
        mm = lambda c, t: sum(by[c][t]) / len(by[c][t])
        for c in sorted(by):
            if c == 'full':
                continue
            row = []
            for ty in ('query', 'modify', 'any'):
                ts = [t for t in by[c] if t in by['full'] and (ty == 'any' or tt[t] == ty)]
                row.append(f'{100 * sum(mm("full", t) - mm(c, t) for t in ts) / len(ts):+6.1f}')
            print(f'{c:45s}', *row)


if __name__ == '__main__':
    main(*sys.argv[1:3])
