"""Report for the 70B transfer control: 8B PRISM-JS policy used unchanged on 70B vs the 70B re-searched policy.

Usage:
    python analysis/js70x_compare.py --a8b results/js70x/a8b.jsonl --a70 results/js70x/a70.jsonl \
        [--old results/js70/test.jsonl]

--a8b: configs/transfer_70b_js8b.json runs: full, js@8 (relabelled js8B_policy).
--a70: configs/transfer_70b.json runs from the same batch: phase_switch, js@8 (relabelled js70B_policy).
--old: the first 70B test batch (full_b1, phase_switch_b1, js70B_policy_b1), shown for reference only.
Task-weighted paired success differences, task-group cluster bootstrap B=10000, seed 20261022, margin 5pp.
Analysis only, no API calls.
"""
from __future__ import annotations

import argparse
import collections

from js2_compare import load, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a8b', required=True)
    ap.add_argument('--a70', required=True)
    ap.add_argument('--old')
    ap.add_argument('--delta', type=float, default=5.0)
    a = ap.parse_args()
    recs = load(a.a8b, keep={'full', 'js@8'}, rename={'js@8': 'js8B_policy'})
    recs += load(a.a70, keep={'phase_switch', 'js@8'}, rename={'js@8': 'js70B_policy'})
    pairs = [('js8B_policy', 'full'), ('js70B_policy', 'full'), ('js70B_policy', 'js8B_policy'),
             ('phase_switch', 'full'), ('js8B_policy', 'phase_switch'), ('js70B_policy', 'phase_switch')]
    if a.old:
        recs += load(a.old, keep={'full', 'phase_switch', 'js@8'},
                     rename={'full': 'full_b1', 'phase_switch': 'phase_switch_b1', 'js@8': 'js70B_policy_b1'})
    print('runs per condition:', dict(collections.Counter(r['condition'] for r in recs)))
    report(recs, 'all tasks', a.delta, pairs)
    for t in ('query', 'modify'):
        report([r for r in recs if r['task_type_pred'] == t], f'{t} tasks', a.delta, pairs)


if __name__ == '__main__':
    main()
