"""Test-set report for the task-type-stratified PRISM-JS (js2s). Analysis only, no API calls.

Usage:
    python analysis/js2s_compare.py --new results/js2s/test.jsonl --old results/js2/test.jsonl

--new: the js2s batch (configs/js2s.json): full and js@1 (relabelled js2s).
--old: the earlier js2 test batch; its phase_switch and js@16 runs (relabelled phase_switch_b2, js2@16_b2) are shown
       as cross-batch references only. The primary comparison is js2s vs full from the same batch.
"""
from __future__ import annotations

import argparse
import collections

from js2_compare import load, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', required=True)
    ap.add_argument('--old')
    ap.add_argument('--delta', type=float, default=5.0)
    a = ap.parse_args()
    recs = load(a.new, rename={'js@1': 'js2s'})
    pairs = [('js2s', 'full')]
    if a.old:
        recs += load(a.old, keep={'phase_switch', 'js@16'},
                     rename={'phase_switch': 'phase_switch_b2', 'js@16': 'js2@16_b2'})
        pairs += [('js2s', 'phase_switch_b2'), ('js2s', 'js2@16_b2')]
    print('runs per condition:', dict(collections.Counter(r['condition'] for r in recs)))
    report(recs, 'all tasks', a.delta, pairs)
    for t in ('query', 'modify'):
        report([r for r in recs if r['task_type_pred'] == t], f'{t} tasks', a.delta, pairs)


if __name__ == '__main__':
    main()
