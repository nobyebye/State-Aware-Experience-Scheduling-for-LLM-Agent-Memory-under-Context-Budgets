"""Export run records to a compact CSV (one row per run) small enough to commit for offline analysis.

Usage:
    python analysis/export_compact.py --runs results/js/formal_calib.jsonl --out reports/formal_calib.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PHASES = ('initial', 'after_success', 'after_error')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cols = ['condition', 'split', 'task_id', 'group_id', 'rep', 'task_type_pred', 'success', 'status', 'n_turns',
            'input_tokens', 'output_tokens', 'cost_usd_est', 'n_errors', 'actions', 'dropped_by_turn'] + \
           [f'tok_{p}' for p in PHASES] + [f'n_{p}' for p in PHASES] + ['fingerprint', 'model']
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(a.out, 'w', newline='', encoding='utf-8') as fo:
        w = csv.DictWriter(fo, fieldnames=cols)
        w.writeheader()
        for p in a.runs:
            with open(p, encoding='utf-8') as f:
                for line in f:
                    r = json.loads(line)
                    if r.get('status') == 'infrastructure_error':
                        continue
                    tok = {ph: 0 for ph in PHASES}
                    cnt = {ph: 0 for ph in PHASES}
                    for t in r.get('turns', []):
                        tok[t['phase']] += t['tokens']['input']
                        cnt[t['phase']] += 1
                    turns = r.get('turns', [])
                    row = {k: r.get(k) for k in cols if k in r}
                    row.update({
                        'n_errors': sum(1 for t in turns if (t.get('exec') or {}).get('ok') is False),
                        'actions': '|'.join(str(t.get('action')) for t in turns),
                        'dropped_by_turn': '|'.join('+'.join((t.get('plan_info') or {}).get('dropped', []))
                                                    for t in turns),
                        **{f'tok_{ph}': tok[ph] for ph in PHASES}, **{f'n_{ph}': cnt[ph] for ph in PHASES},
                    })
                    w.writerow(row)
                    n += 1
    print(f'{n} runs -> {a.out}')


if __name__ == '__main__':
    main()
