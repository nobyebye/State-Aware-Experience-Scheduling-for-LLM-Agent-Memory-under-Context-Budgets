#!/usr/bin/env bash
# 70B transfer control: the 8B PRISM-JS policy (results/js/js_table.json, λ=8) used unchanged on Llama-3.3-70B,
# next to the 70B re-searched policy (results/js70/js_table.json), phase_switch and full, all in one new batch.
# Run in tmux from the repo root after
#   source .venv/bin/activate; export PYTHONPATH=src; export OPENROUTER_API_KEY=$(cat ~/.config/prism/openrouter.key)
# Test set, 114 tasks x 5 reps (as the first 70B batch), 4 conditions = 2280 runs, about $0.5, about 2 hours.
# The two PRISM-JS tables need different configs, so the batch is written to two files:
#   a8b.jsonl  (configs/transfer_70b_js8b.json): full, js@8 = 8B policy
#   a70.jsonl  (configs/transfer_70b.json):      phase_switch, js@8 = 70B policy
# The two files are run in alternating chunks of tasks so both halves cover the same time window.
# Safe to re-run: finished runs are skipped.
set -euo pipefail
WORKERS=${WORKERS:-8}
BUDGET=${BUDGET:-2}
OUT=results/js70x
mkdir -p "$OUT"
echo "== stage 0: waiting for other experiments $(date -u +%FT%TZ)"
while pgrep -f "prism_thesis.*run " >/dev/null || pgrep -f run_queue.sh >/dev/null || pgrep -f run_js2s.sh >/dev/null; do
  sleep 120
done
test -f results/js/js_table.json && test -f results/js70/js_table.json
sha256sum results/js/js_table.json results/js70/js_table.json | tee "$OUT/tables.sha256"
python -m prism_thesis --config configs/transfer_70b_js8b.json dry-run --split test --conditions "full,js@8" --limit 2 --out /tmp/js70x_dry.jsonl
echo "== stage 1: test batch $(date -u +%FT%TZ)"
for rep in 0 1 2 3 4; do
  echo "-- rep block $rep $(date -u +%FT%TZ)"
  python -m prism_thesis --config configs/transfer_70b_js8b.json run --split test --conditions "full,js@8" \
    --reps $((rep + 1)) --workers "$WORKERS" --out "$OUT/a8b.jsonl" --allow-paid --budget-usd "$BUDGET" || echo "a8b block reported errors"
  python -m prism_thesis --config configs/transfer_70b.json run --split test --conditions "phase_switch,js@8" \
    --reps $((rep + 1)) --workers "$WORKERS" --out "$OUT/a70.jsonl" --allow-paid --budget-usd "$BUDGET" || echo "a70 block reported errors"
done
echo "== stage 2: report"
python analysis/js70x_compare.py --a8b "$OUT/a8b.jsonl" --a70 "$OUT/a70.jsonl" --old results/js70/test.jsonl \
  | tee "$OUT/test_report.txt"
echo "== js70x done $(date -u +%FT%TZ)"
