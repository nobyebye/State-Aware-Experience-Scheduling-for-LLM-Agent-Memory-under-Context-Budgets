#!/usr/bin/env bash
# PRISM-JS stratified by task type (js2s), post-hoc analysis; see docs/PRISM-JS分题型校准_事后分析说明_20261112.md.
# Run in tmux from the repo root after
#   source .venv/bin/activate; export PYTHONPATH=src; export OPENROUTER_API_KEY=$(cat ~/.config/prism/openrouter.key)
# Stage 0: wait until no other experiment is running.
# Stage 1: build the policy (query: js2 λ=2, modify: js2 λ=16), no API calls.
# Stage 2: test set (114 tasks x 10 reps): full and js@1 (= js2s) in one new batch (~2280 runs).
# Stage 3: report. Safe to re-run: finished runs are skipped.
set -euo pipefail
WORKERS=${WORKERS:-8}
BUDGET=${BUDGET:-2}
QL=${QL:-2}
ML=${ML:-16}
mkdir -p results/js2s
echo "== stage 0: waiting for other experiments $(date -u +%FT%TZ)"
while pgrep -f "prism_thesis.*run " >/dev/null || pgrep -f run_queue.sh >/dev/null || pgrep -f run_js2.sh >/dev/null; do
  sleep 120
done
echo "== stage 1: stratified policy (query λ=$QL, modify λ=$ML)"
python analysis/js2s_table.py --family results/js2/js_table.json --query-lambda "$QL" --modify-lambda "$ML" \
  --out results/js2s/js_table.json | tee results/js2s/policy.txt
sha256sum results/js2s/js_table.json | tee results/js2s/js_table.sha256
python -m prism_thesis --config configs/js2s.json dry-run --split test --conditions "full,js@1" --limit 2 --out /tmp/js2s_dry.jsonl
echo "== stage 2: test set $(date -u +%FT%TZ)"
python -m prism_thesis --config configs/js2s.json run --split test --conditions "full,js@1" --reps 10 \
  --workers "$WORKERS" --out results/js2s/test.jsonl --allow-paid --budget-usd "$BUDGET" || echo "test run reported errors"
echo "== stage 3: report"
python analysis/js2s_compare.py --new results/js2s/test.jsonl --old results/js2/test.jsonl | tee results/js2s/test_report.txt
echo "== js2s done $(date -u +%FT%TZ)"
