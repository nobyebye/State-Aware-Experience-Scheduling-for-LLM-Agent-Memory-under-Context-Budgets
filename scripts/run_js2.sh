#!/usr/bin/env bash
# PRISM-JS improved version (js2), see docs/PRISM-JS改进版_预先登记_20261110.md.
# Run in tmux from the repo root after
#   source .venv/bin/activate; export PYTHONPATH=src; export OPENROUTER_API_KEY=$(cat ~/.config/prism/openrouter.key)
# Stage 0: wait until no other experiment is running (the MySQL containers allow one run at a time).
# Stage 1: dev search for the new after_success|query path: js-ref + 6 js-abl conditions x 110 tasks x 3 reps (2310 runs).
# Stage 2: js2 policy family (no API calls) -> results/js2/js_table.json, calibration sequence.
# Stage 3: calibration on the build calibration half (203 tasks x 10 reps): full + js2 policies (configs/js2.json).
# Stage 4: fixed-sequence test (delta 5pp) -> lambda*.
# Stage 5: test set (114 tasks x 10 reps): full, phase_switch, js2@lambda* in one batch.
# Stage 6: test report next to the original js@8.
# Safe to re-run: finished runs are skipped.
set -euo pipefail
WORKERS=${WORKERS:-8}
BUDGET=${BUDGET:-3}
mkdir -p results/js2
echo "== stage 0: waiting for other experiments $(date -u +%FT%TZ)"
while pgrep -f "prism_thesis.*run " >/dev/null || pgrep -f run_queue.sh >/dev/null || pgrep -f run_js_formal.sh >/dev/null; do
  sleep 120
done
SEARCH=$(python -c "import sys; sys.path.insert(0,'analysis'); from js2_search import conditions_v2; print(','.join(conditions_v2()))")
echo "== stage 1: dev search for the new path $(date -u +%FT%TZ)"
python -m prism_thesis dry-run --split dev --conditions "$SEARCH" --limit 2 --out /tmp/js2_dry.jsonl
python -m prism_thesis run --split dev --conditions "$SEARCH" --reps 3 --workers "$WORKERS" \
  --out results/js2/search.jsonl --allow-paid --budget-usd "$BUDGET"
echo "== stage 2: js2 policy family"
python analysis/js2_search.py --runs-v1 results/js/search.jsonl --runs-v2 results/js2/search.jsonl \
  --out results/js2/js_table.json | tee results/js2/search_report.txt
SEQ=$(cat results/js2/distinct_lambdas.txt)
CONDS="full$(for l in $SEQ; do printf ',js@%s' "$l"; done)"
CALIB_IDS=$(python -c "import json;print(','.join(map(str,json.load(open('configs/pilot_split.json'))['calibration_task_ids'])))")
sha256sum results/js2/js_table.json | tee results/js2/js_table.sha256
echo "== stage 3: calibration ($CONDS) $(date -u +%FT%TZ)"
python -m prism_thesis --config configs/js2.json run --split calib --task-ids "$CALIB_IDS" --conditions "$CONDS" \
  --reps 10 --workers "$WORKERS" --out results/js2/calib.jsonl --allow-paid --budget-usd "$BUDGET"
echo "== stage 4: fixed-sequence test"
python analysis/js_calibrate.py --runs results/js2/calib.jsonl --sequence "$(echo $SEQ | tr ' ' ',')" --others '' \
  --lambda-out results/js2/lambda_star.txt | tee results/js2/calib_report.txt
L=$(cat results/js2/lambda_star.txt)
if [ "$L" = "none" ]; then echo "no lambda passed; js2 falls back to full. Stopping before the test set."; exit 0; fi
echo "== stage 5: test set with js2@$L $(date -u +%FT%TZ)"
python -m prism_thesis --config configs/js2.json run --split test --conditions "full,phase_switch,js@$L" --reps 10 \
  --workers "$WORKERS" --out results/js2/test.jsonl --allow-paid --budget-usd "$BUDGET" || echo "test run reported errors"
echo "== stage 6: test report"
python analysis/js2_compare.py --new results/js2/test.jsonl --old results/js/formal_test.jsonl \
  | tee results/js2/test_report.txt
echo "== js2 done $(date -u +%FT%TZ)"
