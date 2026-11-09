#!/usr/bin/env bash
# PRISM-JS formal experiment. The policy family results/js/js_table.json is frozen from the pilot search.
# Stage A: calibration on the calibration half of the build set (203 tasks, 10 reps):
#          full, js-ref, phase_switch, js@2, js@4, js@8, js@16 (14210 runs).
# Stage B: fixed-sequence non-inferiority test -> lambda* (no API calls).
# Stage C: test set (114 tasks, 10 reps), all conditions in one batch:
#          full, js-ref, prism@8, phase_switch, js@<lambda*> (5700 runs).
# Stage D: test report (no API calls).
# Run in tmux from the repo root after the usual environment setup. Safe to re-run: finished runs are skipped.
set -euo pipefail
WORKERS=${WORKERS:-8}
BUDGET=${BUDGET:-3}
sha256sum results/js/js_table.json | tee results/js/js_table.sha256
CALIB_IDS=$(python -c "import json;print(','.join(map(str,json.load(open('configs/pilot_split.json'))['calibration_task_ids'])))")
echo "== stage A: calibration (build calibration half)"
python -m prism_thesis run --split calib --task-ids "$CALIB_IDS" \
  --conditions "full,js-ref,phase_switch,js@2,js@4,js@8,js@16" --reps 10 --workers "$WORKERS" \
  --out results/js/formal_calib.jsonl --allow-paid --budget-usd "$BUDGET"
echo "== stage B: fixed-sequence test"
python analysis/js_calibrate.py --runs results/js/formal_calib.jsonl --lambda-out results/js/lambda_star.txt \
  | tee results/js/formal_calib_report.txt
L=$(cat results/js/lambda_star.txt)
if [ "$L" = "none" ]; then echo "no lambda passed; PRISM-JS falls back to full. Stopping before the test set."; exit 0; fi
echo "== stage C: test set with js@$L"
python -m prism_thesis run --split test --conditions "full,js-ref,prism@8,phase_switch,js@$L" --reps 10 \
  --workers "$WORKERS" --out results/js/formal_test.jsonl --allow-paid --budget-usd "$BUDGET"
echo "== stage D: test report"
python analysis/pilot_compare.py --runs results/js/formal_test.jsonl | tee results/js/formal_test_report.txt
