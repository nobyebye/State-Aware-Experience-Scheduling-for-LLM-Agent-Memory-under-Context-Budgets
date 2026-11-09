#!/usr/bin/env bash
# PRISM-JS quick pilot (docs/预实验_PRISM-JS_预先登记_20261109.md). Run on the VM from the repo root after
#   source .venv/bin/activate; export PYTHONPATH=src; export OPENROUTER_API_KEY=$(cat ~/.config/prism/openrouter.key)
# Stage 1: joint-omission search on the development set (13 conditions x 110 tasks x 3 reps = 4290 runs).
# Stage 2: build the λ policy family (no API calls).
# Stage 3: pilot comparison on the pilot half of the build set (9 conditions x 69 tasks x 3 reps = 1863 runs).
# Stage 4: print the pilot comparison (no API calls).
# Safe to re-run: finished runs in the output files are skipped.
set -euo pipefail
WORKERS=${WORKERS:-8}
BUDGET=${BUDGET:-2}
mkdir -p results/js
SEARCH=$(python - <<'PY'
import sys; sys.path.insert(0, 'analysis')
from js_search import conditions
print(','.join(conditions()))
PY
)
echo "== stage 0: free dry run of the new conditions (scripted agent, no API calls)"
python -m prism_thesis dry-run --split dev --conditions "$SEARCH" --limit 3 --out /tmp/js_dry.jsonl
echo "== stage 1: search on dev ($SEARCH)"
python -m prism_thesis run --split dev --conditions "$SEARCH" --reps 3 --workers "$WORKERS" \
  --out results/js/search.jsonl --allow-paid --budget-usd "$BUDGET"
echo "== stage 2: policy family"
python analysis/js_search.py --runs results/js/search.jsonl --out results/js/js_table.json | tee results/js/search_report.txt
PILOT_IDS=$(python -c "import json;print(','.join(map(str,json.load(open('configs/pilot_split.json'))['pilot_task_ids'])))")
PILOT_CONDS="full,js-ref,prism@8,prism-bundle@8,phase_switch,js@4,js@8,js@12,js@16"
echo "== stage 3: pilot comparison on build pilot half ($PILOT_CONDS)"
python -m prism_thesis run --split calib --task-ids "$PILOT_IDS" --conditions "$PILOT_CONDS" --reps 3 \
  --workers "$WORKERS" --out results/js/pilot.jsonl --allow-paid --budget-usd "$BUDGET"
echo "== stage 4: pilot report"
python analysis/pilot_compare.py --runs results/js/pilot.jsonl | tee results/js/pilot_report.txt
