#!/usr/bin/env bash
# Queue for unattended running (start in its own tmux window while run_js_formal.sh is still running):
#   1. wait until the formal PRISM-JS experiment has finished (formal_test_report.txt exists, or no run is active
#      and the formal script has stopped);
#   2. transfer experiment with a second backbone (configs/transfer_70b.json, Llama-3.3-70B-Instruct):
#      PRISM-JS re-searched on the dev set with this model (4290 runs), then on the test set
#      full, phase_switch (rules unchanged) and js@lambda* (5 reps, 1710 runs);
#   3. always (also after a failure): export compact CSVs and reports to reports/ and push them to GitHub.
set -uo pipefail
cd "$(dirname "$0")/.."
WORKERS=${WORKERS:-8}
BUDGET=${BUDGET:-3}
LOG=results/queue.log
mkdir -p results/js70 reports
push_reports() {
  echo "== export and push reports $(date -u +%FT%TZ)"
  for f in formal_calib formal_test search pilot; do
    [ -f "results/js/$f.jsonl" ] && python analysis/export_compact.py --runs "results/js/$f.jsonl" --out "reports/js_$f.csv"
  done
  for f in search test; do
    [ -f "results/js70/$f.jsonl" ] && python analysis/export_compact.py --runs "results/js70/$f.jsonl" --out "reports/js70_$f.csv"
  done
  cp -f results/js/*report*.txt results/js/js_table.json results/js/lambda_star.txt reports/ 2>/dev/null
  mkdir -p reports/js70
  cp -f results/js70/*report*.txt results/js70/js_table.json reports/js70/ 2>/dev/null
  cp -f "$LOG" results/js/formal.log reports/ 2>/dev/null
  git add -f reports
  git commit -q -m "Add experiment reports from the VM ($(date -u +%FT%TZ))" || true
  for i in 1 2 3 4; do git pull -q --rebase && git push -q && { echo "pushed reports"; return 0; }; sleep $((2**i)); done
  echo "PUSH FAILED: copy ~/prism-thesis/reports to E: or paste the report files"
}
trap push_reports EXIT
{
echo "== waiting for the formal experiment $(date -u +%FT%TZ)"
while [ ! -f results/js/formal_test_report.txt ]; do
  if ! pgrep -f run_js_formal.sh >/dev/null && ! pgrep -f "prism_thesis run" >/dev/null; then
    echo "formal script is no longer running and produced no test report; continuing with what exists"; break
  fi
  sleep 300
done
L=$(cat results/js/lambda_star.txt 2>/dev/null || echo 8); [ "$L" = "none" ] && L=8
echo "== transfer (70B): search on dev $(date -u +%FT%TZ)"
SEARCH=$(python -c "import sys; sys.path.insert(0,'analysis'); from js_search import conditions; print(','.join(conditions()))")
python -m prism_thesis --config configs/transfer_70b.json run --split dev --conditions "$SEARCH" --reps 3 \
  --workers "$WORKERS" --out results/js70/search.jsonl --allow-paid --budget-usd "$BUDGET" || echo "search incomplete"
python analysis/js_search.py --runs results/js70/search.jsonl --out results/js70/js_table.json \
  | tee results/js70/search_report.txt || echo "search analysis failed"
if [ -f results/js70/js_table.json ]; then
  echo "== transfer (70B): test set with js@$L $(date -u +%FT%TZ)"
  python -m prism_thesis --config configs/transfer_70b.json run --split test \
    --conditions "full,phase_switch,js@$L" --reps 5 --workers "$WORKERS" --out results/js70/test.jsonl \
    --allow-paid --budget-usd "$BUDGET" || echo "transfer test incomplete"
  python analysis/pilot_compare.py --runs results/js70/test.jsonl | tee results/js70/test_report.txt
fi
echo "== queue done $(date -u +%FT%TZ)"
} 2>&1 | tee -a "$LOG"
