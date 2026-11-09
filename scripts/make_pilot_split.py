"""Split the build (construction) set by task group into a pilot half and a calibration half for PRISM-JS.

The pilot subset is used only for the quick comparison of all conditions; the formal calibration of PRISM-JS
uses only the remaining groups. Deterministic (seed 20261109). Writes configs/pilot_split.json.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N_PILOT_GROUPS = 60

tasks = [t for t in json.loads((ROOT / 'data' / 'splits.json').read_text(encoding='utf-8'))['tasks']
         if t['split'] == 'construction' and not t['excluded']]
groups = sorted({t['group_id'] for t in tasks})
pilot = set(random.Random(20261109).sample(groups, N_PILOT_GROUPS))
out = {
    'seed': 20261109,
    'note': 'build set split by task group: pilot (quick comparison) vs calibration (formal PRISM-JS calibration)',
    'pilot_task_ids': sorted(t['task_id'] for t in tasks if t['group_id'] in pilot),
    'calibration_task_ids': sorted(t['task_id'] for t in tasks if t['group_id'] not in pilot),
}
(ROOT / 'configs' / 'pilot_split.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print(f"pilot {len(out['pilot_task_ids'])} tasks / {N_PILOT_GROUPS} groups; "
      f"calibration {len(out['calibration_task_ids'])} tasks / {len(groups) - N_PILOT_GROUPS} groups")
