"""不调用模型的全流程演练：无记忆运行 → 构建记忆库（含演练洞见）→ 消融 → 风险表 → PRISM 与各对比条件
→ 校准检验 → 比较 → 记忆使用统计。

用脚本化智能体（gold_after_error：先出一次错再执行参考 SQL）运行，检验环境、判分、记忆库构建与检索、
各策略、日志、风险估计与统计脚本能否连通。脚本化智能体不读提示，所以各条件成功率都应为 100%，
风险估计都应接近 0；这些数字不是实验结果。
用法：python scripts/dry_run_pipeline.py [--limit 30]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from prism_thesis.cli import main  # noqa: E402
from prism_thesis.memory.store import EXP_GROUPS  # noqa: E402
from prism_thesis.prompts import SELECTABLE_GROUPS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--limit', type=int, default=30)
ap.add_argument('--workers', type=int, default=2)
args = ap.parse_args()

out = ROOT / 'results' / 'dryrun'
out.mkdir(parents=True, exist_ok=True)
cfg = json.loads((ROOT / 'configs' / 'default.json').read_text(encoding='utf-8'))
cfg['prism']['risk_table'] = str(out / 'risk_table.json')
cfg['prism']['risk_table_single_only'] = str(out / 'risk_table_single_only.json')
cfg['memory']['store'] = str(out / 'memory_store.json')
cfg_path = out / 'config_dryrun.json'
cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding='utf-8')
common = ['--config', str(cfg_path)]
lim = ['--split', 'dev', '--limit', str(args.limit), '--workers', str(args.workers), '--agent', 'gold_after_error']
mem = ['--split', 'memory', '--limit', str(2 * args.limit), '--workers', str(args.workers), '--agent', 'gold_after_error']

phases = ['initial', 'after_error', 'after_success', 'any']
ablations = [f'ablate:{g}@{p}' for g in SELECTABLE_GROUPS + EXP_GROUPS for p in phases]
ablations += [f"ablate:{'+'.join(pair)}@{p}" for pair in cfg['complementary_pairs'] for p in phases]
steps = [
    ['check-data'],
    ['dry-run', *mem, '--conditions', 'nomem', '--out', str(out / 'memory_runs.jsonl')],
    ['distill-insights', '--runs', str(out / 'memory_runs.jsonl'), '--out', str(out / 'insights.json'), '--rehearsal'],
    ['build-memory', '--runs', str(out / 'memory_runs.jsonl'), '--insights', str(out / 'insights.json'),
     '--rehearsal'],
    ['dry-run', *lim, '--conditions', ','.join(['full'] + ablations), '--out', str(out / 'ablation.jsonl')],
    ['risk-estimate', '--runs', str(out / 'ablation.jsonl'), '--out', str(out / 'risk_table.json'), '--rehearsal'],
    ['risk-estimate', '--runs', str(out / 'ablation.jsonl'), '--out', str(out / 'risk_table_single_only.json'),
     '--single-only', '--rehearsal'],
    ['dry-run', *lim, '--conditions', ','.join(
        ['full', 'nomem', 'phase_switch', 'random_drop', 'static@4'] + [f'prism@{e}' for e in cfg['prism']['epsilons_pp']]
        + ['prism-stateless@4', 'prism-norecovery@4', 'prism-tailsys@4', 'prism-singleonly@4', 'prism-nosig@4',
           'prism-greedy@4', 'prism-tail@4', 'full#k=1', 'prism@4#k=1', 'full#k=5', 'prism@4#k=5']),
     '--out', str(out / 'methods.jsonl')],
    ['calibrate', '--runs', str(out / 'methods.jsonl'), '--rehearsal'],
    ['compare', '--runs', str(out / 'methods.jsonl'), '--split', 'dev', '--rehearsal', '--json', str(out / 'compare.json')],
    ['token-audit', '--runs', str(out / 'methods.jsonl')],
    ['memory-stats', '--runs', str(out / 'methods.jsonl')],
]
for s in steps:
    print(f"\n=== {' '.join(s[:1])} ===", flush=True)
    code = main(common + s)
    if code:
        sys.exit(f'步骤失败：{s[0]}')
