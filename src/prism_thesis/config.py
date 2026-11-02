"""读取实验配置（JSON），并根据条件名构造对应的上下文策略。

条件名：
  nomem                        无记忆：完整指令，不注入经验
  full                         完整记忆注入（ExpeL 式）：完整指令 + 任务开始时检索的全部经验；非劣检验的参照
                               （配置 memory.enabled 为 false 时不使用记忆，full 与 nomem 相同）
  phase_switch / random_drop / llmlingua2
  static@ε                     自建静态基线（ε 为风险预算，百分点）
  prism@ε                      PRISM
  prism-stateless@ε            消融：风险不区分状态
  prism-norecovery@ε           消融：去掉补回
  prism-tailsys@ε              消融：临时块放进系统消息
  prism-singleonly@ε           消融：只用单独留一的风险估计（不做互补合并）
  prism-nosig@ε                消融：出错后不按错误签名检索修复经验，只补回指令
  prism-greedy@ε               消融：贪心代替精确求解
  ablate:<组>[+<组>]@<状态>     风险估计用的分层留一消融（组可以是指令组或经验组）
任何条件后可加检索条数修饰 #k=N（如 full#k=5、prism@4#k=1），覆盖 memory.k_examples，用于 k 的敏感性实验。
"""
from __future__ import annotations

import json
from pathlib import Path

from .memory import MemoryStore, Retriever
from .policies import (AblationPolicy, FullPolicy, PhaseSwitchPolicy, PrismPolicy, RandomDropPolicy,
                       StaticPolicy)
from .prism.risk import RiskTable, parse_ablation
from .prompts import set_variant
from .tokens import TokenCounter

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = ROOT / 'configs' / 'default.json') -> dict:
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    set_variant(cfg.get('prompt_variant', 'standard'))
    return cfg


RETRIEVAL_KEYS = ('k_examples', 'k_insights', 'k_repairs', 'bm25')
MODIFIERS = {'k': 'k_examples'}


def split_condition(condition: str) -> tuple[str, dict[str, int]]:
    """'prism@4#k=5' -> ('prism@4', {'k_examples': 5})。"""
    base, _, mods = condition.partition('#')
    out = {}
    for m in filter(None, mods.split(',')):
        key, _, val = m.partition('=')
        if key not in MODIFIERS or not val.isdigit():
            raise ValueError(f'未知的条件修饰：{m}（只支持 #k=N）')
        out[MODIFIERS[key]] = int(val)
    if out and base.startswith('ablate:'):
        raise ValueError('消融条件不支持检索条数修饰')
    return base, out


def uses_memory(cfg: dict, condition: str) -> bool:
    base, _ = split_condition(condition)
    return bool(cfg.get('memory', {}).get('enabled')) and base != 'nomem'


def memory_settings(cfg: dict, condition: str) -> dict | None:
    """该条件实际使用的检索设置（含修饰覆盖）；不使用记忆时为 None。"""
    if not uses_memory(cfg, condition):
        return None
    _, mods = split_condition(condition)
    m = cfg['memory']
    return {**{k: m[k] for k in RETRIEVAL_KEYS if k in m}, **mods}


def memory_path(cfg: dict) -> Path:
    p = Path(cfg['memory']['store'])
    return p if p.is_absolute() else ROOT / p


class RehearsalArtifact(ValueError):
    """演练产物（脚本化运行、未经正式检查的构建或演练洞见）被用于正式实验。"""


def make_retriever(cfg: dict, settings: dict, allow_rehearsal: bool = False) -> Retriever:
    """正式（付费）运行只接受正式构建的记忆库：meta.rehearsal 必须明确为 false。"""
    path = memory_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f'记忆库 {path} 不存在：先用 nomem 条件在构建集与开发集上运行，再 build-memory')
    store = MemoryStore.load(path)
    if not allow_rehearsal and store.meta.get('rehearsal') is not False:
        raise RehearsalArtifact(f'记忆库 {path} 是演练产物或来源不明（meta.rehearsal={store.meta.get("rehearsal")!r}），'
                                '不能用于正式运行；请用正式记录重新 build-memory')
    bm = settings.get('bm25', {})
    return Retriever(store, settings.get('k_examples', 3), settings.get('k_insights', 3),
                     settings.get('k_repairs', 2), bm.get('k1', 1.2), bm.get('b', 0.75))


def table_key(condition: str) -> str | None:
    """该条件使用的风险表在 cfg['prism'] 中的键；不使用风险表时为 None。"""
    name = split_condition(condition)[0].partition('@')[0]
    if name == 'prism-singleonly':
        return 'risk_table_single_only'
    if name in ('static', 'prism', 'prism-stateless', 'prism-norecovery', 'prism-tailsys', 'prism-nosig',
                'prism-greedy'):
        return 'risk_table'
    return None


def table_path(cfg: dict, key: str = 'risk_table') -> Path:
    p = Path(cfg['prism'][key])
    return p if p.is_absolute() else ROOT / p


def _table(cfg: dict, key: str = 'risk_table') -> RiskTable:
    return RiskTable.load(table_path(cfg, key))


def make_policy(condition: str, cfg: dict, counter: TokenCounter, allow_rehearsal_memory: bool = False):
    """allow_rehearsal_memory：只在不付费的演练与查看提示时为 True。"""
    base, _ = split_condition(condition)
    policy = _make_base_policy(base, cfg, counter)
    if base == 'nomem':
        policy.name = 'nomem'
    settings = memory_settings(cfg, condition)
    if settings is not None:
        policy.memory = make_retriever(cfg, settings, allow_rehearsal_memory)
    if base != condition:
        policy.name = condition
    return policy


def _make_base_policy(condition: str, cfg: dict, counter: TokenCounter):
    step = cfg['prism']['step_pp']
    if condition in ('full', 'nomem'):
        return FullPolicy()
    if condition == 'phase_switch':
        return PhaseSwitchPolicy()
    if condition == 'random_drop':
        p = RandomDropPolicy(cfg['baselines']['random_k'])
        p.name = 'random_drop'
        return p
    if condition == 'llmlingua2':
        from .policies.llmlingua2 import LLMLingua2Policy
        c = cfg['baselines']['llmlingua2']
        p = LLMLingua2Policy(c['rate'], c['model_name'], c.get('revision'), c.get('device', 'cpu'))
        p.name = 'llmlingua2'
        return p
    ab = parse_ablation(condition)
    if ab:
        return AblationPolicy(*ab)
    name, _, eps = condition.partition('@')
    if not eps:
        raise ValueError(f'未知条件：{condition}')
    eps = float(eps)
    if name == 'static':
        p = StaticPolicy(_table(cfg), eps, counter, step)
    elif name == 'prism':
        p = PrismPolicy(_table(cfg), eps, counter, step)
    elif name == 'prism-stateless':
        p = PrismPolicy(_table(cfg), eps, counter, step, stateless=True)
    elif name == 'prism-norecovery':
        p = PrismPolicy(_table(cfg), eps, counter, step, recovery=False)
    elif name == 'prism-tailsys':
        p = PrismPolicy(_table(cfg), eps, counter, step, tail_in_system=True)
    elif name == 'prism-singleonly':
        p = PrismPolicy(_table(cfg, 'risk_table_single_only'), eps, counter, step)
    elif name == 'prism-nosig':
        p = PrismPolicy(_table(cfg), eps, counter, step, signature=False)
    elif name == 'prism-greedy':
        p = PrismPolicy(_table(cfg), eps, counter, step, greedy=True)
    else:
        raise ValueError(f'未知条件：{condition}')
    p.name = condition
    return p
