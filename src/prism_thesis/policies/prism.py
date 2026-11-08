from __future__ import annotations

from typing import Optional

from ..memory.store import exp_text, is_exp
from ..prism import knapsack, recovery
from ..prism.risk import ANY, RiskTable
from ..prompts import ALL_IDS, CORE_IDS, unit_text
from ..tokens import TokenCounter
from .base import Plan, Policy, TurnState


def merged_groups(groups: dict, table: RiskTable) -> dict[str, frozenset]:
    """选择组（指令组与经验组），按风险表中的互补合并关系合并。空组去掉。
    groups 可以是 TaskFeatures（取其 groups）或 组 → 单元 的映射。"""
    groups = getattr(groups, 'groups', groups)
    mm = table.merge_map()
    out: dict[str, set] = {}
    for g, units in groups.items():
        if units:
            out.setdefault(mm.get(g, g), set()).update(units)
    return {g: frozenset(u) for g, u in out.items()}


def group_tokens(groups: dict[str, frozenset], task, counter: TokenCounter, store=None) -> dict[str, int]:
    def text(u):
        return exp_text(u, store) if is_exp(u) else unit_text(u, task)
    return {g: sum(counter.count(text(u)) + 1 for u in units) for g, units in groups.items()}


def select(groups, values, table: RiskTable, phase: str, task_type: str, epsilon: float, step: float,
           forced: frozenset = frozenset(), solver=knapsack.solve) -> Optional[frozenset]:
    """返回应省略的组；风险表缺少任一候选组时返回 None（该轮发送全部单元）。"""
    candidates = [g for g in groups if g not in forced]
    risks = {}
    for g in candidates:
        r = table.lookup(g, phase, task_type)
        if r is None:
            return None
        risks[g] = r
    return solver({g: values[g] for g in candidates}, risks, epsilon, step)


def select_bundles(groups, values, table: RiskTable, phase: str, task_type: str, epsilon: float, step: float,
                   forced: frozenset = frozenset(), solver=knapsack.solve) -> Optional[tuple[frozenset, tuple]]:
    """补充实验 PRISM-bundle：在 select 的基础上加入风险表中的组合组作为候选。
    单组风险缺项时与 select 相同，返回 None（该轮发送全部单元）。组合只在其状态下、风险表有该任务类型的
    条目时可选；成员组按互补合并关系换名，只计本轮实际存在的组；任一成员被补回强制保留时不可选。
    返回 (省略的组, 选中的组合名)。"""
    candidates = [g for g in groups if g not in forced]
    risks = {}
    for g in candidates:
        r = table.lookup(g, phase, task_type)
        if r is None:
            return None
        risks[g] = r
    mm = table.merge_map()
    options = {}
    for name, spec in table.bundles().items():
        if spec['phase'] != phase:
            continue
        r = table.lookup(name, phase, task_type)
        if r is None:
            continue
        members = {mm.get(g, g) for g in spec['groups']}
        if members & set(forced):
            continue
        members &= set(candidates)
        if members:
            options[name] = (frozenset(members), r)
    return knapsack.solve_with_bundles({g: values[g] for g in candidates}, risks, epsilon, options, step, solver)


class PrismPolicy(Policy):
    """PRISM：每轮的候选单元 = 指令单元 ∪ 按状态检索的经验（任务开始时的相似示例与洞见；出错后按错误签名
    检索的修复经验，单调累积）。每轮按状态与任务类型查风险表，在风险预算 ε 内用 0-1 背包选出省略 token 最多
    的组；显式错误后把相关指令组与修复经验补回并在之后各轮保持（M_t = M_{t−1} ∪ Map(e_{t−1})）；核心单元与
    任务指令放在稳定前缀，本轮选中的其余单元（含经验）作为末尾临时块，不写入历史。
    layout='official'（正式条件使用）：选中的单元放在与完整提示相同的位置（指令在系统消息，经验块在任务之前），
    与风险估计时消融条件的布局一致，只是省略被删除的组；layout='tail' 为上述稳定前缀 + 临时块布局（消融对照）。
    消融开关：stateless（风险不分状态）、recovery（补回指令组与修复经验）、signature（错误签名检索修复经验）、
    tail_in_system（临时块放进系统消息）、greedy（贪心代替精确求解）。"""

    def __init__(self, table: RiskTable, epsilon: float, counter: TokenCounter, step: float = 0.5,
                 stateless: bool = False, recovery: bool = True, tail_in_system: bool = False,
                 signature: bool = True, greedy: bool = False, name: Optional[str] = None,
                 layout: str = 'tail', bundles: bool = False):
        if bundles and (layout != 'official' or stateless):
            raise ValueError('组合候选只用于正式布局、区分状态的 PRISM')
        self.bundles = bundles
        if layout not in ('tail', 'official'):
            raise ValueError(f'未知布局：{layout}')
        if layout == 'official' and tail_in_system:
            raise ValueError('tail_in_system 只用于 tail 布局')
        self.layout = layout
        self.table, self.epsilon, self.counter, self.step = table, epsilon, counter, step
        self.stateless, self.recovery, self.tail_in_system = stateless, recovery, tail_in_system
        self.signature, self.greedy = signature, greedy
        self.name = name or f'prism@{epsilon:g}'

    def start(self, task, features, rng) -> None:
        super().start(task, features, rng)
        self.reincluded: frozenset = frozenset()

    def plan(self, state: TurnState) -> Plan:
        mm = self.table.merge_map()
        repairs = self.collect_repairs(state) if (self.recovery and self.signature) else frozenset()
        raw = {**self.features.groups, **{g: frozenset(u) for g, u in self.exp_initial.items()}}
        if repairs:
            raw['repair'] = repairs
        groups = merged_groups(raw, self.table)
        if self.recovery:
            self.reincluded |= frozenset(mm.get(g, g) for g in recovery.groups_for(state.last_exec))
            if repairs:
                self.reincluded |= {mm.get('repair', 'repair')}
        values = group_tokens(groups, self.task, self.counter, self.store)
        phase = ANY if self.stateless else state.phase
        solver = knapsack.greedy if self.greedy else knapsack.solve
        info = {'phase': phase, 'reincluded': sorted(self.reincluded), 'n_repairs': len(repairs)}
        if self.bundles:
            picked = select_bundles(groups, values, self.table, phase, self.features.task_type,
                                    self.epsilon, self.step, self.reincluded, solver)
            drop = None if picked is None else picked[0]
            if picked is not None:
                info['bundles'] = list(picked[1])
        else:
            drop = select(groups, values, self.table, phase, self.features.task_type,
                          self.epsilon, self.step, self.reincluded, solver)
        exp_units = frozenset(u for g, us in groups.items() for u in us if is_exp(u))
        if self.layout == 'official':
            if drop is None:              # 风险表缺项：全部发送
                return Plan(prefix=ALL_IDS | exp_units, info={**info, 'fallback': 'unseen_state'})
            selected = frozenset().union(*(u for g, u in groups.items() if g not in drop))
            return Plan(prefix=CORE_IDS | selected, info={**info, 'dropped': sorted(drop)})
        if drop is None:                  # 风险表缺项：全部发送，布局不变（稳定前缀只含核心单元）
            return Plan(prefix=CORE_IDS, tail=(ALL_IDS - CORE_IDS) | exp_units, tail_in_system=self.tail_in_system,
                        info={**info, 'fallback': 'unseen_state'})
        selected = frozenset().union(*(u for g, u in groups.items() if g not in drop))
        return Plan(prefix=CORE_IDS, tail=selected - CORE_IDS, tail_in_system=self.tail_in_system,
                    info={**info, 'dropped': sorted(drop)})
