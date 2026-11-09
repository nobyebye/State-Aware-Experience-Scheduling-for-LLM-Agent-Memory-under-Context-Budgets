"""PRISM-JS（联合删除搜索）：按 (状态, 任务类型) 查表得到整组省略方案，方案的损失是整体实测的联合损失，
不再把单组风险相加。

- 检索与 PRISM / phase_switch 相同（状态感知检索）：任务开始时检索相似示例与洞见；出错后按错误签名检索
  修复经验并单调累积。
- 出错后的轮次发送全部候选（全部指令 + 示例 + 洞见 + 修复经验），与 PRISM 的回退一致。
- 显式错误后，与错误相关的指令组和修复经验补回并在之后各轮保持（与 PRISM 相同的补回映射）。
- 核心单元总是保留；布局与完整提示相同（指令在系统消息，经验块在任务之前）。

两种用法：
  JointPolicy.search(groups, phase)   搜索阶段的联合消融：只在 phase 状态省略 groups（两类任务相同），
                                      其余与状态感知检索的参照相同；groups 为空时就是参照本身（js-ref）。
  JointPolicy.from_table(spec)        正式策略：spec[(phase, task_type)] = 省略的组。
"""
from __future__ import annotations

from ..prism import recovery
from ..prompts import ALL_IDS, CORE_IDS
from .base import Plan, Policy, TurnState

PHASES = ('initial', 'after_success')
TYPES = ('query', 'modify')


class JointPolicy(Policy):
    def __init__(self, spec: dict[tuple[str, str], tuple[str, ...]], name: str = 'js'):
        for (phase, ttype), groups in spec.items():
            if phase not in PHASES or ttype not in TYPES:
                raise ValueError(f'未知的层：{phase}|{ttype}')
            if 'repair' in groups:
                raise ValueError('repair 只在出错后成为候选，不能出现在省略方案中')
        self.spec = {k: tuple(v) for k, v in spec.items()}
        self.name = name

    @classmethod
    def search(cls, groups: tuple[str, ...], phase: str, name: str) -> 'JointPolicy':
        return cls({(phase, t): tuple(groups) for t in TYPES}, name)

    def start(self, task, features, rng) -> None:
        super().start(task, features, rng)
        self.reincluded: frozenset = frozenset()

    def plan(self, state: TurnState) -> Plan:
        repairs = self.collect_repairs(state)
        self.reincluded |= recovery.groups_for(state.last_exec)
        exp = self.exp_initial
        exp_units = frozenset(u for us in exp.values() for u in us) | repairs
        info = {'phase': state.phase, 'reincluded': sorted(self.reincluded), 'n_repairs': len(repairs)}
        if state.phase == 'after_error':
            return Plan(prefix=ALL_IDS | exp_units, info={**info, 'dropped': [], 'n_dropped_units': 0})
        drop = [g for g in self.spec.get((state.phase, self.features.task_type), ())
                if g not in self.reincluded]
        groups = {**self.features.groups, **{g: frozenset(u) for g, u in exp.items()}}
        dropped = frozenset().union(*(groups.get(g, frozenset()) for g in drop)) - CORE_IDS
        units = (ALL_IDS | exp_units) - dropped
        return Plan(prefix=units, info={**info, 'dropped': sorted(drop), 'n_dropped_units': len(dropped)})
