from __future__ import annotations

from ..prism.risk import ANY, RiskTable
from ..tokens import TokenCounter
from .base import Plan, Policy, TurnState
from .prism import group_tokens, merged_groups, select


class StaticPolicy(Policy):
    """自建静态基线（借鉴 ProCut 离线删减的思路，并非 ProCut 本身）：用同一风险表中不区分状态的
    估计，在任务开始时为本题确定一个固定的单元子集（指令与任务开始时检索的经验），所有轮次相同，
    官方布局，无补回。"""

    def __init__(self, table: RiskTable, epsilon: float, counter: TokenCounter, step: float = 0.5):
        self.table, self.epsilon, self.counter, self.step = table, epsilon, counter, step
        self.name = f'static@{epsilon:g}'

    def start(self, task, features, rng) -> None:
        super().start(task, features, rng)
        groups = merged_groups(self.static_groups(), self.table)
        values = group_tokens(groups, task, self.counter, self.store)
        drop = select(groups, values, self.table, ANY, features.task_type, self.epsilon, self.step)
        if drop is None:
            drop = frozenset()
        self.fixed = Plan(prefix=self.drop_groups([g2 for g in drop for g2 in g.split('+')]),
                          info={'dropped': sorted(drop)})

    def plan(self, state: TurnState) -> Plan:
        return self.fixed
