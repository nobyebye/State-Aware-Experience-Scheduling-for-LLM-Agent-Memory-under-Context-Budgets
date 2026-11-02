from .base import Plan, Policy, TurnState


class FullPolicy(Policy):
    """完整上下文：每轮发送全部指令单元（官方布局）；使用记忆时再加上任务开始时检索的全部经验
    （ExpeL 式完整记忆注入，非劣检验的参照）。不使用记忆时即“无记忆”条件（nomem）。"""
    name = 'full'

    def plan(self, state: TurnState) -> Plan:
        return self.full_plan()
