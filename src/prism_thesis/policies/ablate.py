from .base import Plan, Policy, TurnState


class AblationPolicy(Policy):
    """风险估计用的分层留一消融：只在处于 phase 的轮次省略 groups（phase='any' 时每轮都省略），
    其余与完整记忆注入相同（官方布局）。groups 可以是指令组或经验组（exp_top1 / exp_rest / insight / repair）。
    info['n_dropped_units'] 记录本轮实际少发的单元数，用于暴露统计（检索为空的经验组不算暴露）。"""

    def __init__(self, groups: tuple, phase: str):
        self.groups, self.phase = tuple(groups), phase
        self.name = f"ablate:{'+'.join(self.groups)}@{phase}"

    def plan(self, state: TurnState) -> Plan:
        if self.phase != 'any' and state.phase != self.phase:
            return self.full_plan(n_dropped_units=0)
        units = self.drop_groups(self.groups)
        return Plan(prefix=units, info={'dropped': list(self.groups),
                                        'n_dropped_units': len(self.full_units() - units)})
