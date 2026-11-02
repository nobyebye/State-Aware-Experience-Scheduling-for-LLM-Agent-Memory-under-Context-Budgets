from ..memory.store import EXP_GROUPS
from ..prompts import SELECTABLE_GROUPS
from .base import Plan, Policy, TurnState


class RandomDropPolicy(Policy):
    """随机省略：每轮随机省略 k 个可选组（指令组与非空的经验组；k 在开发集上预先确定后冻结）。
    随机数按运行播种，可复现。"""
    name = 'random_drop'

    def __init__(self, k: int):
        self.k = k

    def plan(self, state: TurnState) -> Plan:
        groups = self.static_groups()
        candidates = [g for g in SELECTABLE_GROUPS + EXP_GROUPS if groups.get(g)]
        drop = sorted(self.rng.sample(candidates, min(self.k, len(candidates))))
        return Plan(prefix=self.drop_groups(drop), info={'dropped': drop})
