"""各实验条件的上下文策略。每个策略每轮返回一个 Plan，说明发送哪些单元、放在哪里。"""
from .base import Plan, Policy, TaskFeatures, TurnState
from .full import FullPolicy
from .phase_switch import PhaseSwitchPolicy
from .static import StaticPolicy
from .random_drop import RandomDropPolicy
from .ablate import AblationPolicy
from .prism import PrismPolicy

__all__ = ['Plan', 'Policy', 'TaskFeatures', 'TurnState', 'FullPolicy', 'PhaseSwitchPolicy',
           'StaticPolicy', 'RandomDropPolicy', 'AblationPolicy', 'PrismPolicy']
