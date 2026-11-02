from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..data import Task
from ..env import ExecResult
from ..memory.store import make_uid
from ..prompts import ALL_IDS, CORE_IDS


@dataclass(frozen=True)
class TaskFeatures:
    task_type: str                            # 由任务文本规则判断：query / modify
    predicted_functions: frozenset[str]
    groups: dict                              # 指令选择组 → 单元 id 集合（本题）


@dataclass(frozen=True)
class TurnState:
    turn: int                                 # 从 0 开始
    phase: str                                # initial / after_error / after_success
    last_exec: Optional[ExecResult]
    last_sql: Optional[str] = None            # 上一轮执行的 SQL（错误签名检索用）


@dataclass(frozen=True)
class Plan:
    """prefix：放在稳定位置的单元（系统消息与第一条用户消息；经验单元在第一条用户消息的任务指令之前）；
    tail：本轮的临时块，附在最后一条消息之后，不写入历史；
    tail_in_system：消融用，把临时块放进系统消息；
    system_text / schema_text / memory_text：LLMLingua-2 基线用压缩后的文本替换原文。"""
    prefix: frozenset
    tail: frozenset = frozenset()
    tail_in_system: bool = False
    system_text: Optional[str] = None
    schema_text: Optional[str] = None
    memory_text: Optional[str] = None
    info: dict = field(default_factory=dict, compare=False)

    @property
    def units(self) -> frozenset:
        return self.prefix | self.tail


FULL_PLAN = Plan(prefix=ALL_IDS)


class Policy:
    """策略基类。memory 为 Retriever 时使用经验记忆：
    exp_static 为任务开始时一次检索的经验（ExpeL 式，含按任务相似度检索的修复经验），
    完整记忆注入、消融、静态选择、随机省略与 LLMLingua-2 都以它为可选经验；
    状态感知检索（人工阶段切换、PRISM）用 exp_initial（示例与洞见）加出错后按错误签名检索到的修复经验。"""
    name = 'base'
    memory = None

    def start(self, task: Task, features: TaskFeatures, rng: random.Random) -> None:
        self.task, self.features, self.rng = task, features, rng
        if self.memory is not None:
            self.exp_static = self.memory.static(task, features.task_type)
        else:
            self.exp_static = {}
        self.exp_initial = {g: u for g, u in self.exp_static.items() if g != 'repair'}
        self.sig_repairs: list[str] = []           # 按错误签名检索到的修复经验（单调累积）

    @property
    def store(self):
        return self.memory.store if self.memory is not None else None

    def static_groups(self) -> dict[str, frozenset]:
        """指令选择组 + 任务开始时检索的经验组。"""
        return {**self.features.groups, **{g: frozenset(u) for g, u in self.exp_static.items()}}

    def full_units(self) -> frozenset:
        return ALL_IDS | frozenset(u for us in self.exp_static.values() for u in us)

    def full_plan(self, **info) -> Plan:
        return Plan(prefix=self.full_units(), info=info)

    def drop_groups(self, groups) -> frozenset:
        """完整单元集合（含任务开始时检索的经验）去掉若干组后的单元集合（核心单元总是保留）。"""
        dropped = set()
        all_groups = self.static_groups()
        for g in groups:
            dropped |= all_groups.get(g, frozenset())
        return frozenset(self.full_units() - (dropped - CORE_IDS))

    def collect_repairs(self, state: TurnState) -> frozenset:
        """出错后按错误签名检索修复经验并累积；返回到本轮为止的修复经验单元。"""
        if self.memory is not None and state.last_exec is not None and not state.last_exec.ok:
            have = {u.split(':', 2)[2] for u in self.sig_repairs}
            for e in self.memory.for_error(self.task, state.last_exec, state.last_sql, have):
                self.sig_repairs.append(make_uid('repair', len(self.sig_repairs), e.eid))
        return frozenset(self.sig_repairs)

    def plan(self, state: TurnState) -> Plan:
        raise NotImplementedError
