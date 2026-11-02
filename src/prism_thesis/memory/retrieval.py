"""经验检索（词法与规则匹配，不调用模型）。

- 任务开始时（static）：按任务指令的 BM25 相似度检索 k 条成功示例（第 1 条为 exp_top1，其余为 exp_rest）、
  适用于本任务类型的 k_insights 条洞见；完整记忆注入（ExpeL 式）另按任务相似度检索 k_repairs 条修复经验。
- 出错后（for_error）：按错误签名检索修复经验，逐级放宽：完整签名 → 错误号 + 语句类型 → 错误号；
  同一级内按任务相似度排序。
- 同组排除：来源题目组（或来源题目）与当前题目相同的条目一律不返回，防止信息泄漏。
排序并列时按条目 id，结果确定。
"""
from __future__ import annotations

from typing import Iterable, Optional

from ..data import Task
from ..env import ExecResult
from . import signature as sig
from .bm25 import BM25
from .store import Experience, MemoryStore, make_uid


class Retriever:
    def __init__(self, store: MemoryStore, k_examples: int = 3, k_insights: int = 3, k_repairs: int = 2,
                 k1: float = 1.2, b: float = 0.75):
        self.store = store
        self.k_examples, self.k_insights, self.k_repairs = k_examples, k_insights, k_repairs
        self._docs = {kind: store.of_kind(kind) for kind in ('example', 'repair', 'insight')}
        self._index = {kind: BM25([e.key_text for e in docs], k1, b) for kind, docs in self._docs.items()}

    @staticmethod
    def allowed(e: Experience, task: Task) -> bool:
        return task.group_id not in e.source_groups and task.task_id not in e.source_tasks

    def _ranked(self, kind: str, task: Task, keep=lambda e: True) -> list[Experience]:
        docs = self._docs[kind]
        scores = self._index[kind].scores(task.instruction) if docs else []
        cand = [(-s, e.eid, e) for s, e in zip(scores, docs) if self.allowed(e, task) and keep(e)]
        return [e for _, _, e in sorted(cand, key=lambda t: t[:2])]

    def examples(self, task: Task) -> list[Experience]:
        return self._ranked('example', task)[:self.k_examples]

    def insights(self, task: Task, task_type: str) -> list[Experience]:
        return self._ranked('insight', task, lambda e: e.task_type in (task_type, 'any'))[:self.k_insights]

    def similar_repairs(self, task: Task) -> list[Experience]:
        return self._ranked('repair', task)[:self.k_repairs]

    def static(self, task: Task, task_type: str, with_repairs: bool = True) -> dict[str, tuple[str, ...]]:
        """任务开始时一次检索的经验，按选择组给出单元 id（组内按检索排名）。"""
        ex = self.examples(task)
        groups = {
            'exp_top1': tuple(make_uid('example', i, e.eid) for i, e in enumerate(ex[:1])),
            'exp_rest': tuple(make_uid('example', i + 1, e.eid) for i, e in enumerate(ex[1:])),
            'insight': tuple(make_uid('insight', i, e.eid) for i, e in enumerate(self.insights(task, task_type))),
        }
        if with_repairs:
            groups['repair'] = tuple(make_uid('repair', i, e.eid) for i, e in enumerate(self.similar_repairs(task)))
        return groups

    def for_error(self, task: Task, result: Optional[ExecResult], sql: Optional[str],
                  exclude: Iterable[str] = ()) -> list[Experience]:
        """按错误签名检索修复经验（排除 exclude 中已取到的条目 id）；没有显式错误时为空。"""
        s = sig.signature(result, sql)
        if s is None:
            return []
        skip = set(exclude)
        full, stmt, key = sig.backoff_levels(s)
        tests = (lambda e: e.signature == full,
                 lambda e: e.signature is not None and sig.backoff_levels(e.signature)[1] == stmt,
                 lambda e: e.error_key == key)
        for test in tests:
            hits = self._ranked('repair', task, lambda e: test(e) and e.eid not in skip)
            if hits:
                return hits[:self.k_repairs]
        return []
