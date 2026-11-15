"""环境接口。子类给出下列内容；方法（PRISM 系列与各基线）不含任何环境相关的常量。

提示：单元 id 集合（可随题目变化，如每张表一个单元）、核心单元、可选组、单元文本与三段渲染；
交互：任务特征、状态识别、动作截取与解析、会话（execute）、判分；
恢复：显式错误 → 需要补回的组；错误签名（修复经验检索用）；
人工规则：phase_switch 基线在各状态省略的组（冻结前写定，作为专家参照与搜索种子）；
记忆：成功示例与修复经验的文本格式、提炼洞见的提示；
运行：数据文件（计入指纹）、后端（数据库服务或容器运行时）、脚本化智能体（不花钱的演练）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

PHASES = ('initial', 'after_error', 'after_success')


class NullBackend:
    """不需要常驻服务的环境用的后端。"""

    def start(self, timeout: float = 0) -> None:
        pass

    def stop(self) -> None:
        pass


class Environment:
    name = 'base'
    task_types: tuple = ('query', 'modify')
    sealed_splits: frozenset = frozenset()       # 这些划分的付费运行必须先预先登记（见 pub/seal.py）
    memory_splits: tuple = ()                    # 允许进入记忆库的划分
    obs_label = 'Observation'                    # 提炼洞见时轨迹中观察的标签
    parallel = True                              # 是否允许多个工作线程

    # ---- 提示 ----
    core_ids: frozenset = frozenset()
    selectable_groups: tuple = ()

    def all_ids(self, task) -> frozenset:
        raise NotImplementedError

    def unit_text(self, uid: str, task) -> str:
        raise NotImplementedError

    def render_rules(self, unit_ids: Iterable[str], task) -> str:
        raise NotImplementedError

    def render_system(self, unit_ids: Iterable[str], task) -> str:
        raise NotImplementedError

    def render_first_user(self, unit_ids: Iterable[str], task) -> str:
        raise NotImplementedError

    # ---- 交互 ----
    def task_features(self, task):
        raise NotImplementedError

    def phase(self, turn: int, last_exec) -> str:
        if turn == 0 or last_exec is None:
            return 'initial'
        return 'after_success' if last_exec.ok else 'after_error'

    def first_action(self, response: str) -> str:
        raise NotImplementedError

    def parse(self, response: str):
        raise NotImplementedError

    def grade(self, task, session, status: str, answer: Optional[str]) -> tuple[bool, dict]:
        """回合结束后判分；返回 (是否成功, 写入记录的附加信息)。"""
        raise NotImplementedError

    # ---- 恢复与签名 ----
    def recovery_groups(self, last_exec) -> frozenset:
        raise NotImplementedError

    def error_key(self, result) -> Optional[str]:
        if result is None or result.ok:
            return None
        return str(result.error_code) if result.error_code is not None else result.kind

    def signature(self, result, action: Optional[str]) -> Optional[str]:
        raise NotImplementedError

    # ---- 人工规则 ----
    def expert_drop(self, phase: str, task_type: str) -> tuple:
        """phase_switch 在该状态省略的指令组；出错后必须为空。"""
        raise NotImplementedError

    # ---- 记忆 ----
    def retrieval_prefer(self, experience, task) -> bool:
        """检索排序时优先的条目（如同一数据库的经验）；默认不区分。"""
        return False

    def example_text(self, task, actions: Sequence[str], answer: Optional[str], max_answer_chars: int) -> str:
        raise NotImplementedError

    def repair_text(self, error_line: str, failed: str, fixed: str) -> str:
        raise NotImplementedError

    insight_system = ''
    insight_prompt: Optional[str] = None         # None 时用 memory/build.py 的通用提示

    # ---- 运行 ----
    def load_tasks(self, split=None, include_excluded: bool = False) -> list:
        raise NotImplementedError

    def data_files(self) -> list[Path]:
        raise NotImplementedError

    def fingerprint(self) -> Mapping:
        """计入实验指纹的环境设置。"""
        return {'env': self.name}

    def make_backend(self, cfg: dict, idx: int):
        return NullBackend()

    def open_session(self, backend, task, idx: int, cfg: dict):
        raise NotImplementedError

    def scripted(self, kind: str, task):
        raise NotImplementedError

    def record_extras(self, task, features) -> dict:
        return {'task_type_pred': features.task_type}
