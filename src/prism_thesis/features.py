"""任务特征与交互状态识别（规则，不调用模型）。

- 任务类型：由任务指令开头的动词判断是查询（query）还是修改表（modify）；
- 可能用到的函数：关键词词表预测；
- 交互状态：只描述“发生了什么”——第一轮为 initial，上一轮 SQL 出错为 after_error，
  上一轮 SQL 成功执行为 after_success。不推断“下一步该做什么”。
规则与词表只依据开发集设计；冻结后不得根据构建集或测试集修改。
"""
from __future__ import annotations

import re
from typing import Optional

from .env import ExecResult

PHASES = ('initial', 'after_error', 'after_success')
TASK_TYPES = ('query', 'modify')

_MODIFY = re.compile(r'^\s*(?:please\s+)?(insert|add|update|increase|decrease|raise|reduce|set|change|'
                     r'modify|delete|remove|mark|replace|adjust|correct)\b', re.I)

FUNCTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    'AVG': ('average', 'mean'),
    'SUM': ('total', 'sum', 'combined', 'overall amount'),
    'COUNT': ('number of', 'count', 'how many', 'at least', 'more than', 'fewer than'),
    'MAX': ('maximum', 'highest', 'largest', 'longest', 'most', 'latest', 'greatest', 'range', 'difference between'),
    'MIN': ('minimum', 'lowest', 'smallest', 'shortest', 'least', 'earliest', 'range', 'difference between'),
    'STDDEV': ('standard deviation',),
    'GROUP_CONCAT': ('concatenate', 'comma-separated', 'list of all'),
    'ROUND': ('round', 'decimal place'),
    'ABS': ('absolute',),
    'CHAR_LENGTH': ('length of', 'characters'),
    'UPPER': ('upper case', 'uppercase'),
    'LOWER': ('lower case', 'lowercase'),
    'YEAR': ('year of',),
    'MONTH': ('month of',),
    'DATEDIFF': ('days between',),
}


def task_type(instruction: str) -> str:
    return 'modify' if _MODIFY.search(instruction) else 'query'


def predict_functions(instruction: str) -> frozenset[str]:
    text = instruction.lower()
    return frozenset(f for f, kws in FUNCTION_KEYWORDS.items() if any(k in text for k in kws))


def phase(turn: int, last_exec: Optional[ExecResult]) -> str:
    if turn == 0 or last_exec is None:
        return 'initial'
    return 'after_success' if last_exec.ok else 'after_error'
