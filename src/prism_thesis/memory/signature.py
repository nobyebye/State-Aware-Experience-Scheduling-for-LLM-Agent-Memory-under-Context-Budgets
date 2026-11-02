"""错误签名：错误号（或错误类别）+ 出错 SQL 的结构特征。规则只依据开发集设计。

结构特征：语句类型（SELECT / INSERT / UPDATE / DELETE / OTHER）与是否含 WHERE、GROUP BY、HAVING、
ORDER BY、LIMIT、JOIN、子查询、聚合函数。检索按“完整签名 → 错误号 + 语句类型 → 错误号”逐级放宽。
"""
from __future__ import annotations

import re
from typing import Optional

from ..env import ExecResult

_STRING = re.compile(r"'(?:[^'\\]|\\.|'')*'|\"(?:[^\"\\]|\\.|\"\")*\"")
_FEATURES = (
    ('W', re.compile(r'\bwhere\b')),
    ('G', re.compile(r'\bgroup\s+by\b')),
    ('H', re.compile(r'\bhaving\b')),
    ('O', re.compile(r'\border\s+by\b')),
    ('L', re.compile(r'\blimit\b')),
    ('J', re.compile(r'\bjoin\b')),
    ('S', re.compile(r'\(\s*select\b')),
    ('A', re.compile(r'\b(?:count|sum|avg|min|max|stddev|group_concat)\s*\(')),
)
STATEMENTS = ('select', 'insert', 'update', 'delete')


def statement_type(sql: str) -> str:
    m = re.match(r'\s*(\w+)', sql.lower())
    word = m.group(1) if m else ''
    if word == 'with':
        word = 'select'
    return word.upper() if word in STATEMENTS else 'OTHER'


def structure(sql: str) -> str:
    """'SELECT:WGA' 这样的结构串（字符串字面量先去掉，避免其中的关键词影响特征）。"""
    s = _STRING.sub("''", sql or '').lower()
    flags = ''.join(k for k, rx in _FEATURES if rx.search(s))
    return f'{statement_type(sql or "")}:{flags}'


def error_key(result: Optional[ExecResult]) -> Optional[str]:
    if result is None or result.ok:
        return None
    if result.error_code is not None:
        return str(result.error_code)
    return result.kind


def signature(result: Optional[ExecResult], sql: Optional[str]) -> Optional[str]:
    key = error_key(result)
    return None if key is None else f'{key}|{structure(sql or "")}'


def backoff_levels(sig: str) -> tuple[str, str, str]:
    """(完整签名, 错误号|语句类型, 错误号)。"""
    key, struct = sig.split('|', 1)
    return sig, f'{key}|{struct.split(":", 1)[0]}', key
