"""显式错误触发的补回：环境错误 → 相关单元组。补回集合单调增长 M_t = M_{t-1} ∪ Map(e_{t-1})。

只对有显式信号的错误起作用；SQL 能执行但答案错误的情况不会触发。
映射依据 MySQL 错误号的含义设定，只依据开发集运行记录调整；未列出的错误补回全部可选指令组。
使用经验记忆时，PRISM 另按错误签名检索修复经验（memory/retrieval.py），并同样单调保留。
"""
from __future__ import annotations

from typing import Optional

from ..env import ExecResult
from ..prompts import SELECTABLE_GROUPS

ALL = frozenset(SELECTABLE_GROUPS)

ERROR_MAP: dict[int, frozenset[str]] = {
    1064: frozenset({'one_stmt', 'fn_predicted', 'fn_other'}),            # 语法错误
    1054: frozenset({'schema'}),                                         # 未知列
    1146: frozenset({'schema', 'scope'}),                                # 表不存在
    1049: frozenset({'schema', 'scope'}),                                # 库不存在
    1305: frozenset({'fn_predicted', 'fn_other'}),                       # 函数不存在
    1582: frozenset({'fn_predicted', 'fn_other'}),                       # 函数参数个数错误
    1583: frozenset({'fn_predicted', 'fn_other'}),
    1111: frozenset({'fn_predicted', 'fn_other'}),                       # 聚合函数用法错误
    1140: frozenset({'fn_predicted', 'fn_other', 'schema'}),             # 聚合与非聚合列混用
    1055: frozenset({'fn_predicted', 'fn_other', 'schema'}),             # ONLY_FULL_GROUP_BY
    1142: frozenset({'scope'}), 1044: frozenset({'scope'}), 1227: frozenset({'scope'}),
    1045: frozenset({'scope'}), 1370: frozenset({'scope'}),               # 权限不足
    1136: frozenset({'schema'}), 1048: frozenset({'schema'}), 1366: frozenset({'schema'}),
    1292: frozenset({'schema'}), 1052: frozenset({'schema'}),             # 列数、值类型、歧义列
    1242: frozenset({'empty_scalar'}),                                    # 子查询返回多行
}
KIND_MAP: dict[str, frozenset[str]] = {
    'multi_statement': frozenset({'one_stmt'}),
    'empty': frozenset({'one_stmt'}),
}


def groups_for(result: Optional[ExecResult]) -> frozenset[str]:
    if result is None or result.ok:
        return frozenset()
    if result.kind in KIND_MAP:
        return KIND_MAP[result.kind]
    if result.error_code in ERROR_MAP:
        return ERROR_MAP[result.error_code]
    return ALL
