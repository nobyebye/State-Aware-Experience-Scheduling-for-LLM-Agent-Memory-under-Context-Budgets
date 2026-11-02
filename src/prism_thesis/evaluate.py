"""判分：按 LifelongAgentBench db_bench 官方规则判断一道题是否成功。

- 修改题（INSERT/UPDATE/DELETE）：题目结束时对表计算哈希，与标准哈希相同即成功；
  与智能体给出的“最终答案”文本无关。
- 查询题（SELECT）：把最终答案解析为元组列表，与标准答案逐行逐列比较：
  数值按相对/绝对误差 1e-6 比较，字符串允许去掉两侧多出的一层引号。
本模块按官方 DirectTypeAnswerValidator 的规则重新实现，tests/test_evaluate.py 用官方行为的样例核对。
"""
from __future__ import annotations

import math
import re
from typing import Optional, Sequence


def _tuple_spans(answer: str) -> Optional[list[tuple[int, int]]]:
    """找出答案中每个顶层括号对的位置；Decimal(...) 内的括号不计。解析失败返回 None。"""
    spans, start, in_decimal = [], None, False
    for i, ch in enumerate(answer):
        if ch == '(':
            if answer[max(0, i - 7):i] == 'Decimal':
                if in_decimal:
                    return None
                in_decimal = True
                continue
            start = i
        elif ch == ')':
            if in_decimal:
                in_decimal = False
                continue
            if start is None:
                return None
            spans.append((start, i))
            start = None
    if start is not None:
        return None
    return spans


def _number(text: str) -> Optional[float]:
    """与官方实现逐行一致，包括其起始位置的写法：起点停在最后一个非数字字符上，
    因此形如 '-55' 带一层引号的数值会解析失败（官方行为，保留以保证判分口径相同）。"""
    text = re.sub(r'^Decimal\((["\']?)(.*?)(["\']?)\)$', r'\2', text)
    valid = '0123456789.-'
    lo = 0
    for i, ch in enumerate(text):
        if ch in valid:
            break
        lo = i
    hi = len(text)
    for i in range(len(text) - 1, -1, -1):
        if text[i] in valid:
            break
        hi = i + 1
    try:
        return float(text[lo:hi])
    except ValueError:
        return None


def _match_tuple(text: str, truth: Sequence) -> bool:
    elems = [e.strip() for e in re.split(r'\s*,\s*', text) if e.strip()]
    if len(elems) != len(truth):
        return False
    for got, want in zip(elems, truth):
        if isinstance(want, (int, float)):
            value = _number(got)
            if value is None or not math.isclose(value, want, rel_tol=1e-6, abs_tol=1e-6):
                return False
        elif isinstance(want, str):
            if 0 < len(got) - len(want) <= 2:
                for q in ("'", '"'):
                    if got.startswith(q) and not want.startswith(q):
                        got = got[1:]
                        break
                for q in ("'", '"'):
                    if got.endswith(q) and not want.endswith(q):
                        got = got[:-1]
                        break
            if got != want:
                return False
        else:
            return False
    return True


def match_direct(answer: str, truth: Sequence[Sequence]) -> bool:
    spans = _tuple_spans(answer)
    if spans is None:
        return False
    if not spans:
        if len(truth) == 0:
            return True
        return len(truth) == 1 and _match_tuple(answer, truth[0])
    if len(spans) != len(truth):
        return False
    return all(_match_tuple(answer[s + 1:e], t) for (s, e), t in zip(spans, truth))
