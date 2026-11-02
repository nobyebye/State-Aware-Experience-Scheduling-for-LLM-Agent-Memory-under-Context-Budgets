"""每轮的 0-1 背包：在风险预算内省略 token 最多的单元组。

价值 = 组的 token 数；重量 = 风险（百分点）按 step 向上取整后的格数；容量 = floor(ε / step)。
负的风险估计记为 0。动态规划给出离散化代理问题的精确最优解（不代表对真实准确率最优）。
同等价值时偏向省略更少的组，再按组名排序，保证结果确定。
"""
from __future__ import annotations

import math
from typing import Mapping


def weight(risk_pp: float, step: float = 0.5) -> int:
    if risk_pp <= 0:
        return 0
    return math.ceil(risk_pp / step - 1e-9)


def solve(values: Mapping[str, int], risks_pp: Mapping[str, float], epsilon_pp: float,
          step: float = 0.5) -> frozenset[str]:
    """返回应省略的组集合。values 与 risks_pp 的键为候选组。"""
    capacity = max(0, math.floor(epsilon_pp / step + 1e-9))
    items = sorted(values)
    # best[c] = (总价值, -组数, 选择) ，c 为已用容量上限
    best: list[tuple[int, int, tuple[str, ...]]] = [(0, 0, ())] * (capacity + 1)
    for g in items:
        w, v = weight(risks_pp[g], step), values[g]
        if w > capacity:
            continue
        new = list(best)
        for c in range(w, capacity + 1):
            val, neg_n, chosen = best[c - w]
            cand = (val + v, neg_n - 1, chosen + (g,))
            if cand[:2] > new[c][:2]:
                new[c] = cand
        best = new
    return frozenset(max(best, key=lambda t: t[:2])[2])


def greedy(values: Mapping[str, int], risks_pp: Mapping[str, float], epsilon_pp: float,
           step: float = 0.5) -> frozenset[str]:
    """消融用的贪心近似：按 价值 / 重量 从大到小依次省略能放下的组（重量为 0 的组最先，按价值从大到小）。"""
    capacity = max(0, math.floor(epsilon_pp / step + 1e-9))
    w = {g: weight(risks_pp[g], step) for g in values}
    order = sorted(values, key=lambda g: (w[g] > 0, -(values[g] / w[g]) if w[g] else -values[g], g))
    chosen, used = [], 0
    for g in order:
        if values[g] > 0 and used + w[g] <= capacity:
            chosen.append(g)
            used += w[g]
    return frozenset(chosen)
