"""统计分析：按题目组的整群 bootstrap、配对非劣检验、固定顺序检验。

估计对象默认按题目平均（每道题权重相同）：先对每道题的重复运行取平均成功率，
再对两个条件的逐题差取平均。重抽样单位是题目组（同一组的题目共享表与语境，不独立）。
按组平均只作为敏感性分析。所有区间为双侧 95% 百分位区间，与单侧 α = 0.025 一致。
bootstrap 区间是近似的，不构成严格的有限样本保证。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

B_DEFAULT = 10_000


def per_task_mean(records: Iterable[Mapping], condition: str, key: str = 'success') -> dict[int, float]:
    acc: dict[int, list[float]] = defaultdict(list)
    for r in records:
        if r['condition'] == condition and r.get('status') != 'infrastructure_error':
            acc[r['task_id']].append(float(r[key]))
    return {t: float(np.mean(v)) for t, v in acc.items()}


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    n_tasks: int
    n_groups: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def cluster_bootstrap(values: Mapping[int, float], group_of: Mapping[int, str], b: int = B_DEFAULT,
                      seed: int = 20261022, weighting: str = 'task', level: float = 0.95) -> Interval:
    """values: 题目 → 数值（如配对差）。按组重抽样，返回均值的估计与百分位区间。"""
    groups: dict[str, list[float]] = defaultdict(list)
    for t, v in values.items():
        groups[group_of[t]].append(v)
    keys = sorted(groups)
    sums = np.array([sum(groups[k]) for k in keys])
    ns = np.array([len(groups[k]) for k in keys], dtype=float)
    means = sums / ns
    g = len(keys)
    if g == 0:
        raise ValueError('no data')
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(g, np.full(g, 1.0 / g), size=b).astype(float)
    if weighting == 'task':
        point = sums.sum() / ns.sum()
        boot = (counts @ sums) / (counts @ ns)
    elif weighting == 'group':
        point = means.mean()
        boot = (counts @ means) / g
    else:
        raise ValueError(weighting)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return Interval(float(point), float(lo), float(hi), int(ns.sum()), g)


def paired_difference(records: Sequence[Mapping], treatment: str, control: str,
                      group_of: Mapping[int, str], key: str = 'success', **kw) -> Interval:
    """treatment − control 的逐题配对差（只用两个条件都有结果的题目）。"""
    a, c = per_task_mean(records, treatment, key), per_task_mean(records, control, key)
    common = sorted(set(a) & set(c))
    return cluster_bootstrap({t: a[t] - c[t] for t in common}, group_of, **kw)


def noninferior(diff: Interval, delta: float) -> bool:
    """成功率差（方法 − 完整提示）的 95% 区间下界高于 −δ 即判为非劣。"""
    return diff.low > -delta


@dataclass(frozen=True)
class SequenceStep:
    epsilon: float
    condition: str
    diff: Interval
    passed: bool


def fixed_sequence(records: Sequence[Mapping], ordered: Sequence[tuple[float, str]], control: str,
                   group_of: Mapping[int, str], delta: float, **kw) -> tuple[Optional[float], list[SequenceStep]]:
    """按预先确定的顺序依次检验候选 ε；第一次未通过即停止。返回 (ε*, 各步结果)。
    ε* 为 None 表示无候选通过，应回退完整提示。"""
    steps, best = [], None
    for eps, cond in ordered:
        d = paired_difference(records, cond, control, group_of, **kw)
        ok = noninferior(d, delta)
        steps.append(SequenceStep(eps, cond, d, ok))
        if not ok:
            break
        best = eps
    return best, steps


def relative_saving(records: Sequence[Mapping], treatment: str, control: str,
                    group_of: Mapping[int, str], key: str = 'input_tokens', **kw) -> Interval:
    """token 相对节省 = 1 − 方法平均 / 完整提示平均（按题目平均，组 bootstrap）。"""
    a, c = per_task_mean(records, treatment, key), per_task_mean(records, control, key)
    common = sorted(set(a) & set(c))
    groups: dict[str, list[int]] = defaultdict(list)
    for t in common:
        groups[group_of[t]].append(t)
    keys = sorted(groups)
    sa = np.array([sum(a[t] for t in groups[k]) for k in keys])
    sc = np.array([sum(c[t] for t in groups[k]) for k in keys])
    g = len(keys)
    rng = np.random.default_rng(kw.get('seed', 20261022))
    counts = rng.multinomial(g, np.full(g, 1.0 / g), size=kw.get('b', B_DEFAULT)).astype(float)
    boot = 1 - (counts @ sa) / (counts @ sc)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return Interval(float(1 - sa.sum() / sc.sum()), float(lo), float(hi), len(common), g)
