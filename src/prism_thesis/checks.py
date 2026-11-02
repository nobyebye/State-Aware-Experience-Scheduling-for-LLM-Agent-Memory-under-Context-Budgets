"""正式校准与最终比较之前的输入检查。

正式模式要求：记录全部来自指定划分；每个条件覆盖该划分的全部题目（已排除的题目除外），
且每题的重复编号恰为 0..R−1，R 等于配置 reps 中为该阶段、该条件预定的次数；模型为配置中的目标模型，
不含脚本化智能体；每个条件只有一个实验指纹，且与当前配置、风险表内容与实验相关代码一致。
任何一项不满足都拒绝给出结论。
演练模式（rehearsal）只检查各条件题目集合一致、每题重复完整，用于不花钱的流程演练，结果不得用于冻结或论文。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Optional, Sequence, Union

from .data import load_tasks
from .runner import experiment_fingerprint

REHEARSAL_BANNER = '【演练模式】输入未经正式检查，以下结果不得用于冻结、校准结论或论文。'


def expected_reps(manifest: Mapping[str, int], condition: str) -> Optional[int]:
    """按预定次数表查条件的重复次数：精确匹配 → 最长前缀（键以 * 结尾）→ '*'。"""
    if condition in manifest:
        return manifest[condition]
    prefixes = [k for k in manifest if k.endswith('*') and k != '*' and condition.startswith(k[:-1])]
    if prefixes:
        return manifest[max(prefixes, key=len)]
    return manifest.get('*')


def check_records(records: Sequence[Mapping], conditions: Sequence[str], cfg: dict, split: Optional[str],
                  reps: Union[Mapping[str, int], int, None], rehearsal: bool = False) -> list[str]:
    """返回发现的问题（空列表表示通过）。conditions 为必须出现的条件；reps 为预定次数表（或统一次数）。
    正式模式下 reps 必须给出，且每个条件都要能查到预定次数。"""
    problems: list[str] = []
    by: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for r in records:
        by[r['condition']][r['task_id']].add(r['rep'])
    missing = [c for c in conditions if c not in by]
    if missing:
        problems.append(f'缺少条件 {missing}')
    present = [c for c in conditions if c in by]
    task_sets = {c: frozenset(by[c]) for c in present}
    if len(set(task_sets.values())) > 1:
        sizes = {c: len(s) for c, s in task_sets.items()}
        problems.append(f'各条件的题目集合不同（题数 {sizes}），不能配对比较')
    for c in present:
        rep_sets = {frozenset(v) for v in by[c].values()}
        counts = sorted({len(v) for v in rep_sets})
        if len(rep_sets) != 1 or next(iter(rep_sets)) != frozenset(range(counts[-1])):
            problems.append(f'条件 {c} 各题的重复不完整或不一致（每题重复次数 {counts}）')
        elif not rehearsal:
            want = reps if isinstance(reps, int) else expected_reps(reps or {}, c)
            if want is None:
                problems.append(f'条件 {c} 在配置 reps 中没有预定重复次数')
            elif counts[-1] != want:
                problems.append(f'条件 {c} 每题重复 {counts[-1]} 次，预定 {want} 次')
    if rehearsal:
        return problems
    splits = sorted({str(r.get('split')) for r in records})
    if split is not None and splits != [split]:
        problems.append(f'记录来自划分 {splits}，要求只来自 {split}')
    if split is not None:
        expected = frozenset(t.task_id for t in load_tasks(split))
        for c in present:
            if task_sets[c] != expected:
                problems.append(f'条件 {c} 覆盖 {len(task_sets[c] & expected)}/{len(expected)} 道 {split} 题'
                                + (f'，另有 {len(task_sets[c] - expected)} 道不属于该划分' if task_sets[c] - expected else ''))
    models = sorted({str(r.get('model')) for r in records})
    if models != [cfg['model']['model']]:
        problems.append(f"记录中的模型为 {models}，要求只有 {cfg['model']['model']}（不接受脚本化智能体）")
    for c in present:
        fps = sorted({str(r.get('fingerprint')) for r in records if r['condition'] == c})
        want = experiment_fingerprint(cfg, c, paid=True)
        if fps != [want]:
            problems.append(f'条件 {c} 的实验指纹为 {fps}，与当前配置、代码与风险表内容的 {want} 不一致')
    return problems
