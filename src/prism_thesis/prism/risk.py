"""经验风险表：r̂(组 | 状态, 任务类型)，单位为百分点。组可以是指令组，也可以是经验组
（exp_top1 第 1 条相似示例、exp_rest 其余相似示例、insight 洞见、repair 修复经验）。

由开发集上的分层留一消融得到：条件 ablate:<组>@<状态> 只在处于该状态的轮次省略该组，
与完整记忆注入（full）逐题配对，r̂ = 成功率下降量（完整 − 消融）的 95% 组 bootstrap 区间上界。
状态取 any 表示所有轮次都省略（用于不区分状态的消融与静态基线）；任务类型取 any 表示不分类型。
互补单元的联合消融（ablate:<组1>+<组2>@<状态>）用于检验交互作用；交互作用的区间下界大于 0 时
把两组合并，合并组的风险取联合消融的估计。配置中的每一对互补组（pairs），只要两组在某状态、任务类型下
都有单独条目，就必须有相应的联合消融且覆盖足够；联合消融缺失、全部为基础设施错误或暴露不足时，
交互无法检验，两组在该状态、任务类型下的单独条目都不写入（部署时回退完整提示）。
消融对照 prism-singleonly 用 single_only_records 去掉联合消融后估计（pairs 为空），整个互补机制
（合并与交互无法检验时的删除）都不参与，结果只取决于完整提示与单组消融的记录。
风险表不构成任务失败风险的上界；准确率结论只来自独立校准与测试集检验。

估计对象：对全部题目施加“只在该状态省略该组”的策略后，整题成功率的平均下降（包含从未进入该状态的
运行），与 ε 约束整题成功率的含义一致；不按处理后才出现的状态筛选样本。
覆盖要求：只有实际省略过该组（该轮发送的单元少于完整提示）的运行才算“暴露”。暴露只在实际参与该项
估计的样本上统计（与完整提示配对的题目；交互检验为四个条件共同的题目）。暴露的题目组数少于
min_exposed_groups 时不写入该条目，选择器查不到风险，回退完整提示；不以未暴露的数据推断零风险。
各条目的暴露情况写入 meta['coverage']。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .. import stats
from ..prompts import ALL_IDS

ANY = 'any'


@dataclass
class RiskTable:
    entries: dict[str, float] = field(default_factory=dict)        # 'group|phase|type' -> pp
    merged: list[tuple[str, ...]] = field(default_factory=list)    # 合并的互补组
    meta: dict = field(default_factory=dict)

    @staticmethod
    def key(group: str, phase: str, task_type: str) -> str:
        return f'{group}|{phase}|{task_type}'

    def lookup(self, group: str, phase: str, task_type: str) -> Optional[float]:
        return self.entries.get(self.key(group, phase, task_type))

    def bundles(self) -> dict[str, dict]:
        """组合组：名称 → {'groups': [...], 'phase': 状态}（add_bundles 写入；没有时为空）。"""
        return self.meta.get('bundles', {})

    def merge_map(self) -> dict[str, str]:
        """原组名 → 合并后的组名。"""
        out = {}
        for gs in self.merged:
            name = '+'.join(gs)
            for g in gs:
                out[g] = name
        return out

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({'entries': self.entries, 'merged': [list(m) for m in self.merged],
                                    'meta': self.meta}, ensure_ascii=False, indent=1), encoding='utf-8')

    @classmethod
    def load(cls, path: Path) -> 'RiskTable':
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        return cls(d['entries'], [tuple(m) for m in d.get('merged', [])], d.get('meta', {}))


def parse_ablation(condition: str) -> Optional[tuple[tuple[str, ...], str]]:
    """'ablate:a+b@after_success' -> (('a','b'), 'after_success')"""
    if not condition.startswith('ablate:'):
        return None
    body, phase = condition[len('ablate:'):].rsplit('@', 1)
    return tuple(body.split('+')), phase


def single_only_records(records: Sequence[Mapping]) -> list:
    """prism-singleonly 用的输入：去掉全部联合消融记录。"""
    return [r for r in records if not ((a := parse_ablation(r['condition'])) and len(a[0]) > 1)]


def _dropped(turn: Mapping) -> bool:
    n = (turn.get('plan_info') or {}).get('n_dropped_units')
    if n is not None:
        return n > 0
    # 没有记录少发单元数时（早期记录）：只比较指令单元
    return {u for u in turn.get('units', ALL_IDS) if not u.startswith('exp:')} != ALL_IDS


def exposed(record: Mapping) -> bool:
    """该次运行是否至少有一轮实际少发了单元（消融真正生效）。经验组检索为空的题目不算暴露。"""
    return any(_dropped(t) for t in record.get('turns', []))


def coverage(records: Sequence[Mapping], condition: str, group_of: Mapping[int, str],
             tasks: Optional[set] = None) -> dict:
    """condition 在 tasks（实际参与估计的题目；None 表示全部）上的暴露情况。基础设施错误的记录不计。"""
    rs = [r for r in records if r['condition'] == condition and r.get('status') != 'infrastructure_error'
          and (tasks is None or r['task_id'] in tasks)]
    ex = [r for r in rs if exposed(r)]
    return {'runs': len(rs), 'exposed_runs': len(ex), 'exposed_tasks': len({r['task_id'] for r in ex}),
            'exposed_groups': len({group_of[r['task_id']] for r in ex})}


def estimate(records: Sequence[Mapping], group_of: Mapping[int, str], type_of: Mapping[int, str],
             control: str = 'full', b: int = stats.B_DEFAULT, seed: int = 20261022,
             min_exposed_groups: int = 10, pairs: Sequence[Sequence[str]] = ()) -> RiskTable:
    """从消融运行记录估计风险表，并做互补组检验。records 中每条需含 condition/task_id/success/turns。
    pairs 为配置中的互补组；记录中出现的联合消融也会检验。"""
    conditions = sorted({r['condition'] for r in records if parse_ablation(r['condition'])})
    drops: dict[tuple[tuple[str, ...], str, str], stats.Interval] = {}
    cover: dict[tuple[tuple[str, ...], str, str], dict] = {}
    for cond in conditions:
        groups, phase = parse_ablation(cond)
        for ttype in ('query', 'modify', ANY):
            sub = [r for r in records if ttype == ANY or type_of[r['task_id']] == ttype]
            try:
                d = stats.paired_difference(sub, control, cond, group_of, b=b, seed=seed)
            except ValueError:
                continue
            drops[(groups, phase, ttype)] = d
            paired = set(stats.per_task_mean(sub, control)) & set(stats.per_task_mean(sub, cond))
            cover[(groups, phase, ttype)] = coverage(sub, cond, group_of, paired)
    table = RiskTable(meta={'control': control, 'b': b, 'seed': seed, 'n_records': len(records),
                            'min_exposed_groups': min_exposed_groups,
                            'coverage': {f"{'+'.join(g)}|{p}|{t}": c for (g, p, t), c in cover.items()}})

    def covered(k) -> bool:
        return cover[k]['exposed_groups'] >= min_exposed_groups

    interactions = {}
    for (groups, phase, ttype), d in drops.items():
        if len(groups) == 1 and covered((groups, phase, ttype)):
            table.entries[RiskTable.key(groups[0], phase, ttype)] = 100 * d.high
    # 待检验的互补组：配置中的每一对（两组都有单独估计的状态、任务类型），加上记录中出现的联合消融
    joint_conds = {(frozenset(parse_ablation(c)[0]), parse_ablation(c)[1]): c
                   for c in conditions if len(parse_ablation(c)[0]) == 2}
    todo: dict[tuple[frozenset, str, str], tuple[str, ...]] = {}
    for pair in pairs:
        pair = tuple(pair)
        for (gs, phase, ttype) in drops:
            if gs == pair[:1] and (pair[1:], phase, ttype) in drops:
                todo.setdefault((frozenset(pair), phase, ttype), pair)
    for (gs, phase, ttype) in drops:
        if len(gs) == 2:
            todo[(frozenset(gs), phase, ttype)] = gs
    # 互补检验：交互 = 联合下降 − 单独下降之和（逐题配对后组 bootstrap）
    for (fs, phase, ttype), groups in sorted(todo.items(), key=lambda kv: (kv[1], kv[0][1], kv[0][2])):
        joint = joint_conds.get((fs, phase))
        name = f"{'+'.join(groups)}|{phase}|{ttype}"
        sub = [r for r in records if ttype == ANY or type_of[r['task_id']] == ttype]
        if joint is None:
            reason, jc = 'joint ablation not run', None
        else:
            singles = [f'ablate:{g}@{phase}' for g in groups]
            f = stats.per_task_mean(sub, control)
            j = stats.per_task_mean(sub, joint)
            s = [stats.per_task_mean(sub, c) for c in singles]
            common = set(f) & set(j) & set(s[0]) & set(s[1])
            jc = coverage(sub, joint, group_of, common)
            reason = None if jc['exposed_groups'] >= min_exposed_groups else \
                'joint ablation missing or under-exposed on common tasks'
        if reason:
            # 交互无法检验：两个组的单独条目也不写入（同时省略两组的风险未知），部署时回退完整提示。
            interactions[name] = {'untested': reason, **(jc or {})}
            for g in groups:
                table.entries.pop(RiskTable.key(g, phase, ttype), None)
            continue
        inter = {t: (f[t] - j[t]) - (f[t] - s[0][t]) - (f[t] - s[1][t]) for t in common}
        iv = stats.cluster_bootstrap(inter, group_of, b=b, seed=seed)
        interactions[name] = {**iv.as_dict(), **jc}
        if iv.low > 0 and groups not in table.merged:
            table.merged.append(groups)
    for groups in table.merged:
        for (gs, phase, ttype), d in drops.items():
            if gs == groups and covered((gs, phase, ttype)):
                table.entries[RiskTable.key('+'.join(groups), phase, ttype)] = 100 * d.high
    table.meta['interactions'] = interactions
    return table


def bundle_condition(groups: Sequence[str], phase: str) -> str:
    return f"ablate:{'+'.join(groups)}@{phase}"


def add_bundles(table: RiskTable, records: Sequence[Mapping], bundles: Mapping[str, Mapping],
                group_of: Mapping[int, str], type_of: Mapping[int, str], control: str = 'full',
                b: int = stats.B_DEFAULT, seed: int = 20261022, min_exposed_groups: int = 10) -> RiskTable:
    """在已有风险表上加入组合组（补充实验 PRISM-bundle）。组合的风险直接由联合消融估计：
    records 中条件 ablate:<组1>+…@<状态> 相对 control 的成功率下降的 95% 题目组 bootstrap 区间上界（百分点），
    按任务类型分别估计，暴露题目组少于 min_exposed_groups 时不写入。原表的单组条目不变。
    records 必须自带 control 的记录（与联合消融同一批运行，逐题配对）。"""
    names = set(bundles)
    clash = sorted(names & {k.split('|')[0] for k in table.entries})
    if clash:
        raise ValueError(f'组合名与已有组重名：{clash}')
    if not any(r['condition'] == control for r in records):
        raise ValueError(f'记录中没有对照条件 {control}')
    entries = dict(table.entries)
    cover, spec = {}, {}
    for name, bd in sorted(bundles.items()):
        groups, phase = list(bd['groups']), bd['phase']
        cond = bundle_condition(groups, phase)
        if not any(r['condition'] == cond for r in records):
            raise ValueError(f'记录中没有组合 {name} 的联合消融 {cond}')
        spec[name] = {'groups': groups, 'phase': phase, 'condition': cond}
        for ttype in ('query', 'modify', ANY):
            sub = [r for r in records if ttype == ANY or type_of[r['task_id']] == ttype]
            d = stats.paired_difference(sub, control, cond, group_of, b=b, seed=seed)
            paired = set(stats.per_task_mean(sub, control)) & set(stats.per_task_mean(sub, cond))
            c = coverage(sub, cond, group_of, paired)
            key = RiskTable.key(name, phase, ttype)
            cover[key] = {**c, 'estimate_pp': 100 * d.estimate, 'low_pp': 100 * d.low, 'high_pp': 100 * d.high}
            if c['exposed_groups'] >= min_exposed_groups:
                entries[key] = 100 * d.high
    meta = {**table.meta, 'bundles': spec, 'bundle_coverage': cover,
            'bundle_settings': {'control': control, 'b': b, 'seed': seed, 'min_exposed_groups': min_exposed_groups}}
    return RiskTable(entries, list(table.merged), meta)
