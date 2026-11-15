"""由 LifelongAgentBench 的 os_interaction 原始文件生成操作系统环境的数据（data/os/）。

输入：HuggingFace csyq/LifelongAgentBench 的 os_interaction/train-00000-of-00001.parquet，或官方仓库的
entry_dict.json。每道题的字段：instruction、initialization_command_item、evaluation_info
（ground_truth_command_item 与 evaluation_command_item）、skill_list；字段可以是嵌套对象，也可以是
JSON / Python 字面量字符串。
步骤：
1. 转成统一格式（tasks.jsonl）；
2. 在容器中核对每道题（需要 Docker）：初始化成功；不做任何操作时评测不通过（否则不操作也算成功，排除）；
   执行参考解法后评测通过（否则无法判分，排除）；
3. 题目组：指令去掉引号内容、路径与数字后相同的题目为一组；
4. 划分：build / search / calib / test，多出的进 reserve（固定种子）。
"""
from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ..env import InfrastructureError
from ..envs.os_shell import OsEnv, task_from_row
from .bird_prep import make_splits

DEFAULT_SIZES = {'build': 120, 'search': 80, 'calib': 120, 'test': 180}


def _lit(v):
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in '[{':
            for load in (json.loads, ast.literal_eval):
                try:
                    return load(s)
                except (ValueError, SyntaxError):
                    continue
    return v


def _cmd(d) -> Optional[dict]:
    d = _lit(d)
    if not d:
        return None
    if isinstance(d, str):
        return {'command_name': 'bash', 'script': d}
    return {'command_name': d.get('command_name') or 'bash', 'script': d.get('script') or ''}


def load_raw(path: Path) -> list[dict]:
    path = Path(path)
    if path.suffix == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        items = data.items() if isinstance(data, dict) else enumerate(data)
        return [{'sample_index': k, **v} for k, v in items]
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def convert(rows: Sequence[Mapping]) -> list[dict]:
    out = []
    for i, raw in enumerate(rows):
        ev = _lit(raw.get('evaluation_info')) or {}
        idx = raw.get('sample_index', i)
        out.append({
            'task_id': int(idx) if str(idx).lstrip('-').isdigit() else i,
            'instruction': raw['instruction'],
            'init': _cmd(raw.get('initialization_command_item')),
            'ground_truth': _cmd(ev.get('ground_truth_command_item')),
            'evaluation': _cmd(ev.get('evaluation_command_item')),
            'skills': list(_lit(raw.get('skill_list')) or []),
        })
    if len({t['task_id'] for t in out}) != len(out):
        raise ValueError('题号有重复')
    return sorted(out, key=lambda t: t['task_id'])


def group_key(instruction: str) -> str:
    s = re.sub(r'"[^"]*"|\'[^\']*\'|`[^`]*`', '?', instruction.lower())
    s = re.sub(r'(?:/[\w.\-]+)+/?', '/?', s)
    s = re.sub(r'\d+', '0', s)
    return ' '.join(re.findall(r'[a-z0-9/?]+', s))


def verify(env: OsEnv, rows: Sequence[Mapping], cfg: dict, log=print) -> dict[int, Optional[str]]:
    """题号 → 排除原因（None 表示可用）。"""
    out: dict[int, Optional[str]] = {}
    backend = env.make_backend(cfg, 'prep')
    backend.start()
    try:
        for i, r in enumerate(rows):
            reason = None
            if r['evaluation'] is None or r['ground_truth'] is None:
                reason = 'no_evaluation_or_ground_truth'
            else:
                task = task_from_row(dict(r), 'g', 'unassigned')
                try:
                    with env.open_session(backend, task, 'prep', cfg) as s:
                        if s.evaluate() == 0:
                            reason = 'passes_without_action'
                        else:
                            s.run(task.ground_truth, timeout_s=120)
                            if s.evaluate() != 0:
                                reason = 'ground_truth_fails'
                except InfrastructureError as e:
                    reason = f'init_fails: {str(e)[:80]}'
            out[r['task_id']] = reason
            if (i + 1) % 50 == 0:
                log(f'已核对 {i + 1}/{len(rows)}')
    finally:
        backend.stop()
    return out


def prepare(raw: Path, cfg: dict, sizes: Mapping[str, int] = DEFAULT_SIZES, seed: int = 20261115,
            skip_verify: bool = False, log=print) -> dict:
    env = OsEnv(cfg)
    rows = convert(load_raw(raw))
    reasons = {r['task_id']: None for r in rows} if skip_verify else verify(env, rows, cfg, log)
    keys: dict[str, str] = {}
    groups = {r['task_id']: keys.setdefault(group_key(r['instruction']), f'os/g{len(keys):04d}') for r in rows}
    shaped = [{'question_id': r['task_id'], 'db_id': 'os', 'excluded': reasons[r['task_id']] is not None}
              for r in rows]
    splits = make_splits(shaped, groups, sizes, seed)
    env.data_dir.mkdir(parents=True, exist_ok=True)
    with (env.data_dir / 'tasks.jsonl').open('w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    meta = [{'task_id': r['task_id'], 'split': splits.get(r['task_id'], 'excluded'), 'group_id': groups[r['task_id']],
             'excluded': reasons[r['task_id']] is not None, 'reason': reasons[r['task_id']]} for r in rows]
    (env.data_dir / 'splits.json').write_text(json.dumps(
        {'seed': seed, 'sizes': dict(sizes), 'verified': not skip_verify, 'tasks': meta},
        ensure_ascii=False, indent=1), encoding='utf-8')
    counts: dict[str, int] = defaultdict(int)
    for m in meta:
        counts[m['split']] += 1
    why: dict[str, int] = defaultdict(int)
    for v in reasons.values():
        if v:
            why[v.split(':')[0]] += 1
    summary = {'tasks': len(rows), 'splits': dict(counts), 'excluded': dict(why), 'groups': len(keys)}
    log(json.dumps(summary, ensure_ascii=False))
    return summary
