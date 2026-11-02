"""离线构建记忆库：从构建集与开发集上不带记忆（nomem）的运行记录中整理经验条目。

- 成功示例：每道题取一次成功运行（轮数最少、重复编号最小者），记录任务指令、按顺序执行成功的全部 SQL
  （多步操作如先 UPDATE 再 DELETE 都保留；出错的操作不写入示例）与最终答案；
  答案超过 max_answer_chars 个字符时只说明格式，不写出具体值；
- 修复经验：成功运行中“某轮 SQL 出错、之后某轮 SQL 执行成功”的片段，记录错误签名、错误信息首行、
  出错的 SQL 与改正后的 SQL；同一道题同一签名只保留一条；
- 洞见（参照 ExpeL）：同一道题既有失败又有成功的运行，组成“失败—成功”对，按任务类型分批交给模型，
  请模型写出可迁移的一般规则。这是唯一调用模型的步骤，只在构建记忆库时进行；每条洞见记录产生它的
  全部来源题目与题目组，检索时按同组排除。
测试集记录一律拒绝。
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable, Iterable, Mapping, Optional, Sequence

from ..data import Task
from ..env import ExecResult
from ..features import task_type
from . import signature as sig
from .store import FORBIDDEN_SPLITS, Experience

MAX_ERROR_CHARS = 200
MAX_OBS_CHARS = 300


def _exec_of(turn: Mapping) -> Optional[ExecResult]:
    e = turn.get('exec')
    if e is None:
        return None
    return ExecResult(bool(e['ok']), turn.get('observation', ''), e.get('error_code'), e.get('kind', 'ok'))


def check_sources(records: Iterable[Mapping]) -> None:
    bad = sorted({r['task_id'] for r in records if r.get('split') in FORBIDDEN_SPLITS})
    if bad:
        raise ValueError(f'输入中有 {len(bad)} 道测试集题目的记录，测试集不得进入记忆库')


def _usable(records: Iterable[Mapping]) -> list[Mapping]:
    rs = [r for r in records if r.get('status') != 'infrastructure_error']
    return sorted(rs, key=lambda r: (r['task_id'], r['rep']))


def example_text(task: Task, sqls: Sequence[str], answer: str, max_answer_chars: int) -> str:
    if isinstance(sqls, str):
        sqls = [sqls]
    if answer is None or len(answer) > max_answer_chars:
        answer = '(the result rows exactly as returned by MySQL)' if task_type(task.instruction) == 'query' \
            else 'done'
    ops = ''.join(f'Action: Operation\n```sql\n{sql}\n```\n' for sql in sqls)
    return f'Example task: {task.instruction}\n{ops}Action: Answer\nFinal Answer: {answer}'


def repair_text(error_line: str, failed: str, fixed: str) -> str:
    return f'Fix for a similar error: {error_line}\nFailed SQL:\n{failed}\nCorrected SQL:\n{fixed}'


def _source(task: Task) -> dict:
    return {'source_tasks': (task.task_id,), 'source_groups': (task.group_id,), 'source_splits': (task.split,)}


def extract_examples(records: Sequence[Mapping], tasks: Mapping[int, Task],
                     max_answer_chars: int = 300) -> list[Experience]:
    best: dict[int, tuple] = {}
    for r in _usable(records):
        if not r.get('success'):
            continue
        ok_sql = [t['content'] for t in r.get('turns', []) if t.get('action') == 'operation'
                  and t.get('exec', {}).get('ok')]
        if not ok_sql:
            continue
        key = (len(r.get('turns', [])), r['rep'])
        if r['task_id'] not in best or key < best[r['task_id']][0]:
            best[r['task_id']] = (key, tuple(ok_sql), r.get('answer'))
    out = []
    for tid in sorted(best):
        task = tasks[tid]
        _, sqls, answer = best[tid]
        out.append(Experience(f'ex-{tid}', 'example', example_text(task, sqls, answer, max_answer_chars),
                              task.instruction, task_type(task.instruction), **_source(task)))
    return out


def extract_repairs(records: Sequence[Mapping], tasks: Mapping[int, Task]) -> list[Experience]:
    out, seen = [], set()
    per_task: dict[int, int] = defaultdict(int)
    for r in _usable(records):
        if not r.get('success'):
            continue
        turns = [t for t in r.get('turns', []) if t.get('action') == 'operation' and 'exec' in t]
        for i, t in enumerate(turns):
            res = _exec_of(t)
            if res.ok:
                continue
            fix = next((u for u in turns[i + 1:] if u['exec']['ok']), None)
            if fix is None:
                continue
            s = sig.signature(res, t['content'])
            if (r['task_id'], s) in seen:
                continue
            seen.add((r['task_id'], s))
            task = tasks[r['task_id']]
            n = per_task[task.task_id]
            per_task[task.task_id] += 1
            line = (res.text or '').strip().split('\n')[0][:MAX_ERROR_CHARS]
            out.append(Experience(f'rp-{task.task_id}-{n}', 'repair', repair_text(line, t['content'], fix['content']),
                                  task.instruction, task_type(task.instruction), **_source(task),
                                  error_key=sig.error_key(res), signature=s))
    return out


# ---- 洞见（离线，调用模型） ----

INSIGHT_SYSTEM = ('You analyse attempts of an agent that operates a MySQL database through single SQL '
                  'statements and then gives a final answer.')
INSIGHT_PROMPT = ('Below are pairs of attempts at database tasks. In each pair, the first attempt failed and the '
                  'second attempt succeeded on the same task.\n\n{pairs}\n\nWrite at most {n} general rules that '
                  'would help the agent succeed on similar future tasks. Each rule must be a single line starting '
                  "with '- ', must not mention specific table names, column names or values, and must apply to "
                  'other tasks as well. Output only the rules.')


def render_trajectory(task: Task, rec: Mapping) -> str:
    lines = [f'Task: {task.instruction}']
    for t in rec.get('turns', []):
        lines.append(f"Agent: {t.get('response', '').strip()}")
        if 'observation' in t:
            lines.append(f"MySQL: {t['observation'][:MAX_OBS_CHARS]}")
    lines.append(f"Outcome: {'success' if rec.get('success') else 'failure (' + str(rec.get('status')) + ')'}")
    return '\n'.join(lines)


def insight_batches(records: Sequence[Mapping], tasks: Mapping[int, Task], batch_size: int = 4) -> list[dict]:
    """同一道题的第一次失败与第一次成功组成一对，按任务类型分批（批内按题目编号）。"""
    by_task: dict[int, dict] = defaultdict(dict)
    for r in _usable(records):
        slot = 'succ' if r.get('success') else 'fail'
        by_task[r['task_id']].setdefault(slot, r)
    pairs: dict[str, list] = defaultdict(list)
    for tid in sorted(by_task):
        d = by_task[tid]
        if 'succ' in d and 'fail' in d:
            task = tasks[tid]
            pairs[task_type(task.instruction)].append((task, d['fail'], d['succ']))
    out = []
    for ttype in sorted(pairs):
        ps = pairs[ttype]
        for i in range(0, len(ps), batch_size):
            out.append({'task_type': ttype, 'pairs': ps[i:i + batch_size]})
    return out


def insight_messages(batch: Mapping, max_rules: int) -> list[dict]:
    blocks = []
    for n, (task, fail, succ) in enumerate(batch['pairs'], 1):
        blocks.append(f'Pair {n}\n[Failed attempt]\n{render_trajectory(task, fail)}\n'
                      f'[Successful attempt]\n{render_trajectory(task, succ)}')
    return [{'role': 'system', 'content': INSIGHT_SYSTEM},
            {'role': 'user', 'content': INSIGHT_PROMPT.format(pairs='\n\n'.join(blocks), n=max_rules)}]


_RULE = re.compile(r'^\s*[-*•]\s+(.+?)\s*$')


def parse_rules(text: str, max_rules: int) -> list[str]:
    rules = []
    for line in (text or '').splitlines():
        m = _RULE.match(line)
        if m and m.group(1):
            rules.append(m.group(1))
    return rules[:max_rules]


def distill_insights(batches: Sequence[Mapping], chat: Callable[[list[dict]], str],
                     max_rules: int = 3) -> list[Experience]:
    """chat(messages) -> 回复文本。相同文本（忽略大小写与首尾空白）的洞见合并，来源取并集。"""
    merged: dict[str, dict] = {}
    for b_idx, batch in enumerate(batches):
        reply = chat(insight_messages(batch, max_rules))
        tasks = [p[0] for p in batch['pairs']]
        for j, rule in enumerate(parse_rules(reply, max_rules)):
            norm = ' '.join(rule.lower().split())
            d = merged.setdefault(norm, {'eid': f'in-{b_idx}-{j}', 'text': rule, 'types': set(),
                                         'tasks': set(), 'groups': set(), 'splits': set()})
            d['types'].add(batch['task_type'])
            d['tasks'] |= {t.task_id for t in tasks}
            d['groups'] |= {t.group_id for t in tasks}
            d['splits'] |= {t.split for t in tasks}
    out = []
    for d in merged.values():
        ttype = next(iter(d['types'])) if len(d['types']) == 1 else 'any'
        out.append(Experience(d['eid'], 'insight', f"Rule: {d['text']}", d['text'], ttype,
                              tuple(sorted(d['tasks'])), tuple(sorted(d['groups'])), tuple(sorted(d['splits']))))
    return sorted(out, key=lambda e: e.eid)
