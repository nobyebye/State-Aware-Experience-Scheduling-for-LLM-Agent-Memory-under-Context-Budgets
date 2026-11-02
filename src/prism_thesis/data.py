"""数据加载：LifelongAgentBench 数据库任务（db_bench）的公开原始文件 + 固定的题目划分。

原始文件 data/db_bench.parquet 每行一道题，字段为 Python 字面量字符串：
  answer_info  {'direct': 查询题的标准答案行列表 或 None, 'md5': 修改题执行后的表哈希 或 None, 'sql': 参考 SQL}
  table_info   {'name': 表名, 'column_info_list': [{'name','type'}], 'row_list': 初始数据}
  instruction  自然语言任务
  skill_list   该题涉及的 SQL 技能
划分文件 data/splits.json 给出每道题所属的数据集（construction / development / test）、题目组和是否排除。
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parents[2] / 'data'


@dataclass(frozen=True)
class Column:
    name: str
    type: str


@dataclass(frozen=True)
class Task:
    task_id: int
    group_id: str
    split: str                      # construction / development / test
    instruction: str                # 自然语言任务
    table: str                      # 表名，同时作为数据库名（与官方实现一致）
    columns: tuple[Column, ...]
    rows: tuple[tuple, ...]         # 初始数据
    reference_sql: str              # 参考 SQL，只用于核对数据，不进入提示
    answer_direct: Optional[tuple[tuple, ...]]  # 查询题的标准答案
    answer_md5: Optional[str]       # 修改题的标准表哈希
    skills: tuple[str, ...]

    @property
    def schema_text(self) -> str:
        """官方 db_bench 附在任务指令后的表信息句（表名与列名）。"""
        return (f'The name of this table is {self.table}, and the headers of this table are '
                f'{", ".join(c.name for c in self.columns)}.')

    @property
    def official_prompt(self) -> str:
        """官方实现交给智能体的完整任务文本：任务指令 + 换行 + 表信息句。"""
        return f'{self.instruction}\n{self.schema_text}'

    @property
    def is_query(self) -> bool:
        """标准答案是否为查询结果（SELECT 题）。只用于分析，方法本身不得读取。"""
        return self.answer_direct is not None


def load_tasks(split: Optional[str] = None, include_excluded: bool = False,
               parquet: Path = DATA_DIR / 'db_bench.parquet',
               splits: Path = DATA_DIR / 'splits.json') -> list[Task]:
    import pyarrow.parquet as pq  # 只在加载数据时需要

    table = pq.read_table(parquet).to_pylist()
    meta = {r['task_id']: r for r in json.loads(splits.read_text(encoding='utf-8'))['tasks']}
    tasks = []
    for raw in table:
        tid = int(raw['sample_index'])
        m = meta[tid]
        if m['excluded'] and not include_excluded:
            continue
        if split is not None and m['split'] != split:
            continue
        answer = ast.literal_eval(raw['answer_info'])
        info = ast.literal_eval(raw['table_info'])
        direct = answer['direct']
        tasks.append(Task(
            task_id=tid,
            group_id=m['group_id'],
            split=m['split'],
            instruction=raw['instruction'],
            table=info['name'],
            columns=tuple(Column(c['name'], c['type']) for c in info['column_info_list']),
            rows=tuple(tuple(r) for r in info['row_list']),
            reference_sql=answer['sql'].strip(),
            answer_direct=None if direct is None else tuple(tuple(r) for r in direct),
            answer_md5=answer['md5'],
            skills=tuple(ast.literal_eval(raw['skill_list'])),
        ))
    tasks.sort(key=lambda t: t.task_id)
    return tasks
