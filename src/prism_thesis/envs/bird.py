"""BIRD-Agent：把 BIRD dev 的文本到 SQL 问题改为多轮、只读的 SQLite 智能体任务。

每道题给出自然语言问题、官方提示（evidence）与数据库的表信息；智能体每轮执行一条 SQL 并看到结果，
确认后用 "Action: Answer / Final Answer: submit" 提交“最近一次成功执行的查询”的结果（与 InterCode 的
submit 协议相同）。判分为 BIRD 官方的执行准确率（EX）：提交结果的行集合与参考 SQL 的行集合相等。

为什么用 SQLite 而不是 MySQL：BIRD 的参考 SQL 是 SQLite 方言，官方 EX 也在 SQLite 上计算；MySQL 版
（Mini-Dev）只覆盖 500 题，跨方言比较结果并不可靠。用原生 SQLite 可以使用全部 dev 题目并保持官方判分。

数据由 scripts/pub/prepare_bird.py 生成（data/bird/）：
  tasks.jsonl        每行一道题：question_id、db_id、question、evidence、SQL、difficulty、gold_hash、gold_rows
  splits.json        题目 → 划分（build / search / calib / test / reserve）、题目组、是否排除
  schema/<db>.json   每张表的建表语句、列说明与样例行
  db_manifest.json   各 .sqlite 文件的大小与 SHA-256（计入实验指纹）
数据库文件本身（约 1.4 GB）不进仓库，路径由配置 bird.db_root 给出。

上下文单元：核心（角色、动作格式、任务与提示）；规则；SQLite 函数名（按问题关键词拆成 fn_predicted /
fn_other）；每张表三类单元——建表语句、列说明、样例行——按“问题或提示中是否提到该表或其列”拆成
*_pred / *_other 两组。规则与词表在看到任何 BIRD 运行结果之前写定。
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .. import parser
from ..env import ExecResult, InfrastructureError, split_statements
from ..llm import ScriptedLLM
from ..memory import signature as sig
from ..policies.base import TaskFeatures
from .base import Environment

ROOT = Path(__file__).resolve().parents[3]
SPLITS = ('build', 'search', 'calib', 'test', 'reserve')


@dataclass(frozen=True)
class BirdTask:
    task_id: int                 # BIRD question_id
    group_id: str                # '<db_id>/g<编号>'：同库内参考 SQL 骨架或问题文本相同的题目为一组
    split: str
    instruction: str             # 问题
    evidence: str                # 官方提示（外部知识），可为空
    db_id: str
    reference_sql: str           # 只用于核对数据与脚本化演练，不进入提示
    difficulty: str
    gold_hash: str               # 参考结果行集合的哈希
    gold_rows: int

    is_query = True


def result_hash(rows) -> str:
    """行集合的规范哈希（BIRD 官方 EX 比较的是 set(行元组)，与顺序和重复无关）。"""
    canon = sorted(repr(tuple(r)) for r in {tuple(r) for r in rows})
    return hashlib.sha256('\n'.join(canon).encode('utf-8', 'replace')).hexdigest()[:32]


def connect_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f'file:{Path(path).as_posix()}?mode=ro', uri=True, check_same_thread=False)
    conn.text_factory = lambda b: b.decode('utf-8', errors='replace')
    return conn


def run_with_timeout(conn: sqlite3.Connection, sql: str, timeout_s: float, max_rows: int):
    """执行一条语句；返回 (行列表, 是否超过 max_rows)。超时抛 TimeoutError。"""
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 20000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        return [tuple(r) for r in rows[:max_rows]], len(rows) > max_rows
    except sqlite3.OperationalError as e:
        if 'interrupt' in str(e).lower():
            raise TimeoutError from None
        raise
    finally:
        conn.set_progress_handler(None, 0)


# ---- 提示单元 ----
CORE = (
    ('role', "You answer the user's question by querying a SQLite database."),
    ('op_format', 'To run a query output exactly:\nAction: Operation\n```sql\nYOUR_SQL\n```'),
    ('answer_action', 'When the last query you ran returns exactly the rows that answer the question, '
                      'submit its result by outputting exactly:\nAction: Answer\nFinal Answer: submit'),
)
RULES = (
    ('observe', 'Use actual tool observations.'),
    ('one_stmt', 'Use exactly one SQL statement per operation; the database is read-only.'),
    ('dialect', 'Use SQLite syntax and functions; quote identifiers that contain spaces or special characters '
                'with backticks.'),
    ('explore', 'If a table, column or value format is unclear, inspect it first with a small '
                'SELECT ... LIMIT query.'),
    ('final_shape', 'The submitted query must return only the columns asked for, in the order asked, '
                    'without extra columns.'),
    ('use_hint', 'When a hint defines how a term maps to columns, values or a formula, follow it.'),
    ('rounds', None),            # 文本随配置的轮数生成
)
FUNCTIONS = ('COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'TOTAL', 'GROUP_CONCAT', 'ROUND', 'ABS', 'CAST', 'COALESCE',
             'IFNULL', 'NULLIF', 'IIF', 'LENGTH', 'UPPER', 'LOWER', 'SUBSTR', 'INSTR', 'REPLACE', 'TRIM', 'LIKE',
             'STRFTIME', 'DATE', 'JULIANDAY', 'ROW_NUMBER', 'RANK', 'DENSE_RANK')
FUNCTION_HEADER = 'Available SQL functions:'
FUNCTION_KEYWORDS = {
    'AVG': ('average', 'mean'),
    'SUM': ('total', 'sum', 'combined'),
    'TOTAL': ('total',),
    'COUNT': ('number of', 'count', 'how many', 'at least', 'more than', 'fewer than', 'percentage', 'ratio',
              'proportion'),
    'MAX': ('maximum', 'highest', 'largest', 'longest', 'most', 'latest', 'greatest', 'oldest', 'best'),
    'MIN': ('minimum', 'lowest', 'smallest', 'shortest', 'least', 'earliest', 'youngest', 'worst'),
    'GROUP_CONCAT': ('concatenate', 'comma-separated'),
    'ROUND': ('round', 'decimal'),
    'ABS': ('absolute', 'difference'),
    'CAST': ('percentage', 'percent', 'ratio', 'rate', 'average', 'proportion', 'divide'),
    'IIF': ('percentage', 'percent', 'ratio', 'proportion', 'whether'),
    'LENGTH': ('length of', 'characters'),
    'SUBSTR': ('first letter', 'starts with', 'begin with', 'last digit', 'initial'),
    'LIKE': ('contain', 'starts with', 'begin with', 'ends with', 'include'),
    'STRFTIME': ('year', 'month', 'day of', 'date'),
    'DATE': ('date',),
    'JULIANDAY': ('days between', 'how long', 'duration', 'age'),
    'ROW_NUMBER': ('rank', 'top ', 'second', 'third', 'nth'),
    'RANK': ('rank',),
    'DENSE_RANK': ('rank',),
}
RULE_IDS = tuple(u for u, _ in RULES)
FUNCTION_IDS = frozenset(f'fn:{f}' for f in FUNCTIONS)
CORE_IDS = frozenset(u for u, _ in CORE) | {'task'}
TABLE_KINDS = ('tbl', 'desc', 'smp')
TABLE_GROUPS = {'tbl': 'schema', 'desc': 'desc', 'smp': 'samples'}
SELECTABLE_GROUPS = RULE_IDS + ('fn_predicted', 'fn_other', 'schema_pred', 'schema_other', 'desc_pred',
                                'desc_other', 'samples_pred', 'samples_other')
_WORD = re.compile(r'[a-z0-9]+')

ERROR_KINDS = (                  # (错误类别, 错误信息中的特征)；按顺序匹配
    ('no_column', 'no such column'), ('no_table', 'no such table'), ('no_function', 'no such function'),
    ('ambiguous', 'ambiguous column'), ('syntax', 'syntax error'), ('incomplete', 'incomplete input'),
    ('misuse_aggregate', 'misuse of'), ('wrong_args', 'wrong number of arguments'),
    ('readonly', 'readonly'), ('readonly', 'not authorized'),
)
RECOVERY = {
    'syntax': ('one_stmt', 'dialect', 'fn_predicted', 'fn_other'),
    'incomplete': ('one_stmt', 'dialect'),
    'multi_statement': ('one_stmt',), 'empty': ('one_stmt',), 'readonly': ('one_stmt',),
    'no_column': ('schema_pred', 'schema_other', 'desc_pred', 'desc_other', 'dialect'),
    'no_table': ('schema_pred', 'schema_other'),
    'ambiguous': ('schema_pred', 'schema_other'),
    'no_function': ('dialect', 'fn_predicted', 'fn_other'),
    'wrong_args': ('dialect', 'fn_predicted', 'fn_other'),
    'misuse_aggregate': ('dialect', 'fn_predicted', 'fn_other'),
    'timeout': ('explore', 'samples_pred', 'samples_other'),
}
MULTI_STATEMENT_ERROR = 'Error: only one SQL statement is allowed per operation.'


def error_kind(message: str) -> str:
    m = message.lower()
    return next((k for k, pat in ERROR_KINDS if pat in m), 'other')


def _clip_cell(v, n: int = 60):
    if isinstance(v, str) and len(v) > n:
        return v[:n] + '…'
    if isinstance(v, bytes):
        return '<blob>'
    return v


class BirdSession:
    """一道题的只读 SQLite 会话。last_rows 为最近一次成功执行的查询的全部结果行（提交时判分用）。"""

    def __init__(self, path: Path, timeout_s: float, obs_chars: int, max_rows: int):
        self.path, self.timeout_s, self.obs_chars, self.max_rows = path, timeout_s, obs_chars, max_rows
        self.last_rows: Optional[list] = None
        self.last_sql: Optional[str] = None

    def __enter__(self) -> 'BirdSession':
        if not Path(self.path).exists():
            raise InfrastructureError(f'数据库文件不存在：{self.path}')
        self.conn = connect_readonly(self.path)
        return self

    def __exit__(self, *exc) -> None:
        self.conn.close()

    def execute(self, sql: str) -> ExecResult:
        stmts = split_statements(sql)
        if not stmts:
            return ExecResult(False, 'Error: the SQL statement is empty.', None, 'empty')
        if len(stmts) != 1:
            return ExecResult(False, MULTI_STATEMENT_ERROR, None, 'multi_statement')
        try:
            rows, overflow = run_with_timeout(self.conn, stmts[0], self.timeout_s, self.max_rows)
        except TimeoutError:
            return ExecResult(False, f'Error: the query was stopped after {self.timeout_s:g} seconds.', None, 'timeout')
        except (sqlite3.Warning, sqlite3.ProgrammingError) as e:
            if 'one statement' in str(e).lower():
                return ExecResult(False, MULTI_STATEMENT_ERROR, None, 'multi_statement')
            return ExecResult(False, f'Error: {e}', None, error_kind(str(e)))
        except sqlite3.Error as e:
            return ExecResult(False, f'Error: {e}', None, error_kind(str(e)))
        if overflow:
            self.last_rows, self.last_sql = None, stmts[0]
            return ExecResult(True, f'The query returned more than {self.max_rows} rows; such a result cannot be '
                                    'submitted. Narrow the query.', None, 'ok')
        self.last_rows, self.last_sql = rows, stmts[0]
        return ExecResult(True, self.render(rows), None, 'ok')

    def render(self, rows: list) -> str:
        text = str(rows)
        if len(text) <= self.obs_chars:
            return text
        shown, used = [], 1
        for r in rows:
            piece = repr(tuple(_clip_cell(v) for v in r))
            if used + len(piece) + 2 > self.obs_chars and shown:
                break
            shown.append(piece)
            used += len(piece) + 2
        return f"[{', '.join(shown)}, ...] (showing {len(shown)} of {len(rows)} rows)"


class BirdEnv(Environment):
    name = 'bird'
    task_types = ('query',)
    sealed_splits = frozenset({'test'})
    memory_splits = ('build',)
    obs_label = 'SQLite'
    core_ids = CORE_IDS
    selectable_groups = SELECTABLE_GROUPS
    insight_system = ('You analyse attempts of an agent that answers questions by running single SQL statements '
                      'on a SQLite database and then submitting the result of its last query.')

    def __init__(self, cfg: dict):
        b = cfg.get('bird', {})
        self.data_dir = self._path(b.get('data_dir', 'data/bird'))
        self.db_root = self._path(b.get('db_root', 'data/bird/dev_databases'))
        self.timeout_s = float(b.get('timeout_s', 30))
        self.obs_chars = int(b.get('obs_chars', 2000))
        self.max_rows = int(b.get('max_rows', 100000))
        self.max_rounds = int(cfg['agent']['max_rounds'])
        self._schema = lru_cache(maxsize=None)(self._load_schema)
        self._features = lru_cache(maxsize=4096)(self._task_features)

    @staticmethod
    def _path(p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    # ---- 数据 ----
    def db_path(self, db_id: str) -> Path:
        return self.db_root / db_id / f'{db_id}.sqlite'

    def _load_schema(self, db_id: str) -> dict:
        d = json.loads((self.data_dir / 'schema' / f'{db_id}.json').read_text(encoding='utf-8'))
        d['by_name'] = {t['name']: t for t in d['tables']}
        return d

    def load_tasks(self, split=None, include_excluded=False):
        meta = {m['task_id']: m for m in
                json.loads((self.data_dir / 'splits.json').read_text(encoding='utf-8'))['tasks']}
        wanted = None if split is None else ((split,) if isinstance(split, str) else tuple(split))
        out = []
        for line in (self.data_dir / 'tasks.jsonl').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            m = meta[r['question_id']]
            if (m['excluded'] and not include_excluded) or (wanted is not None and m['split'] not in wanted):
                continue
            out.append(BirdTask(r['question_id'], m['group_id'], m['split'], r['question'], r.get('evidence') or '',
                                r['db_id'], r['SQL'], r.get('difficulty', ''), r.get('gold_hash') or '',
                                r.get('gold_rows') or 0))
        return sorted(out, key=lambda t: t.task_id)

    def data_files(self):
        return [self.data_dir / 'tasks.jsonl', self.data_dir / 'splits.json', self.data_dir / 'db_manifest.json',
                *sorted((self.data_dir / 'schema').glob('*.json'))]

    def fingerprint(self):
        return {'env': self.name, 'timeout_s': self.timeout_s, 'obs_chars': self.obs_chars,
                'max_rows': self.max_rows}

    # ---- 单元 ----
    def tables(self, task) -> list[str]:
        return [t['name'] for t in self._schema(task.db_id)['tables']]

    def all_ids(self, task) -> frozenset:
        ids = set(CORE_IDS) | set(RULE_IDS) | FUNCTION_IDS
        for t in self._schema(task.db_id)['tables']:
            ids.add(f"tbl:{t['name']}")
            if any(c.get('description') for c in t['columns']):
                ids.add(f"desc:{t['name']}")
            if t.get('samples'):
                ids.add(f"smp:{t['name']}")
        return frozenset(ids)

    def unit_text(self, uid, task) -> str:
        if uid == 'task':
            hint = f'\nHint: {task.evidence}' if task.evidence else ''
            return f'{task.instruction}{hint}\nDatabase: {task.db_id}'
        if uid == 'rounds':
            return f'You have at most {self.max_rounds} rounds, and submitting the answer takes one of them.'
        if uid in FUNCTION_IDS:
            return uid[3:]
        kind, _, name = uid.partition(':')
        if kind in TABLE_KINDS:
            t = self._schema(task.db_id)['by_name'][name]
            if kind == 'tbl':
                return t['ddl']
            if kind == 'desc':
                lines = [f"- {c['name']}: {c['description']}" for c in t['columns'] if c.get('description')]
                return f'Columns of {name}:\n' + '\n'.join(lines)
            cols = ', '.join(c['name'] for c in t['columns'])
            rows = '\n'.join(repr(tuple(_clip_cell(v, 40) for v in r)) for r in t['samples'])
            return f'Sample rows of {name} ({cols}):\n{rows}'
        return dict(CORE + RULES)[uid]

    def render_rules(self, unit_ids, task) -> str:
        ids = set(unit_ids)
        lines = [self.unit_text(u, task) for u, _ in CORE if u in ids]
        lines += [self.unit_text(u, task) for u in RULE_IDS if u in ids]
        fns = [f for f in FUNCTIONS if f'fn:{f}' in ids]
        if fns:
            lines.append(f"{FUNCTION_HEADER} {', '.join(fns)}.")
        return '\n'.join(lines)

    def render_system(self, unit_ids, task) -> str:
        return self.render_rules(unit_ids, task)

    def render_first_user(self, unit_ids, task) -> str:
        ids = set(unit_ids)
        parts = [self.unit_text('task', task)]
        for kind, title in (('tbl', 'Tables:'), ('desc', 'Column descriptions:'), ('smp', 'Sample rows:')):
            texts = [self.unit_text(f'{kind}:{n}', task) for n in self.tables(task) if f'{kind}:{n}' in ids]
            if texts:
                parts.append(title + '\n' + ('\n\n' if kind != 'tbl' else '\n').join(texts))
        return '\n\n'.join(parts)

    # ---- 特征 ----
    def predicted_tables(self, task) -> frozenset:
        """问题或提示中提到的表：表名（或去掉复数 s）作为词出现，或其某一列名（长度不小于 4）出现。
        一张表都没有提到时视为全部提到。"""
        text = f'{task.instruction} {task.evidence}'.lower()
        words = set(_WORD.findall(text))
        flat = ' '.join(_WORD.findall(text))
        hit = set()
        for t in self._schema(task.db_id)['tables']:
            name = ' '.join(_WORD.findall(t['name'].lower()))
            names = {name, name.rstrip('s')} if len(name) > 3 else {name}
            if any((n in words) or (' ' in n and n in flat) for n in names if n):
                hit.add(t['name'])
                continue
            for c in t['columns']:
                col = ' '.join(_WORD.findall(c['name'].lower()))
                if len(col) >= 4 and ((col in words) or (' ' in col and col in flat)):
                    hit.add(t['name'])
                    break
        return frozenset(hit or self.tables(task))

    def _task_features(self, task) -> TaskFeatures:
        text = f'{task.instruction} {task.evidence}'.lower()
        fns = frozenset(f for f, kws in FUNCTION_KEYWORDS.items() if any(k in text for k in kws))
        pred = self.predicted_tables(task)
        ids = self.all_ids(task)
        groups: dict[str, set] = {g: set() for g in SELECTABLE_GROUPS}
        for u in ids - CORE_IDS:
            if u in FUNCTION_IDS:
                groups['fn_predicted' if u[3:] in fns else 'fn_other'].add(u)
                continue
            kind, _, name = u.partition(':')
            if kind in TABLE_KINDS:
                groups[f"{TABLE_GROUPS[kind]}_{'pred' if name in pred else 'other'}"].add(u)
            else:
                groups[u].add(u)
        return TaskFeatures('query', fns, {g: frozenset(v) for g, v in groups.items()}, env=self)

    def task_features(self, task) -> TaskFeatures:
        return self._features(task)

    # ---- 交互 ----
    def first_action(self, response):
        return parser.first_action(response)

    def parse(self, response):
        return parser.parse(response)

    def grade(self, task, session, status, answer):
        rows = session.last_rows
        ok = status == 'answered' and rows is not None and result_hash(rows) == task.gold_hash
        return ok, {'submitted_sql': session.last_sql, 'submitted_rows': None if rows is None else len(rows)}

    def recovery_groups(self, last_exec):
        if last_exec is None or last_exec.ok:
            return frozenset()
        return frozenset(RECOVERY.get(last_exec.kind, SELECTABLE_GROUPS))

    def signature(self, result, action):
        return sig.signature(result, action)       # 错误类别 | SQL 结构特征

    def expert_drop(self, phase, task_type):
        """人工规则（看到 BIRD 运行结果之前写定）：初始只发预测会用到的函数与提到的表的样例行；
        成功执行后省略语法类规则、函数、样例行，以及没有提到的表的建表语句与列说明。"""
        if phase == 'after_error':
            return ()
        if phase == 'initial':
            return ('fn_other', 'samples_other')
        return ('one_stmt', 'dialect', 'explore', 'fn_predicted', 'fn_other', 'samples_pred', 'samples_other',
                'schema_other', 'desc_other')

    # ---- 记忆 ----
    def retrieval_prefer(self, experience, task) -> bool:
        return any(g.startswith(f'{task.db_id}/') for g in experience.source_groups)

    def example_text(self, task, actions, answer, max_answer_chars):
        sql = actions[-1] if not isinstance(actions, str) else actions     # 提交的是最后一次成功的查询
        hint = f'\nHint: {task.evidence}' if task.evidence else ''
        return (f'Example question (database {task.db_id}): {task.instruction}{hint}\n'
                f'Action: Operation\n```sql\n{sql}\n```\nAction: Answer\nFinal Answer: submit')

    def repair_text(self, error_line, failed, fixed):
        return f'Fix for a similar error: {error_line}\nFailed SQL:\n{failed}\nCorrected SQL:\n{fixed}'

    # ---- 运行 ----
    def open_session(self, backend, task, idx, cfg):
        return BirdSession(self.db_path(task.db_id), self.timeout_s, self.obs_chars, self.max_rows)

    def scripted(self, kind, task):
        def op(sql):
            return f'I will run the query.\nAction: Operation\n```sql\n{sql}\n```'

        def reply(messages):
            n = sum(m['role'] == 'assistant' for m in messages)
            if kind == 'gold_after_error':
                if n == 0:
                    return op('SELECT no_such_column FROM sqlite_master')
                n -= 1
            return op(task.reference_sql) if n == 0 else 'Action: Answer\nFinal Answer: submit'
        if kind not in ('gold', 'gold_after_error'):
            raise KeyError(kind)
        return ScriptedLLM(reply, model=f'scripted-{kind}')

    def record_extras(self, task, features) -> dict:
        return {'task_type_pred': features.task_type, 'is_query': True, 'db_id': task.db_id,
                'difficulty': task.difficulty, 'predicted_functions': sorted(features.predicted_functions)}
