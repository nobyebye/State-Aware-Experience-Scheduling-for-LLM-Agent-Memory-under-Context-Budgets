"""上下文单元：系统指令与任务描述被划分为可整体保留或省略的单元。

完整提示 = 全部单元，按 UNITS 的固定顺序排列；其他方法只决定每轮发送哪些单元，不改写单元文本
（LLMLingua-2 基线除外，它对可压缩文本做 token 级删除）。

单元的三种位置（kind）：
  system  系统指令中的一条规则或说明；
  schema  官方附在任务指令后的表信息句，属于可压缩内容；
  task    自然语言任务指令，所有条件中都原样发送，不参与选择。
group 是风险估计与选择的最小单位：同一组的单元总是一起保留或省略。
函数说明在运行时按任务预测拆成两组：fn_predicted（预测会用到的函数）与 fn_other（其余函数）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .data import Task


@dataclass(frozen=True)
class Unit:
    uid: str
    group: str
    text: str
    core: bool = False
    kind: str = 'system'


# 单元文本沿用预实验验证过的措辞（Llama-3.1-8B 在开发集前 40 题上 60% 成功，与预实验 59% 相当）；
# 改写后的措辞（提到“修改行”“done”）会让该模型在查询题中自行执行修改，成功率降到 15%–40%。
ACTION_UNITS = [
    Unit('role', 'core', "You operate a MySQL database to complete the user's task.", core=True),
    Unit('op_format', 'core', 'For a database operation output exactly:\nAction: Operation\n```sql\nYOUR_SQL\n```',
         core=True),
    Unit('answer_action', 'core', 'For the final answer output exactly:\n'
         'Action: Answer\nFinal Answer: YOUR_ANSWER_ON_ONE_LINE', core=True),
]

RULE_UNITS = [
    Unit('observe', 'observe', 'Use actual tool observations.'),
    Unit('one_stmt', 'one_stmt', 'Use exactly one SQL statement per operation; only the supplied task table is accessible.'),
    Unit('scope', 'scope', 'Do not use session variables, file operations, locks, hints or nondeterministic/system '
         'functions.'),
    Unit('answer_query', 'answer_query', 'For a query, Final Answer must be a literal list of tuples, '
         'e.g. [(1,"Alice"),(2,"Bob")].'),
    Unit('empty_scalar', 'empty_scalar', 'For an empty query result use []; for a scalar result use [(value,)].'),
    Unit('modify_answer', 'modify_answer', 'Do not claim success without observing the result.'),
    Unit('rounds', 'rounds', 'You have at most three rounds.'),
]

# 函数说明：函数集合覆盖 db_bench 任务可能用到的常见 MySQL 函数。
# 标准版（主实验）只列函数名；扩展版（指令长度受控实验）每个函数附一行用法说明。
FUNCTIONS = [
    ('COUNT', 'COUNT(expr): number of non-NULL values; COUNT(*) counts rows'),
    ('SUM', 'SUM(expr): sum of values'),
    ('AVG', 'AVG(expr): average of values'),
    ('MIN', 'MIN(expr): smallest value'),
    ('MAX', 'MAX(expr): largest value'),
    ('STDDEV', 'STDDEV(expr): population standard deviation'),
    ('GROUP_CONCAT', 'GROUP_CONCAT(expr): concatenate values of a group'),
    ('ROUND', 'ROUND(x, d): round x to d decimal places'),
    ('ABS', 'ABS(x): absolute value'),
    ('CEIL', 'CEIL(x): smallest integer not less than x'),
    ('FLOOR', 'FLOOR(x): largest integer not greater than x'),
    ('MOD', 'MOD(x, y): remainder of x divided by y'),
    ('POWER', 'POWER(x, y): x raised to the power y'),
    ('CONCAT', 'CONCAT(s1, s2, ...): join strings'),
    ('LENGTH', 'LENGTH(s): length of s in bytes'),
    ('CHAR_LENGTH', 'CHAR_LENGTH(s): length of s in characters'),
    ('UPPER', 'UPPER(s): s in upper case'),
    ('LOWER', 'LOWER(s): s in lower case'),
    ('SUBSTRING', 'SUBSTRING(s, pos, len): part of s'),
    ('TRIM', 'TRIM(s): s without leading and trailing spaces'),
    ('REPLACE', 'REPLACE(s, from, to): replace all occurrences in s'),
    ('LEFT', 'LEFT(s, n): first n characters of s'),
    ('COALESCE', 'COALESCE(a, b, ...): first non-NULL argument'),
    ('IFNULL', 'IFNULL(a, b): a if a is not NULL, else b'),
    ('IF', 'IF(cond, a, b): a if cond is true, else b'),
    ('CAST', 'CAST(expr AS type): convert expr to type, e.g. CAST(x AS DECIMAL(10,2))'),
    ('STR_TO_DATE', "STR_TO_DATE(s, fmt): parse a date string, e.g. STR_TO_DATE(d, '%Y-%m-%d')"),
    ('YEAR', 'YEAR(d): year part of a date'),
    ('MONTH', 'MONTH(d): month part of a date'),
    ('DATEDIFF', 'DATEDIFF(d1, d2): days between two dates'),
]
FUNCTION_HEADER = 'Available SQL functions:'
FUNCTION_UNITS = [Unit(f'fn:{name}', 'functions', name) for name, _ in FUNCTIONS]
FUNCTION_USAGE = dict(FUNCTIONS)
VARIANTS = ('standard', 'extended')
_variant = 'standard'


def set_variant(variant: str) -> None:
    """选择提示版本；在程序启动时按配置设置一次。"""
    global _variant
    if variant not in VARIANTS:
        raise ValueError(variant)
    _variant = variant


SCHEMA_STYLES = ('official', 'typed', 'legacy')
_overrides: dict[str, str] = {}
_schema_style = 'official'


def set_prompt_options(overrides: dict | None = None, schema_style: str = 'official') -> None:
    """配置中的提示调试选项（在程序启动时设置一次）：
    overrides     单元 id → 替换文本，用于对比不同措辞（只允许替换已有的非函数、非任务单元）；
    schema_style  official：官方表信息句（表名与列名）；typed：表名与带类型的列；
                  legacy：预实验的写法（Schema: {列名: 类型} 换行 Table: 表名）。"""
    global _overrides, _schema_style
    overrides = dict(overrides or {})
    bad = [u for u in overrides if u not in UNIT_BY_ID or u in FUNCTION_IDS or u in ('task', 'schema')]
    if bad:
        raise ValueError(f'prompt_overrides 中的单元不可替换：{bad}')
    if schema_style not in SCHEMA_STYLES:
        raise ValueError(schema_style)
    _overrides, _schema_style = overrides, schema_style

SCHEMA_UNIT = Unit('schema', 'schema', '', kind='schema')   # 文本随题目变化，见 unit_text
TASK_UNIT = Unit('task', 'task', '', core=True, kind='task')

UNITS: list[Unit] = ACTION_UNITS + RULE_UNITS + FUNCTION_UNITS + [SCHEMA_UNIT, TASK_UNIT]
UNIT_BY_ID = {u.uid: u for u in UNITS}
ORDER = {u.uid: i for i, u in enumerate(UNITS)}
ALL_IDS = frozenset(UNIT_BY_ID)
CORE_IDS = frozenset(u.uid for u in UNITS if u.core)
FUNCTION_IDS = frozenset(u.uid for u in FUNCTION_UNITS)

# 选择时使用的组（核心与任务指令不参与选择）
SELECTABLE_GROUPS = ('observe', 'one_stmt', 'scope', 'answer_query', 'empty_scalar', 'modify_answer',
                     'rounds', 'fn_predicted', 'fn_other', 'schema')


def schema_text(task: Task) -> str:
    if _schema_style == 'typed':
        cols = ', '.join(f'{c.name} {c.type}' for c in task.columns)
        return f'The name of this table is {task.table}, and its columns are: {cols}.'
    if _schema_style == 'legacy':
        cols = {c.name: c.type for c in sorted(task.columns, key=lambda c: c.name)}
        return f'Schema: {cols}\nTable: {task.table}'
    return task.schema_text


def unit_text(uid: str, task: Task) -> str:
    if uid == 'schema':
        return schema_text(task)
    if uid in _overrides:
        return _overrides[uid]
    if uid == 'task':
        return task.instruction
    if uid in FUNCTION_IDS and _variant == 'extended':
        return FUNCTION_USAGE[uid[3:]]
    return UNIT_BY_ID[uid].text


def group_units(predicted_functions: Iterable[str]) -> dict[str, frozenset[str]]:
    """本题各选择组包含的单元（函数说明按预测拆成两组）。"""
    predicted = {f'fn:{f}' for f in predicted_functions} & FUNCTION_IDS
    groups: dict[str, set[str]] = {g: set() for g in SELECTABLE_GROUPS}
    for u in UNITS:
        if u.core:
            continue
        if u.uid in FUNCTION_IDS:
            groups['fn_predicted' if u.uid in predicted else 'fn_other'].add(u.uid)
        else:
            groups[u.group].add(u.uid)
    return {g: frozenset(v) for g, v in groups.items()}


def render_rules(unit_ids: Iterable[str], task: Task) -> str:
    """把一组单元（不含任务指令）按固定顺序渲染为文本。函数说明集中在一个标题下。"""
    ids = sorted(set(unit_ids) - {'task'}, key=ORDER.__getitem__)
    lines, fns = [], []
    for uid in ids:
        if uid in FUNCTION_IDS:
            fns.append(unit_text(uid, task))
        elif uid != 'schema':
            lines.append(unit_text(uid, task))
    if fns:
        sep = ', ' if _variant == 'standard' else '\n'
        lines.append(FUNCTION_HEADER + (' ' if _variant == 'standard' else '\n') + sep.join(fns) +
                     ('.' if _variant == 'standard' else ''))
    if 'schema' in ids:
        lines.append(unit_text('schema', task))
    return '\n'.join(lines)


def render_system(unit_ids: Iterable[str], task: Task) -> str:
    return render_rules(set(unit_ids) - {'schema'}, task)


def render_first_user(unit_ids: Iterable[str], task: Task) -> str:
    """第一条用户消息：任务指令；表信息句被选中时按官方格式换行附在其后。"""
    if 'schema' in set(unit_ids):
        return f'{task.instruction}\n{schema_text(task)}'
    return task.instruction
