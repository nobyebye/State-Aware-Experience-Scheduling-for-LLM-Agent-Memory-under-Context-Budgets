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


ACTION_UNITS = [
    Unit('role', 'core', "You are an agent that operates a MySQL database to complete the user's task. "
         'In every response, output exactly one action: either a database operation or the final answer.',
         core=True),
    Unit('op_format', 'core', 'To operate the database, output exactly the following; the first line is always '
         'the literal text "Action: Operation", and the SQL must be inside the ```sql code block:\n'
         'Action: Operation\n```sql\nYOUR_SQL\n```\n'
         'For example:\nAction: Operation\n```sql\nSELECT name FROM employees WHERE age > 30;\n```\n'
         'The SQL will be executed and you will receive the raw MySQL result. '
         'Do not write the result yourself.', core=True),
    Unit('answer_action', 'core', 'When you are done, output exactly (the first line is always "Action: Answer"):\n'
         'Action: Answer\nFinal Answer: YOUR_ANSWER_ON_ONE_LINE', core=True),
]

RULE_UNITS = [
    Unit('observe', 'observe', 'Base the final answer only on results you have actually observed; '
         'do not guess values or claim success without executing the operation.'),
    Unit('one_stmt', 'one_stmt', 'Write exactly one SQL statement per operation.'),
    Unit('scope', 'scope', 'Only the given table is accessible. You may read and modify its rows, '
         'but do not create, drop or alter tables or databases.'),
    Unit('answer_query', 'answer_query', 'If the task asks a question about the data, run a SELECT and give the '
         'Final Answer as the result rows: a list of tuples on one line, with values exactly as returned by MySQL, '
         "e.g. [(1, 'Alice'), (2, 'Bob')]. Never insert, update or delete rows for such a task."),
    Unit('empty_scalar', 'empty_scalar', 'If the result is empty, answer []. '
         'If the result is a single value, answer [(value,)].'),
    Unit('modify_answer', 'modify_answer', 'Only if the task itself asks to insert, update or delete rows: '
         'execute exactly that modification; the table is checked afterwards, so the Final Answer can then be a '
         'short confirmation such as done.'),
    Unit('rounds', 'rounds', 'You have at most three responses in total.'),
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


def unit_text(uid: str, task: Task) -> str:
    if uid == 'schema':
        return task.schema_text
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
        return task.official_prompt
    return task.instruction
