"""解析智能体的回复，规则与官方 db_bench 的 _parse_agent_response 相同，只有一处差别（见下）。

- 先找第一个 "Action: (Operation|Answer)"，找不到即为无效动作；
- Operation：取第一个 ```sql 代码块，去掉首尾空白，保留内部换行。
  官方把换行替换为空格，这会改变 SQL 语义：“-- ”与 # 行注释会吞掉下一行（如 WHERE 条件），
  字符串中的换行也会被改写。本研究保留原样，所有条件相同；

- Answer：取 "\nFinal Answer:" 之后到行尾的文本。

first_action() 在解析之前对回复做单动作规范化（配置项 agent.first_action_only，所有实验条件统一使用）：
1. 只保留回复中第一个完整动作（相当于在动作结束处停止生成）。
   Llama-3.1-8B 常在 SQL 操作之后自行编造结果并给出答案；
2. 操作动作的标签统一为 Operation。Llama-3.1-8B 常把标签写成 SQL 动词
   （"Action: Query" / "Action: Select" / "Action: Update"），后面跟着合规的 ```sql 代码块；
   这类回复改写为 "Action: Operation" 加该代码块。代码块的语言标记不区分大小写（sql / mysql）；
3. 没有代码块、动作行之后直接写 SQL 的回复（Llama-3.1-8B 常见），取这条裸 SQL：
   从 SQL 关键字开始，到第一个以分号结尾的行、空行或模型自行编造的结果表格为止。
这两步只对官方解析判为无效的回复生效；已经合规的回复只做第 1 步。
既无可识别的 SQL、也没有 Final Answer 行的回复保持原样，由 parse() 判为无效动作。
原始回复与截掉的字符数记录在日志中。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

ACTION = re.compile(r'Action: (Operation|Answer)')
SQL_BLOCK = re.compile(r'```sql\n([\s\S]*?)\n```')
FINAL_ANSWER = re.compile(r'\nFinal Answer:(.*)')


@dataclass(frozen=True)
class Parsed:
    kind: str                 # operation / answer / invalid
    content: Optional[str]    # SQL 或答案文本
    reason: Optional[str] = None


def parse(response: str) -> Parsed:
    m = ACTION.search(response)
    if m is None:
        return Parsed('invalid', None, 'no_action')
    if m.group(1) == 'Operation':
        sql = SQL_BLOCK.search(response)
        if sql is None:
            return Parsed('invalid', None, 'no_sql_block')
        return Parsed('operation', sql.group(1).strip())
    ans = FINAL_ANSWER.search(response)
    if ans is None:
        return Parsed('invalid', None, 'no_final_answer')
    return Parsed('answer', ans.group(1).strip())


_ANY_ACTION = re.compile(r'^Action:[ \t]*([A-Za-z][A-Za-z _-]*?)[ \t]*$', re.MULTILINE)
_LOOSE_BLOCK = re.compile(r'```(?:sql|mysql)[ \t]*\n([\s\S]*?)\n[ \t]*```', re.IGNORECASE)
_FINAL_LINE = re.compile(r'^Final Answer:[^\r\n]*', re.MULTILINE)


def first_action(response: str) -> str:
    """单动作规范化，以回复中第一个动作行为准。
    第一个动作行的标签不是 Operation/Answer（如 "Action: Query"）时直接按该动作改写：
    官方规则会跳过这种标签，去取后面模型自行编造的 "Action: Answer"，把猜测的答案当成最终答案。
    第一个动作行是 Operation/Answer 时与官方解析一致（只做截断）；官方解析无效时再尝试改写。"""
    first = _ACTION_LINE.search(response)
    if first is not None and first.group(1).strip() not in ('Operation', 'Answer'):
        return _relabel(response)
    cut = _truncate(response)
    if parse(cut).kind != 'invalid':
        return cut
    return _relabel(response)


def _truncate(response: str) -> str:
    """截断到第一个完整动作的结尾。"""
    m = ACTION.search(response)
    if m is None:
        return response
    tail = (SQL_BLOCK if m.group(1) == 'Operation' else FINAL_ANSWER).search(response)
    if tail is None:
        return response
    return response[:max(m.end(), tail.end())]


_ACTION_LINE = re.compile(r'^Action:[ \t]*([^\r\n]*)$', re.MULTILINE)
_SQL_START = re.compile(r'(SELECT|INSERT|UPDATE|DELETE|REPLACE|WITH)\b', re.IGNORECASE)
_BARE_STOP = re.compile(r'^\s*$|^\s*[+|]|^\s*```|^(Final Answer|Result|Output|Observation)\b', re.IGNORECASE)


def _bare_sql(body: str) -> str | None:
    """没有代码块时，取紧跟在动作行之后的裸 SQL：从 SQL 关键字开始，
    到第一个以分号结尾的行、空行、结果表格（+--- 或 |）或说明行为止。"""
    body = body.lstrip('\r\n')
    if not _SQL_START.match(body.lstrip()):
        return None
    lines = []
    for line in body.splitlines():
        if lines and _BARE_STOP.match(line):
            break
        lines.append(line.rstrip())
        if line.rstrip().endswith(';'):
            break
    sql = '\n'.join(lines).strip()
    return sql or None


def _relabel(response: str) -> str:
    """官方解析无效时的规范化：
    - 标签写成 SQL 动词（Action: Query / SELECT / Update）或 Operation，后跟 ```sql 代码块 → 取该代码块；
    - 没有代码块、动作行后直接是 SQL（或 SQL 就写在动作行上）→ 取这条裸 SQL（见 _bare_sql）；
    都改写为标准的 "Action: Operation" 加代码块。只看第一个动作。"""
    m = _ACTION_LINE.search(response)
    if m is None:
        return response
    label = m.group(1).strip()
    rest = response[m.end():]
    nxt = _ACTION_LINE.search(rest)
    scope = rest[:nxt.start()] if nxt else rest
    if label.lower() == 'answer':
        final = _FINAL_LINE.search(scope)
        if final is None:
            return response
        return 'Action: Answer\n' + final.group(0).rstrip()
    if re.fullmatch(r'[A-Za-z][A-Za-z _-]*', label) or not label:
        body = scope
    elif _SQL_START.match(label):          # SQL 直接写在动作行上
        body = label + scope
    else:
        return response
    block = _LOOSE_BLOCK.search(body)
    sql = block.group(1).strip() if block else _bare_sql(body)
    if not sql:
        return response
    return 'Action: Operation\n```sql\n' + sql + '\n```'
