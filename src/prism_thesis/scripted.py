"""不调用模型的脚本化智能体，用于流程演练与测试（不产生费用）。

gold：第一轮执行参考 SQL，第二轮把观察到的结果原样作为答案（修改题答 done）。
gold_after_error：第一轮先执行一条会报“未知列”的 SQL，之后同 gold，用于检验出错后的状态与补回。
它们不读取提示内容，因此各条件的成功率都应为 100%；用途是检验环境、判分、日志与各策略的组装。
"""
from __future__ import annotations

from .data import Task
from .llm import ScriptedLLM


def _op(sql: str) -> str:
    return f'I will run the query.\nAction: Operation\n```sql\n{sql}\n```'


def _answer(messages: list[dict], task: Task) -> str:
    obs = messages[-1]['content'].split('\n\n')[0]   # 去掉临时块
    final = obs if task.answer_direct is not None else 'done'
    return f'Action: Answer\nFinal Answer: {final}'


def make(kind: str, task: Task) -> ScriptedLLM:
    def gold(messages):
        n_assistant = sum(m['role'] == 'assistant' for m in messages)
        if n_assistant == 0:
            return _op(task.reference_sql)
        return _answer(messages, task)

    def gold_after_error(messages):
        n_assistant = sum(m['role'] == 'assistant' for m in messages)
        if n_assistant == 0:
            return _op(f'SELECT no_such_column FROM `{task.table}`')
        if n_assistant == 1:
            return _op(task.reference_sql)
        return _answer(messages, task)

    fns = {'gold': gold, 'gold_after_error': gold_after_error}
    return ScriptedLLM(fns[kind], model=f'scripted-{kind}')
