"""智能体主循环。所有实验条件共用这一循环，只在策略（每轮发送哪些单元、放在哪里）上不同。

每轮：策略给出 Plan → 组装消息 → 调用模型 → 截取第一个动作 → 解析 →
执行 SQL（观察写入历史）或结束。最多 max_rounds 轮。
判分（与官方一致）：修改题以结束时的表哈希为准，与是否给出答案无关；
查询题需要给出 Final Answer，并按官方规则与标准答案比较。
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Optional

from . import features as feat
from .data import Task
from .env import ExecResult, TaskSession
from .evaluate import match_direct
from .parser import first_action, parse
from .policies.base import Plan, Policy, TaskFeatures, TurnState
from .memory.store import is_exp, render_memory
from .prompts import group_units, render_first_user, render_rules, render_system
from .tokens import PER_MESSAGE, PER_REQUEST, TokenCounter

TAIL_SEPARATOR = '\n\n'
MEMORY_SEPARATOR = '\n\n'


def task_features(task: Task) -> TaskFeatures:
    fns = feat.predict_functions(task.instruction)
    return TaskFeatures(feat.task_type(task.instruction), fns, group_units(fns))


def build_messages(task: Task, plan: Plan, history: list[dict], store=None) -> tuple[list[dict], dict]:
    """返回 (消息列表, 各部分文本)。history 为 [{'role','content'}]，不含临时块。
    经验单元：放在 prefix 中的渲染为经验块，置于第一条用户消息的任务指令之前；放在 tail 中的渲染在临时块开头。"""
    prefix = frozenset(u for u in plan.prefix if not is_exp(u))
    tail_ids = frozenset(u for u in plan.tail if not is_exp(u))
    system = plan.system_text if plan.system_text is not None else render_system(prefix, task)
    if plan.schema_text is not None:
        first_user = f'{task.instruction}\n{plan.schema_text}'
    else:
        first_user = render_first_user(prefix, task)
    memory = plan.memory_text if plan.memory_text is not None else render_memory(plan.prefix, store)
    if memory:
        first_user = f'{memory}{MEMORY_SEPARATOR}{first_user}'
    tail_memory = render_memory(plan.tail, store)
    tail = MEMORY_SEPARATOR.join(x for x in (tail_memory, render_rules(tail_ids, task) if tail_ids else '') if x)
    if tail and plan.tail_in_system:
        system = f'{system}\n{tail}' if system else tail
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': first_user}]
    messages += [dict(m) for m in history]
    if tail and not plan.tail_in_system:
        messages[-1]['content'] += TAIL_SEPARATOR + tail
    parts = {'system': system, 'first_user': first_user, 'tail': tail, 'memory': memory,
             'tail_memory': tail_memory}
    return messages, parts


@dataclass
class EpisodeResult:
    success: bool
    status: str                       # answered / invalid_action / round_limit
    answer: Optional[str]
    turns: list[dict] = field(default_factory=list)
    input_tokens: int = 0             # 本地计数的累计输入 token
    output_tokens: int = 0            # 截取后的输出 token（写入历史的部分）
    memory_tokens: int = 0            # 累计输入中经验块的 token（本地计数）
    raw_output_tokens: int = 0        # 模型实际生成的 token（本地计数）
    api_prompt_tokens: Optional[int] = None
    api_completion_tokens: Optional[int] = None
    api_cached_tokens: Optional[int] = None
    final_md5: Optional[str] = None


def run_episode(task: Task, policy: Policy, llm, session: TaskSession, counter: TokenCounter,
                max_rounds: int = 3, first_action_only: bool = True,
                rng: Optional[random.Random] = None) -> EpisodeResult:
    features = task_features(task)
    policy.start(task, features, rng or random.Random(0))
    history: list[dict] = []
    last_exec: Optional[ExecResult] = None
    last_sql: Optional[str] = None
    store = getattr(policy, 'store', None)
    status, answer = 'round_limit', None
    res = EpisodeResult(False, status, None)
    api_p = api_c = api_cache = 0
    api_complete = True
    for turn in range(max_rounds):
        state = TurnState(turn, feat.phase(turn, last_exec), last_exec, last_sql)
        plan = policy.plan(state)
        messages, parts = build_messages(task, plan, history, store)
        n_in = counter.count_messages(messages)
        n_mem = counter.count(parts['memory']) + counter.count(parts['tail_memory'])
        started = time.monotonic()
        resp = llm.chat(messages)
        latency = time.monotonic() - started
        raw = resp.content
        content = first_action(raw) if first_action_only else raw
        parsed = parse(content)
        rec = {
            'turn': turn, 'phase': state.phase, 'units': sorted(plan.units), 'tail': sorted(plan.tail),
            'experiences': sorted(u for u in plan.units if is_exp(u)),
            'plan_info': plan.info,
            'tokens': {'input': n_in, 'system': counter.count(parts['system']),
                       'first_user': counter.count(parts['first_user']), 'tail': counter.count(parts['tail']),
                       'history': sum(counter.count(m['content']) for m in history), 'memory': n_mem,
                       'format': PER_MESSAGE * len(messages) + PER_REQUEST},
            'api': {'prompt_tokens': resp.prompt_tokens, 'completion_tokens': resp.completion_tokens,
                    'cached_tokens': resp.cached_tokens, 'model': resp.model, 'provider': resp.provider,
                    **resp.raw},
            'latency_s': round(latency, 3),
            'raw_response': raw, 'response': content, 'trimmed_chars': len(raw) - len(content),
            'action': parsed.kind, 'content': parsed.content, 'parse_reason': parsed.reason,
        }
        res.input_tokens += n_in
        res.memory_tokens += n_mem
        res.output_tokens += counter.count(content)
        res.raw_output_tokens += counter.count(raw)
        if resp.prompt_tokens is None:
            api_complete = False
        else:
            api_p += resp.prompt_tokens
            api_c += resp.completion_tokens or 0
            api_cache += resp.cached_tokens or 0
        history.append({'role': 'assistant', 'content': content})
        if parsed.kind == 'invalid':
            status = 'invalid_action'
            res.turns.append(rec)
            break
        if parsed.kind == 'answer':
            status, answer = 'answered', parsed.content
            res.turns.append(rec)
            break
        last_sql = parsed.content
        last_exec = session.execute(parsed.content)
        rec['observation'] = last_exec.text
        rec['exec'] = {'ok': last_exec.ok, 'error_code': last_exec.error_code, 'kind': last_exec.kind}
        history.append({'role': 'user', 'content': last_exec.text})
        res.turns.append(rec)
    res.status, res.answer = status, answer
    if task.answer_md5 is not None:
        res.final_md5 = session.table_md5()
        res.success = res.final_md5 == task.answer_md5
    else:
        res.success = status == 'answered' and match_direct(answer, task.answer_direct)
    if api_complete:
        res.api_prompt_tokens, res.api_completion_tokens, res.api_cached_tokens = api_p, api_c, api_cache
    return res
