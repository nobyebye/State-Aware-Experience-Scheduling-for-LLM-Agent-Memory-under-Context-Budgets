import random

from prism_thesis.agent import build_messages, task_features
from prism_thesis.data import load_tasks
from prism_thesis.env import ExecResult
from prism_thesis.policies import (AblationPolicy, FullPolicy, PhaseSwitchPolicy, PrismPolicy, RandomDropPolicy,
                                   StaticPolicy, TurnState)
from prism_thesis.prism.risk import RiskTable
from prism_thesis.prompts import ALL_IDS, CORE_IDS, SELECTABLE_GROUPS
from prism_thesis.tokens import TokenCounter

COUNTER = TokenCounter()
TASKS = load_tasks('development')
QUERY = next(t for t in TASKS if t.is_query)
MODIFY = next(t for t in TASKS if not t.is_query)
S0 = TurnState(0, 'initial', None)
S_OK = TurnState(1, 'after_success', ExecResult(True, '[(1,)]', None))
S_ERR = TurnState(1, 'after_error', ExecResult(False, 'Error 1054: Unknown column', 1054, 'mysql_error'))


def start(policy, task):
    policy.start(task, task_features(task), random.Random(0))
    return policy


def table(value=0.0, phases=('initial', 'after_error', 'after_success', 'any')):
    return RiskTable({RiskTable.key(g, p, x): value for g in SELECTABLE_GROUPS for p in phases
                      for x in ('query', 'modify', 'any')})


def test_full_prompt_matches_official_task_text():
    msgs, _ = build_messages(QUERY, start(FullPolicy(), QUERY).plan(S0), [])
    assert msgs[1]['content'] == QUERY.official_prompt
    assert 'Action: Operation' in msgs[0]['content']


def test_phase_switch_rules():
    p = start(PhaseSwitchPolicy(), QUERY)
    f = task_features(QUERY)
    assert p.plan(S_ERR).prefix == ALL_IDS
    assert not (p.plan(S0).prefix & f.groups['fn_other'])
    ok = p.plan(S_OK).prefix
    assert 'schema' not in ok and 'one_stmt' not in ok and 'answer_query' in ok
    m = start(PhaseSwitchPolicy(), MODIFY)
    assert 'answer_query' not in m.plan(S0).prefix


def test_ablation_only_in_its_phase():
    p = start(AblationPolicy(('schema',), 'after_success'), QUERY)
    assert 'schema' in p.plan(S0).prefix and 'schema' not in p.plan(S_OK).prefix


def test_random_drop_reproducible():
    a = start(RandomDropPolicy(3), QUERY).plan(S0)
    b = start(RandomDropPolicy(3), QUERY).plan(S0)
    assert a == b and CORE_IDS <= a.prefix and len(ALL_IDS - a.prefix) > 0


def test_prism_zero_risk_drops_all_and_keeps_core():
    p = start(PrismPolicy(table(0.0), 2.0, COUNTER), QUERY)
    plan = p.plan(S0)
    assert plan.prefix == CORE_IDS and plan.tail == frozenset()


def test_prism_high_risk_keeps_everything_in_tail():
    p = start(PrismPolicy(table(50.0), 2.0, COUNTER), QUERY)
    plan = p.plan(S0)
    assert plan.prefix | plan.tail == ALL_IDS and plan.tail and not plan.tail_in_system
    msgs, parts = build_messages(QUERY, plan, [])
    assert msgs[-1]['content'].startswith(QUERY.instruction) and msgs[-1]['content'].endswith(QUERY.schema_text)


def test_prism_unseen_state_sends_all():
    p = start(PrismPolicy(table(0.0, phases=('initial',)), 2.0, COUNTER), QUERY)
    assert p.plan(S0).prefix == CORE_IDS
    plan = p.plan(S_OK)
    assert plan.units == ALL_IDS and plan.info['fallback'] == 'unseen_state'


def test_prism_recovery_is_monotone():
    p = start(PrismPolicy(table(0.0), 2.0, COUNTER), QUERY)
    p.plan(S0)
    assert 'schema' in p.plan(S_ERR).tail
    assert 'schema' in p.plan(TurnState(2, 'after_success', ExecResult(True, '[]', None))).tail
    q = start(PrismPolicy(table(0.0), 2.0, COUNTER, recovery=False), QUERY)
    assert 'schema' not in q.plan(S_ERR).units


def test_tail_not_written_to_history_and_cache_prefix_stable():
    p = start(PrismPolicy(table(50.0), 2.0, COUNTER), QUERY)
    history = [{'role': 'assistant', 'content': 'Action: Operation\n```sql\nSELECT 1\n```'},
               {'role': 'user', 'content': '[(1,)]'}]
    m1, _ = build_messages(QUERY, p.plan(S0), [])
    m2, _ = build_messages(QUERY, p.plan(S_OK), history)
    assert m1[0] == m2[0]                                   # 系统消息不变
    assert m2[1]['content'] == QUERY.instruction            # 历史中的首条用户消息不含临时块
    assert m1[1]['content'].startswith(m2[1]['content'])
    assert history[-1]['content'] == '[(1,)]'


def test_tail_in_system_ablation():
    plan = start(PrismPolicy(table(50.0), 2.0, COUNTER, tail_in_system=True), QUERY).plan(S0)
    msgs, _ = build_messages(QUERY, plan, [])
    assert msgs[1]['content'] == QUERY.instruction and QUERY.schema_text in msgs[0]['content']


def test_static_fixed_across_turns():
    p = start(StaticPolicy(table(0.0), 2.0, COUNTER), QUERY)
    assert p.plan(S0) == p.plan(S_OK) == p.plan(S_ERR)


def test_merged_groups_are_selected_together():
    t = table(0.0)
    t.merged = [('answer_query', 'empty_scalar')]
    for ph in ('initial', 'after_error', 'after_success'):
        for x in ('query', 'modify'):
            t.entries[RiskTable.key('answer_query+empty_scalar', ph, x)] = 50.0
    plan = start(PrismPolicy(t, 2.0, COUNTER), QUERY).plan(S0)
    assert {'answer_query', 'empty_scalar'} <= plan.tail
