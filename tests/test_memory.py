"""经验记忆库：抽取、索引、同组排除、状态感知检索与补回、与指令单元的联合调度、条件与实验指纹。
不调用模型、不需要 MySQL。"""
import copy
import json
import random

import pytest

from prism_thesis.prompts import schema_text as _schema
from prism_thesis import cli
from prism_thesis.agent import build_messages, task_features
from prism_thesis.config import ROOT, load_config, make_policy, split_condition
from prism_thesis.data import load_tasks
from prism_thesis.env import ExecResult
from prism_thesis.memory import MemoryStore, Retriever, render_memory
from prism_thesis.memory import build as mbuild
from prism_thesis.memory import signature as sig
from prism_thesis.memory.bm25 import BM25
from prism_thesis.memory.store import EXP_GROUPS, MEMORY_HEADER, Experience, is_exp
from prism_thesis.policies import AblationPolicy, FullPolicy, PhaseSwitchPolicy, PrismPolicy, TurnState
from prism_thesis.prism import knapsack
from prism_thesis.prism.risk import RiskTable, exposed
from prism_thesis.prompts import ALL_IDS, CORE_IDS, SELECTABLE_GROUPS
from prism_thesis.runner import experiment_fingerprint
from prism_thesis.tokens import TokenCounter

COUNTER = TokenCounter()
DEV = load_tasks('development')
CONS = load_tasks('construction')
TASKS = {t.task_id: t for t in load_tasks(include_excluded=True)}
QUERY = next(t for t in DEV if t.is_query)
S0 = TurnState(0, 'initial', None)
ERR = ExecResult(False, "(1054, \"Unknown column 'x' in 'field list'\")", 1054, 'mysql_error')
S_ERR = TurnState(1, 'after_error', ERR, 'SELECT x FROM t WHERE y = 1')
S_OK = TurnState(2, 'after_success', ExecResult(True, '[(1,)]', None), 'SELECT 1')


def op(sql, ok=True, code=None, obs='[(1,)]'):
    return {'action': 'operation', 'content': sql, 'response': f'Action: Operation\n```sql\n{sql}\n```',
            'observation': obs, 'exec': {'ok': ok, 'error_code': code, 'kind': 'ok' if ok else 'mysql_error'}}


def ans(a):
    return {'action': 'answer', 'content': a, 'response': f'Action: Answer\nFinal Answer: {a}'}


def rec(task, rep, success, turns, answer='[(1,)]', cond='nomem'):
    return {'run_key': f'{cond}|{task.task_id}|{rep}', 'condition': cond, 'split': task.split,
            'task_id': task.task_id, 'group_id': task.group_id, 'rep': rep, 'success': success,
            'status': 'answered', 'answer': answer, 'turns': turns}


def sample_records(tasks):
    out = []
    for t in tasks:
        out.append(rec(t, 0, True, [op(f'SELECT a FROM `{t.table}`'), ans('[(1,)]')]))
        out.append(rec(t, 1, True, [op(f'SELECT x FROM `{t.table}` WHERE y = 1', False, 1054,
                                       "(1054, \"Unknown column 'x' in 'field list'\")"),
                                    op(f'SELECT a FROM `{t.table}` WHERE y = 1'), ans('[(1,)]')]))
    return out


def sample_store(tasks=None, insights=()):
    tasks = tasks or DEV[:6]
    recs = sample_records(tasks)
    entries = mbuild.extract_examples(recs, TASKS) + mbuild.extract_repairs(recs, TASKS) + list(insights)
    return MemoryStore(entries)


def insight(eid, text, ttype='any', task=DEV[0]):
    return Experience(eid, 'insight', f'Rule: {text}', text, ttype, (task.task_id,), (task.group_id,), (task.split,))


def table(value=0.0, groups=SELECTABLE_GROUPS + EXP_GROUPS):
    return RiskTable({RiskTable.key(g, p, x): value for g in groups
                      for p in ('initial', 'after_error', 'after_success', 'any') for x in ('query', 'modify', 'any')})


def with_memory(policy, store, **kw):
    policy.memory = Retriever(store, **kw)
    return policy


def start(policy, task=QUERY):
    policy.start(task, task_features(task), random.Random(0))
    return policy


# ---- 索引与签名 ----

def test_bm25_ranks_lexically_closest_first():
    idx = BM25(['average salary of employees', 'insert a new book', 'delete old orders'])
    s = idx.scores('What is the average salary?')
    assert s[0] > s[1] and s[0] > s[2]


def test_error_signature_structure_and_backoff():
    s = sig.signature(ERR, "SELECT COUNT(x) FROM t WHERE name = 'group by' GROUP BY y")
    assert s == '1054|SELECT:WGA'                       # 字符串中的关键词不计入结构
    assert sig.backoff_levels(s) == ('1054|SELECT:WGA', '1054|SELECT', '1054')
    assert sig.signature(ExecResult(True, '[]', None), 'SELECT 1') is None
    assert sig.error_key(ExecResult(False, 'x', None, 'multi_statement')) == 'multi_statement'


# ---- 抽取 ----

def test_extract_examples_and_repairs():
    recs = sample_records(DEV[:2])
    ex = mbuild.extract_examples(recs, TASKS)
    rp = mbuild.extract_repairs(recs, TASKS)
    assert [e.eid for e in ex] == [f'ex-{DEV[0].task_id}', f'ex-{DEV[1].task_id}']
    assert 'SELECT a FROM' in ex[0].text and DEV[0].instruction in ex[0].text and ex[0].source_groups == (DEV[0].group_id,)
    assert len(rp) == 2 and rp[0].error_key == '1054' and rp[0].signature == '1054|SELECT:W'
    assert 'Failed SQL:\nSELECT x' in rp[0].text and 'Corrected SQL:\nSELECT a' in rp[0].text


def test_multi_step_success_keeps_all_operations():
    """审查 P2：先 UPDATE 再 DELETE 的成功轨迹，示例保留两步；出错的操作不写入。"""
    t = DEV[0]
    r = rec(t, 0, True, [op('UPDATE t SET a = 1'), op('SELEC x', False, 1064, 'syntax'), op('DELETE FROM t WHERE a = 2'),
                         ans('done')], answer='done')
    text = mbuild.extract_examples([r], TASKS)[0].text
    assert text.index('UPDATE t SET a = 1') < text.index('DELETE FROM t WHERE a = 2') and 'SELEC x' not in text
    assert text.count('Action: Operation') == 2


def test_failed_runs_give_no_examples_and_long_answers_are_hidden():
    t = DEV[0]
    assert mbuild.extract_examples([rec(t, 0, False, [op('SELECT 1'), ans('[(2,)]')])], TASKS) == []
    long = mbuild.extract_examples([rec(t, 0, True, [op('SELECT 1'), ans('[' + '(1,), ' * 200 + ']')],
                                        answer='[' + '(1,), ' * 200 + ']')], TASKS, max_answer_chars=50)
    assert '(1,), (1,)' not in long[0].text


def test_test_split_never_enters_memory():
    t = next(t for t in TASKS.values() if t.split == 'test')
    with pytest.raises(ValueError):
        mbuild.check_sources([rec(t, 0, True, [op('SELECT 1'), ans('[]')])])
    bad = Experience('ex-1', 'example', 'x', 'x', 'query', (t.task_id,), (t.group_id,), ('test',))
    with pytest.raises(ValueError):
        MemoryStore([bad])


def test_store_roundtrip(tmp_path):
    store = sample_store(insights=[insight('in-0-0', 'Check the result first.')])
    store.save(tmp_path / 'm.json')
    again = MemoryStore.load(tmp_path / 'm.json')
    assert again.entries == store.entries and again.content_hash() == store.content_hash()


# ---- 检索 ----

def test_same_group_is_excluded():
    store = sample_store(DEV[:6])
    r = Retriever(store, k_examples=10, k_repairs=10)
    for t in DEV[:6]:
        got = r.static(t, 'query')
        ids = [u for us in got.values() for u in us]
        srcs = {store.by_id[u.split(':', 2)[2]].source_groups[0] for u in ids}
        assert t.group_id not in srcs and ids
    other = next(t for t in DEV if t.group_id not in {x.group_id for x in DEV[:6]})
    assert len(r.examples(other)) == 6


def test_static_groups_split_top1_and_rest():
    r = Retriever(sample_store(DEV[:6]), k_examples=3)
    g = r.static(QUERY, 'query')
    assert len(g['exp_top1']) == 1 and len(g['exp_rest']) == 2 and len(g['repair']) == 2
    assert 'repair' not in r.static(QUERY, 'query', with_repairs=False)


def test_error_signature_retrieval_backs_off():
    r = Retriever(sample_store(DEV[:6]), k_repairs=2)
    hits = r.for_error(QUERY, ERR, 'SELECT x FROM t WHERE y = 1')
    assert hits and all(e.signature == '1054|SELECT:W' for e in hits)
    looser = r.for_error(QUERY, ERR, 'SELECT x FROM t')               # 结构不同，退到 错误号 + 语句类型
    assert looser and all(e.error_key == '1054' for e in looser)
    assert r.for_error(QUERY, ExecResult(False, 'x', 1146, 'mysql_error'), 'SELECT 1') == []
    assert r.for_error(QUERY, ExecResult(True, '[]', None), 'SELECT 1') == []


def test_insights_filtered_by_task_type():
    store = sample_store(DEV[:2], insights=[insight('in-0-0', 'answer rows as tuples', 'query', DEV[5]),
                                            insight('in-0-1', 'execute the modification first', 'modify', DEV[5])])
    r = Retriever(store)
    assert [e.eid for e in r.insights(QUERY, 'query')] == ['in-0-0']
    assert [e.eid for e in r.insights(DEV[5], 'query')] == []            # 洞见的来源组与当前题相同


# ---- 洞见提炼（离线） ----

def test_distill_insights_records_sources_and_merges_duplicates():
    t1, t2 = DEV[0], DEV[1]
    recs = [rec(t1, 0, False, [op('SELECT 1'), ans('[(9,)]')]), rec(t1, 1, True, [op('SELECT 1'), ans('[(1,)]')]),
            rec(t2, 0, False, [op('SELECT 2'), ans('[(9,)]')]), rec(t2, 1, True, [op('SELECT 2'), ans('[(1,)]')])]
    batches = mbuild.insight_batches(recs, TASKS, batch_size=1)
    assert len(batches) == 2 and 'Failed attempt' in mbuild.insight_messages(batches[0], 3)[1]['content']
    out = mbuild.distill_insights(batches, lambda m: '- Always verify.\nnoise\n* Use exact values.', max_rules=3)
    assert {e.text for e in out} == {'Rule: Always verify.', 'Rule: Use exact values.'}
    assert all(set(e.source_groups) == {t1.group_id, t2.group_id} for e in out)
    assert mbuild.parse_rules('- a\n- b\n- c\n- d', 2) == ['a', 'b']


# ---- 策略与消息组装 ----

def test_full_memory_injection_puts_experience_before_task():
    store = sample_store()
    p = start(with_memory(FullPolicy(), store))
    plan = p.plan(S0)
    exp = {u for u in plan.prefix if is_exp(u)}
    assert ALL_IDS <= plan.prefix and len(exp) == 3 + 2            # 3 条示例 + 2 条修复经验（无洞见）
    msgs, parts = build_messages(QUERY, plan, [], store)
    assert msgs[1]['content'].startswith(MEMORY_HEADER) and msgs[1]['content'].endswith((QUERY.instruction + '\n' + _schema(QUERY)))
    assert parts['memory'] == render_memory(exp, store)
    plain, _ = build_messages(QUERY, start(FullPolicy()).plan(S0), [])
    assert plain[1]['content'] == (QUERY.instruction + '\n' + _schema(QUERY))


def test_ablation_of_experience_group_and_exposure():
    store = sample_store()
    p = start(with_memory(AblationPolicy(('exp_rest',), 'initial'), store))
    plan = p.plan(S0)
    assert plan.info['n_dropped_units'] == 2 and len([u for u in plan.prefix if is_exp(u)]) == 3
    assert p.plan(S_OK).info['n_dropped_units'] == 0
    q = start(with_memory(AblationPolicy(('exp_rest',), 'initial'), store, k_examples=1))
    turn = {'units': sorted(q.plan(S0).units), 'plan_info': q.plan(S0).info}
    assert not exposed({'turns': [turn]})                          # k=1 时“其余示例”为空，不算暴露


def test_prism_schedules_experience_with_instructions():
    store = sample_store(insights=[insight('in-0-0', 'Check results.', 'any', DEV[40])])
    low = start(with_memory(PrismPolicy(table(0.0), 2.0, COUNTER), store)).plan(S0)
    assert low.prefix == CORE_IDS and low.tail == frozenset()
    high = start(with_memory(PrismPolicy(table(50.0), 2.0, COUNTER), store)).plan(S0)
    exp = {u for u in high.tail if is_exp(u)}
    assert len(exp) == 3 + 1 and not any(u.startswith('exp:r') for u in exp)   # 初始状态不检索修复经验
    msgs, parts = build_messages(QUERY, high, [], store)
    assert msgs[1]['content'].startswith(QUERY.instruction)       # 稳定前缀不含经验
    assert parts['tail'].startswith(MEMORY_HEADER) and parts['tail'].endswith(_schema(QUERY))


def test_prism_error_driven_repair_is_forced_and_monotone():
    store = sample_store()
    p = start(with_memory(PrismPolicy(table(0.0), 2.0, COUNTER), store))
    p.plan(S0)
    err = p.plan(S_ERR)
    repairs = {u for u in err.tail if u.startswith('exp:r')}
    assert len(repairs) == 2 and 'schema' in err.tail and err.info['n_repairs'] == 2
    later = p.plan(S_OK)
    assert repairs <= later.tail and 'schema' in later.tail       # 单调保留
    nosig = start(with_memory(PrismPolicy(table(0.0), 2.0, COUNTER, signature=False), store))
    t = nosig.plan(S_ERR).tail
    assert 'schema' in t and not any(u.startswith('exp:r') for u in t)
    norec = start(with_memory(PrismPolicy(table(0.0), 2.0, COUNTER, recovery=False), store))
    assert norec.plan(S_ERR).tail == frozenset()


def test_prism_missing_experience_risk_falls_back_to_full():
    store = sample_store()
    p = start(with_memory(PrismPolicy(table(0.0, SELECTABLE_GROUPS), 2.0, COUNTER), store))
    plan = p.plan(S0)
    assert plan.info['fallback'] == 'unseen_state' and plan.prefix == CORE_IDS and ALL_IDS <= plan.units
    assert len({u for u in plan.tail if is_exp(u)}) == 3


def test_fallback_keeps_stable_prefix():
    """审查 P2：有风险估计的状态与缺项状态之间，系统消息与第一条用户消息不变。"""
    store = sample_store()
    t = table(0.0)
    for k in [k for k in t.entries if k.split('|')[1] == 'after_success']:
        del t.entries[k]
    p = start(with_memory(PrismPolicy(t, 2.0, COUNTER), store))
    history = [{'role': 'assistant', 'content': 'Action: Operation\n```sql\nSELECT 1\n```'},
               {'role': 'user', 'content': '[(1,)]'}]
    m0, _ = build_messages(QUERY, p.plan(S0), [], store)
    fb = p.plan(S_OK)
    assert fb.info['fallback'] == 'unseen_state'
    m1, _ = build_messages(QUERY, fb, history, store)
    assert m1[0] == m0[0] and m1[1] == m0[1]


def test_phase_switch_with_memory():
    store = sample_store(insights=[insight('in-0-0', 'Check results.', 'any', DEV[40])])
    p = start(with_memory(PhaseSwitchPolicy(), store))
    init = {u[:5] for u in p.plan(S0).prefix if is_exp(u)}
    assert init == {'exp:x', 'exp:i'}
    err = {u for u in p.plan(S_ERR).prefix if is_exp(u)}
    assert any(u.startswith('exp:r') for u in err) and ALL_IDS <= p.plan(S_ERR).prefix
    assert {u[:5] for u in p.plan(S_OK).prefix if is_exp(u)} == {'exp:i'}


def test_greedy_is_feasible_and_never_better_than_exact():
    rng = random.Random(1)
    for _ in range(200):
        vals = {f'g{i}': rng.randint(1, 50) for i in range(6)}
        risks = {g: rng.choice([0, 0.5, 1, 1.5, 3]) for g in vals}
        eps = rng.choice([0.5, 1, 2, 4])
        g, e = knapsack.greedy(vals, risks, eps), knapsack.solve(vals, risks, eps)
        assert sum(knapsack.weight(risks[x]) for x in g) <= eps / 0.5 + 1e-9
        assert sum(vals[x] for x in g) <= sum(vals[x] for x in e)


# ---- 条件、指纹与命令行 ----

def mem_cfg(tmp_path, store=None):
    cfg = copy.deepcopy(load_config(ROOT / 'configs' / 'default.json'))
    cfg['memory']['store'] = str(tmp_path / 'memory.json')
    store = store or sample_store()
    store.meta['rehearsal'] = False
    store.save(tmp_path / 'memory.json')
    return cfg


def test_baseline_parameters_in_identity_with_k_modifier(tmp_path):
    """审查 P1：random_drop#k=1、llmlingua2#k=1 的指纹也包含基线参数。"""
    cfg = mem_cfg(tmp_path)
    for cond in ('random_drop', 'random_drop#k=1', 'llmlingua2#k=1'):
        a = experiment_fingerprint(cfg, cond)
        c2 = copy.deepcopy(cfg)
        c2['baselines']['random_k'] += 1
        c2['baselines']['llmlingua2']['rate'] = 0.5
        assert experiment_fingerprint(c2, cond) != a, cond


def test_rehearsal_memory_refused_for_formal_runs(tmp_path):
    """审查 P1：演练记忆库或来源不明的记忆库不能用于正式（付费）运行；演练运行仍可使用。"""
    from prism_thesis.config import RehearsalArtifact
    from prism_thesis.runner import run_experiment
    cfg = mem_cfg(tmp_path)
    for flag in (True, None):
        st = MemoryStore.load(tmp_path / 'memory.json')
        st.meta['rehearsal'] = flag
        st.save(tmp_path / 'memory.json')
        with pytest.raises(RehearsalArtifact):
            make_policy('full', cfg, COUNTER)
        with pytest.raises(RehearsalArtifact):
            run_experiment(cfg, DEV[:1], ['full'], 1, tmp_path / 'r.jsonl', lambda t: None, paid=True, workers=1,
                           log=lambda m: None)
        assert make_policy('full', cfg, COUNTER, allow_rehearsal_memory=True).memory is not None
        assert make_policy('nomem', cfg, COUNTER).memory is None


def test_formal_build_refuses_rehearsal_insights(tmp_path, capsys):
    """审查 P1：演练洞见不能进入正式记忆库；演练构建的记忆库带演练标记。"""
    from prism_thesis.memory.build import check_sources  # noqa: F401
    cfg_path = tmp_path / 'cfg.json'
    cfg_path.write_text(json.dumps(mem_cfg(tmp_path)), encoding='utf-8')
    ins = tmp_path / 'ins.json'
    for meta in ({'rehearsal': True}, {}):
        ins.write_text(json.dumps({'meta': meta, 'insights': []}), encoding='utf-8')
        assert cli._insight_file(ins, rehearsal=True) == []
        with pytest.raises(ValueError):
            cli._insight_file(ins, rehearsal=False)
    ins.write_text(json.dumps({'meta': {'rehearsal': False}, 'insights': []}), encoding='utf-8')
    assert cli._insight_file(ins, rehearsal=False) == []
    runs = tmp_path / 'runs.jsonl'
    _write(runs, sample_records(DEV[:2]))
    out = tmp_path / 'store.json'
    assert cli.main(['--config', str(cfg_path), 'build-memory', '--runs', str(runs), '--out', str(out),
                     '--rehearsal']) == 0
    assert MemoryStore.load(out).meta['rehearsal'] is True


def test_condition_modifiers():
    assert split_condition('prism@4#k=5') == ('prism@4', {'k_examples': 5})
    assert split_condition('full') == ('full', {})
    with pytest.raises(ValueError):
        split_condition('ablate:schema@initial#k=1')
    with pytest.raises(ValueError):
        split_condition('full#n=2')


def test_make_policy_memory_conditions(tmp_path):
    cfg = mem_cfg(tmp_path)
    assert make_policy('nomem', cfg, COUNTER).memory is None
    full = make_policy('full', cfg, COUNTER)
    assert full.memory is not None and full.memory.k_examples == 3
    k1 = make_policy('full#k=1', cfg, COUNTER)
    assert k1.memory.k_examples == 1 and k1.name == 'full#k=1'
    cfg['memory']['store'] = str(tmp_path / 'missing.json')
    with pytest.raises(FileNotFoundError):
        make_policy('full', cfg, COUNTER)
    assert make_policy('nomem', cfg, COUNTER).name == 'nomem'


def test_memory_content_is_part_of_identity(tmp_path):
    cfg = mem_cfg(tmp_path)
    full, nomem, k5 = (experiment_fingerprint(cfg, c) for c in ('full', 'nomem', 'full#k=5'))
    assert len({full, nomem, k5}) == 3
    sample_store(DEV[:3]).save(tmp_path / 'memory.json')
    assert experiment_fingerprint(cfg, 'full') != full
    assert experiment_fingerprint(cfg, 'nomem') == nomem          # 无记忆条件不依赖记忆库


def _write(path, recs):
    path.write_text(''.join(json.dumps(r) + '\n' for r in recs), encoding='utf-8')


def test_cli_build_memory_rehearsal_and_test_refusal(tmp_path, capsys):
    cfg_path = tmp_path / 'cfg.json'
    cfg = mem_cfg(tmp_path)
    cfg_path.write_text(json.dumps(cfg), encoding='utf-8')
    runs = tmp_path / 'runs.jsonl'
    _write(runs, sample_records(DEV[:3] + CONS[:3]))
    ins = tmp_path / 'ins.json'
    assert cli.main(['--config', str(cfg_path), 'distill-insights', '--runs', str(runs), '--out', str(ins)]) == 2
    out = tmp_path / 'store.json'
    assert cli.main(['--config', str(cfg_path), 'build-memory', '--runs', str(runs), '--out', str(out),
                     '--rehearsal']) == 0
    store = MemoryStore.load(out)
    assert store.counts() == {'example': 6, 'repair': 6, 'insight': 0}
    assert cli.main(['--config', str(cfg_path), 'build-memory', '--runs', str(runs), '--out', str(out)]) == 1
    test_task = next(t for t in TASKS.values() if t.split == 'test')
    _write(runs, sample_records(DEV[:2] + [test_task]))
    assert cli.main(['--config', str(cfg_path), 'build-memory', '--runs', str(runs), '--out', str(out),
                     '--rehearsal']) == 1
    assert '测试集' in capsys.readouterr().out


def test_distill_retries_rate_limits_only():
    from prism_thesis.llm import LLMError
    calls, waits = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise LLMError("HTTP 429: b'temporarily rate-limited upstream'")
        return 'ok'
    assert cli.retry_rate_limited(flaky, 5, 60, sleep=waits.append) == 'ok' and waits == [60, 60]
    with pytest.raises(LLMError):
        cli.retry_rate_limited(lambda: (_ for _ in ()).throw(LLMError('HTTP 400: bad')), 5, 60, sleep=waits.append)
    assert waits == [60, 60]
    calls.clear()
    with pytest.raises(LLMError):
        cli.retry_rate_limited(flaky, 1, 1, sleep=waits.append)
