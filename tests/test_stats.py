import pytest
import numpy as np

from prism_thesis import stats
from prism_thesis.prism.risk import estimate
from prism_thesis.prompts import ALL_IDS

FULL_TURN = {'units': sorted(ALL_IDS)}
DROP_TURN = {'units': sorted(ALL_IDS - {'schema'})}


def recs(cond, succ, turns=None):
    if turns is None:
        turns = [FULL_TURN] if cond == 'full' else [DROP_TURN]
    return [{'condition': cond, 'task_id': t, 'success': s, 'turns': turns} for t, s in succ.items()]


def test_bootstrap_point_and_interval():
    vals = {i: float(i % 2) for i in range(100)}
    groups = {i: f'g{i // 2}' for i in range(100)}
    iv = stats.cluster_bootstrap(vals, groups, b=2000)
    assert iv.estimate == 0.5 and iv.low <= 0.5 <= iv.high and iv.n_groups == 50
    g = stats.cluster_bootstrap(vals, groups, b=2000, weighting='group')
    assert g.estimate == 0.5 and g.high - g.low < 1e-12   # 每组均值都是 0.5


def test_fixed_sequence_stops_at_first_failure():
    rng = np.random.default_rng(0)
    groups = {i: f'g{i}' for i in range(200)}
    base = {i: 1.0 for i in range(200)}
    records = recs('full', base)
    records += recs('prism@1', base)
    records += recs('prism@2', {i: float(rng.random() > 0.3) for i in range(200)})
    records += recs('prism@4', base)
    best, steps = stats.fixed_sequence(records, [(1, 'prism@1'), (2, 'prism@2'), (4, 'prism@4')], 'full',
                                       groups, 0.05, b=2000)
    assert best == 1 and len(steps) == 2 and not steps[1].passed


def test_risk_estimate_and_merge():
    groups = {i: f'g{i}' for i in range(300)}
    types = {i: 'query' for i in range(300)}
    rng = np.random.default_rng(1)
    full = {i: 1.0 for i in range(300)}
    a = {i: float(rng.random() > 0.02) for i in range(300)}
    b = {i: float(rng.random() > 0.02) for i in range(300)}
    ab = {i: float(rng.random() > 0.5) for i in range(300)}        # 联合省略损失远大于单独之和
    r = (recs('full', full) + recs('ablate:a@initial', a) + recs('ablate:b@initial', b) +
         recs('ablate:a+b@initial', ab))
    t = estimate(r, groups, types, b=2000, min_exposed_groups=10)
    assert 0 < t.lookup('a', 'initial', 'query') < 10
    assert ('a', 'b') in t.merged and t.lookup('a+b', 'initial', 'query') > 40


def test_unexposed_state_is_not_zero_risk():
    """消融条件从未真正生效（如从未进入出错后状态，或该组在本题为空）时不得写入零风险。"""
    groups = {i: f'g{i}' for i in range(10)}
    types = {i: 'query' for i in range(10)}
    ok = {i: 1.0 for i in range(10)}
    r = recs('full', ok) + recs('ablate:schema@after_error', ok, turns=[FULL_TURN])
    t = estimate(r, groups, types, b=100, min_exposed_groups=1)
    assert t.lookup('schema', 'after_error', 'query') is None
    assert t.meta['coverage']['schema|after_error|query']['exposed_groups'] == 0


def test_exposure_threshold():
    groups = {i: f'g{i // 2}' for i in range(20)}          # 10 个题目组
    types = {i: 'query' for i in range(20)}
    ok = {i: 1.0 for i in range(20)}
    r = recs('full', ok)
    r += [x for i, x in enumerate(recs('ablate:schema@after_error', ok)) if i < 9]          # 5 组暴露
    r += [x for i, x in enumerate(recs('ablate:schema@after_error', ok, turns=[FULL_TURN])) if i >= 9]
    assert estimate(r, groups, types, b=100, min_exposed_groups=6).lookup('schema', 'after_error', 'query') is None
    assert estimate(r, groups, types, b=100, min_exposed_groups=5).lookup('schema', 'after_error', 'query') == 0.0


def test_prism_falls_back_when_entry_missing_for_uncovered_state():
    from prism_thesis.policies import PrismPolicy
    from test_policies import QUERY, S_ERR, start, COUNTER
    groups = {i: f'g{i}' for i in range(10)}
    types = {i: 'query' for i in range(10)}
    ok = {i: 1.0 for i in range(10)}
    from prism_thesis.prompts import SELECTABLE_GROUPS
    r = recs('full', ok)
    for g in SELECTABLE_GROUPS:
        r += recs(f'ablate:{g}@initial', ok) + recs(f'ablate:{g}@after_error', ok, turns=[FULL_TURN])
    t = estimate(r, groups, types, b=100, min_exposed_groups=1)
    plan = start(PrismPolicy(t, 2.0, COUNTER), QUERY).plan(S_ERR)
    assert plan.units == ALL_IDS and plan.info['fallback'] == 'unseen_state'


# ---- 第三轮：R1 暴露只在实际配对的题目上统计 ----

def test_coverage_counted_on_paired_tasks_only():
    """审查复现：full 只有 task 0（未暴露），消融有 task 0–10，后 10 题暴露在 10 个组。"""
    groups = {i: f'g{i}' for i in range(11)}
    types = {i: 'query' for i in range(11)}
    r = recs('full', {0: 1.0})
    r += recs('ablate:schema@after_error', {0: 1.0}, turns=[FULL_TURN])
    r += recs('ablate:schema@after_error', {i: 1.0 for i in range(1, 11)})
    t = estimate(r, groups, types, b=100, min_exposed_groups=10)
    assert t.lookup('schema', 'after_error', 'query') is None
    assert t.meta['coverage']['schema|after_error|query']['exposed_groups'] == 0
    r += recs('full', {i: 1.0 for i in range(1, 11)})            # 补齐对照后才计入
    t = estimate(r, groups, types, b=100, min_exposed_groups=10)
    assert t.lookup('schema', 'after_error', 'query') == 0.0


def test_untestable_interaction_removes_single_entries():
    groups = {i: f'g{i}' for i in range(12)}
    types = {i: 'query' for i in range(12)}
    ok = {i: 1.0 for i in range(12)}
    r = (recs('full', ok) + recs('ablate:a@initial', ok) + recs('ablate:b@initial', ok)
         + recs('ablate:a+b@initial', {i: 1.0 for i in range(3)}))         # 联合消融只覆盖 3 组
    t = estimate(r, groups, types, b=100, min_exposed_groups=10)
    assert t.lookup('a', 'initial', 'query') is None and t.lookup('b', 'initial', 'query') is None
    assert 'untested' in t.meta['interactions']['a+b|initial|query']


def test_missing_joint_ablation_removes_single_entries():
    """配置了互补组但没跑联合消融：两组单独条目不写入（第四轮 N2）。"""
    groups = {i: f'g{i}' for i in range(12)}
    types = {i: 'query' for i in range(12)}
    ok = {i: 1.0 for i in range(12)}
    r = recs('full', ok) + recs('ablate:a@initial', ok) + recs('ablate:b@initial', ok) + recs('ablate:c@initial', ok)
    t = estimate(r, groups, types, b=100, min_exposed_groups=10, pairs=[['a', 'b']])
    assert t.lookup('a', 'initial', 'query') is None and t.lookup('b', 'initial', 'query') is None
    assert t.lookup('c', 'initial', 'query') == 0.0
    assert t.meta['interactions']['a+b|initial|query']['untested'] == 'joint ablation not run'


def test_all_infrastructure_joint_ablation_removes_single_entries():
    groups = {i: f'g{i}' for i in range(12)}
    types = {i: 'query' for i in range(12)}
    ok = {i: 1.0 for i in range(12)}
    joint = [dict(x, status='infrastructure_error') for x in recs('ablate:a+b@initial', ok)]
    r = recs('full', ok) + recs('ablate:a@initial', ok) + recs('ablate:b@initial', ok) + joint
    t = estimate(r, groups, types, b=100, min_exposed_groups=10, pairs=[['a', 'b']])
    assert t.lookup('a', 'initial', 'query') is None and t.lookup('b', 'initial', 'query') is None
    assert 'untested' in t.meta['interactions']['a+b|initial|query']


def test_covered_joint_ablation_keeps_single_entries():
    groups = {i: f'g{i}' for i in range(12)}
    types = {i: 'query' for i in range(12)}
    ok = {i: 1.0 for i in range(12)}
    r = (recs('full', ok) + recs('ablate:a@initial', ok) + recs('ablate:b@initial', ok)
         + recs('ablate:b+a@initial', ok))                                  # 联合消融按相反顺序书写
    t = estimate(r, groups, types, b=100, min_exposed_groups=10, pairs=[['a', 'b']])
    assert t.lookup('a', 'initial', 'query') == 0.0 and t.lookup('b', 'initial', 'query') == 0.0
    assert 'untested' not in t.meta['interactions']['b+a|initial|query'] and t.merged == []


@pytest.mark.parametrize('joint_exposed', [0, 3, 12])
def test_single_only_table_ignores_joint_ablations(joint_exposed):
    """single-only 对照只取决于 full 与单组消融；主方法仍按联合消融覆盖保守删除（第五轮 N4）。"""
    from prism_thesis.prism.risk import single_only_records
    groups = {i: f'g{i}' for i in range(12)}
    types = {i: 'query' for i in range(12)}
    ok = {i: 1.0 for i in range(12)}
    base = recs('full', ok) + recs('ablate:a@initial', ok) + recs('ablate:b@initial', ok)
    joint = [dict(x, turns=[DROP_TURN] if x['task_id'] < joint_exposed else [FULL_TURN])
             for x in recs('ablate:a+b@initial', ok)] if joint_exposed else []
    single = estimate(single_only_records(base + joint), groups, types, b=100, min_exposed_groups=10)
    assert single.entries == {'a|initial|query': 0.0, 'b|initial|query': 0.0, 'a|initial|any': 0.0,
                              'b|initial|any': 0.0}
    assert single.meta['interactions'] == {} and single.merged == []
    main = estimate(base + joint, groups, types, b=100, min_exposed_groups=10, pairs=[['a', 'b']])
    assert (main.lookup('a', 'initial', 'query') is None) == (joint_exposed < 10)
