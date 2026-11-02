"""正式校准与比较的输入检查（S2）。"""
import copy

from prism_thesis import cli
from prism_thesis.checks import check_records
from prism_thesis.config import ROOT, load_config
from prism_thesis.data import load_tasks
from prism_thesis.runner import experiment_fingerprint

CFG = load_config(ROOT / 'configs' / 'default.json')
CONDS = ['full'] + [f'prism@{e:g}' for e in CFG['prism']['epsilons_pp']]
MODEL = CFG['model']['model']


def records(split='construction', reps=10, conds=CONDS, model=MODEL, tasks=None, cfg=CFG):
    tasks = tasks if tasks is not None else load_tasks(split)
    fps = {c: experiment_fingerprint(cfg, c) for c in conds}
    return [{'run_key': f'{c}|{t.task_id}|{r}', 'condition': c, 'task_id': t.task_id, 'rep': r, 'split': t.split,
             'model': model, 'fingerprint': fps[c], 'success': True, 'status': 'answered'}
            for c in conds for t in tasks for r in range(reps)]


def test_complete_construction_data_passes():
    assert check_records(records(), CONDS, CFG, 'construction', CFG['reps']['calibrate']) == []


def test_one_task_scripted_dev_rehearsal_is_rejected():
    """审查中的复现：开发集 1 题、脚本化模型、每条件 1 次，原先四个候选全部“通过”。"""
    t = load_tasks('development')[:1]
    recs = records('development', reps=1, model='scripted-gold', tasks=t)
    problems = check_records(recs, CONDS, CFG, 'construction', 10)
    text = ' '.join(problems)
    assert '重复' in text and 'development' in text and 'scripted-gold' in text and '覆盖' in text


def test_missing_rep_and_missing_tasks_rejected():
    recs = records()
    assert check_records([r for r in recs if not (r['condition'] == 'full' and r['rep'] == 9)], CONDS, CFG,
                         'construction', 10)
    first = recs[0]['task_id']
    assert check_records([r for r in recs if not (r['condition'] == 'prism@2' and r['task_id'] == first)], CONDS,
                         CFG, 'construction', 10)


def test_wrong_split_rejected():
    assert check_records(records('test'), CONDS, CFG, 'construction', 10)
    assert check_records(records('development', reps=10), CONDS, CFG, 'test', None)


def test_fingerprint_must_match_current_config():
    cfg = copy.deepcopy(CFG)
    cfg['model']['temperature'] = 0.9
    assert check_records(records(cfg=cfg), CONDS, CFG, 'construction', 10)


def test_rehearsal_only_checks_consistency():
    t = load_tasks('development')[:3]
    recs = records('development', reps=1, model='scripted-gold', tasks=t)
    assert check_records(recs, CONDS, CFG, 'construction', 10, rehearsal=True) == []
    assert check_records(recs[1:], CONDS, CFG, 'construction', 10, rehearsal=True)     # 题目集合不一致仍拒绝


def test_cli_calibrate_refuses_rehearsal_data(tmp_path, capsys):
    import json
    t = load_tasks('development')[:1]
    p = tmp_path / 'r.jsonl'
    p.write_text('\n'.join(json.dumps(r) for r in records('development', reps=1, model='scripted-gold', tasks=t)))
    assert cli.main(['calibrate', '--runs', str(p)]) == 1
    assert 'ε*' not in capsys.readouterr().out


# ---- 第三轮：R2 最终比较按预定次数检查 ----

def compare_records(reps_of):
    tasks = load_tasks('test')
    out = []
    for c, n in reps_of.items():
        out += records('test', reps=n, conds=[c], tasks=tasks)
    return out


def test_expected_reps_manifest():
    from prism_thesis.checks import expected_reps
    m = CFG['reps']['compare']
    assert expected_reps(m, 'full') == 10 and expected_reps(m, 'prism@4') == 10
    assert expected_reps(m, 'prism-stateless@4') == 10 and expected_reps(m, 'random_drop') == 10   # 测试集各条件均 10 次
    assert expected_reps(m, 'nomem') == 10 and expected_reps(CFG['reps']['memory_build'], 'nomem') == 2
    assert expected_reps(CFG['reps']['risk_estimate'], 'ablate:schema@initial') == 2


def test_compare_with_one_rep_everywhere_is_rejected():
    """审查复现：测试集 114 题，full 与 phase_switch 各只有 rep=0。"""
    recs = compare_records({'full': 1, 'phase_switch': 1})
    problems = check_records(recs, ['full', 'phase_switch'], CFG, 'test', CFG['reps']['compare'])
    assert len(problems) == 2 and all('预定 10 次' in p for p in problems)


def test_compare_plan_reps_pass_and_any_shortfall_fails():
    plan = {'full': 10, 'nomem': 10, 'phase_switch': 10, 'prism@4': 10, 'random_drop': 10}
    recs = compare_records(plan)
    assert check_records(recs, list(plan), CFG, 'test', CFG['reps']['compare']) == []
    short = [r for r in recs if not (r['condition'] == 'random_drop' and r['rep'] == 9)]
    assert check_records(short, list(plan), CFG, 'test', CFG['reps']['compare'])


def test_condition_without_planned_reps_is_rejected():
    recs = records('construction', reps=10, conds=['full'])
    assert check_records(recs, ['full'], CFG, 'construction', {'prism@*': 10})


def test_mixed_code_versions_rejected(monkeypatch):
    from prism_thesis import runner
    recs = records()
    monkeypatch.setattr(runner, 'experiment_code_hash', lambda: 'NEW-CODE')
    assert any('指纹' in p for p in check_records(recs, CONDS, CFG, 'construction', 10))


def test_required_joint_ablations_follow_config_pairs():
    from prism_thesis.cli import _required_joints
    conds = ['full', 'ablate:one_stmt@initial', 'ablate:scope@initial', 'ablate:scope@after_error',
             'ablate:answer_query@initial', 'ablate:empty_scalar@initial', 'ablate:empty_scalar+answer_query@initial']
    pairs = [['answer_query', 'empty_scalar'], ['one_stmt', 'scope']]
    assert _required_joints(conds, pairs) == ['ablate:one_stmt+scope@initial']
