"""批量运行器：续跑的实验身份（S3）、费用累计（S5）、线程失败传播（S6）。用假的模型与数据库，不产生费用。"""
import copy
import json

import pytest

from prism_thesis import runner
from prism_thesis.agent import EpisodeResult
from prism_thesis.config import ROOT, load_config
from prism_thesis.data import load_tasks
from prism_thesis.llm import LLMResponse
from prism_thesis.runner import (ExperimentMismatch, exit_code, experiment_fingerprint, load_records,
                                 run_experiment, run_key)

CFG = load_config(ROOT / 'configs' / 'default.json')
CFG['memory']['enabled'] = False          # 运行器机制测试不依赖记忆库（记忆相关见 test_memory.py）
TASKS = load_tasks('development')[:2]


class FakeServer:
    fail = False

    def __init__(self, **kw):
        pass

    def start(self):
        if FakeServer.fail:
            raise RuntimeError('docker not running')

    def stop(self):
        pass


class FakeSession:
    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class PaidLLM:
    model = 'meta-llama/llama-3.1-8b-instruct'

    def chat(self, messages):
        return LLMResponse('x', prompt_tokens=1000, completion_tokens=100)


def fake_episode(fail_after_first_call=False):
    def run(task, policy, llm, session, counter, **kw):
        llm.chat([])
        if fail_after_first_call:
            raise ConnectionError('provider dropped the second request')
        return EpisodeResult(True, 'answered', 'done', api_prompt_tokens=1000, api_completion_tokens=100)
    return run


@pytest.fixture
def fake_env(monkeypatch):
    FakeServer.fail = False
    monkeypatch.setattr(runner, 'MySQLServer', FakeServer)
    monkeypatch.setattr(runner, 'TaskSession', FakeSession)
    monkeypatch.setattr(runner, 'run_episode', fake_episode())
    return monkeypatch


def go(cfg, out, reps=1, conditions=('full',), paid=True, logs=None):
    return run_experiment(cfg, TASKS, list(conditions), reps, out, lambda t: PaidLLM(), paid=paid, workers=1,
                          log=(logs.append if logs is not None else (lambda m: None)))


def test_resume_same_experiment_skips_done(fake_env, tmp_path):
    out = tmp_path / 'r.jsonl'
    assert go(CFG, out)['ran'] == 2
    state = go(CFG, out)
    assert state['ran'] == 0 and exit_code(state) == 0


@pytest.mark.parametrize('change', ['model', 'variant', 'decoding'])
def test_resume_refuses_different_experiment(fake_env, tmp_path, change):
    out = tmp_path / 'r.jsonl'
    go(CFG, out)
    cfg = copy.deepcopy(CFG)
    if change == 'model':
        cfg['model']['model'] = 'other/model'
    elif change == 'variant':
        cfg['prompt_variant'] = 'extended'
    else:
        cfg['model']['temperature'] = 0.7
    with pytest.raises(ExperimentMismatch):
        go(cfg, out)


def test_old_record_without_fingerprint_is_refused(fake_env, tmp_path):
    out = tmp_path / 'r.jsonl'
    out.write_text(json.dumps({'run_key': run_key('full', TASKS[0].task_id, 0), 'condition': 'full',
                               'model': 'old-model', 'status': 'answered', 'success': True}) + '\n')
    with pytest.raises(ExperimentMismatch):
        go(CFG, out)


def test_risk_table_content_is_part_of_identity(tmp_path):
    cfg = copy.deepcopy(CFG)
    table = tmp_path / 'risk.json'
    cfg['prism']['risk_table'] = str(table)
    table.write_text(json.dumps({'entries': {'schema|initial|query': 1.0}}))
    a = experiment_fingerprint(cfg, 'prism@2')
    full = experiment_fingerprint(cfg, 'full')
    table.write_text(json.dumps({'entries': {'schema|initial|query': 9.0}}))
    assert experiment_fingerprint(cfg, 'prism@2') != a
    assert experiment_fingerprint(cfg, 'full') == full          # 完整提示不依赖风险表


def test_load_records_rejects_conflicting_identities(tmp_path):
    a, b = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    rec = {'run_key': 'full|1|0', 'condition': 'full', 'status': 'answered', 'fingerprint': 'aaa'}
    a.write_text(json.dumps(rec) + '\n')
    b.write_text(json.dumps({**rec, 'fingerprint': 'bbb'}) + '\n')
    with pytest.raises(ExperimentMismatch):
        load_records([a, b])
    b.write_text(json.dumps(rec) + '\n')
    assert len(load_records([a, b])) == 1


def test_budget_counts_previous_spending(fake_env, tmp_path):
    out = tmp_path / 'r.jsonl'
    cfg = copy.deepcopy(CFG)
    go(cfg, out)                                               # 记录 2 次费用
    spent = sum(r['cost_usd_est'] for r in load_records([out]))
    cfg['run']['budget_usd'] = spent / 2
    calls = []
    state = run_experiment(cfg, TASKS, ['full'], 2, out, lambda t: calls.append(t) or PaidLLM(), paid=True,
                           workers=1, log=lambda m: None)
    assert calls == [] and state['stopped'] and exit_code(state) == 3


def test_failed_run_keeps_cost_of_returned_responses(fake_env, tmp_path):
    fake_env.setattr(runner, 'run_episode', fake_episode(fail_after_first_call=True))
    out = tmp_path / 'r.jsonl'
    state = go(CFG, out)
    recs = [json.loads(l) for l in out.read_text(encoding='utf-8').splitlines()]
    assert all(r['status'] == 'infrastructure_error' and r['cost_usd_est'] > 0 for r in recs)
    assert state['cost'] == pytest.approx(sum(r['cost_usd_est'] for r in recs)) and exit_code(state) == 1


def test_zero_budget_makes_no_calls(fake_env, tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg['run']['budget_usd'] = 0.0
    calls = []
    state = run_experiment(cfg, TASKS, ['full'], 1, tmp_path / 'r.jsonl', lambda t: calls.append(t) or PaidLLM(),
                           paid=True, workers=1, log=lambda m: None)
    assert calls == [] and exit_code(state) == 3


def test_worker_start_failure_is_reported(fake_env, tmp_path):
    FakeServer.fail = True
    state = go(CFG, tmp_path / 'r.jsonl')
    assert state['errors'] and state['unfinished'] == 2 and exit_code(state) == 1


def test_missing_risk_table_fails_before_dispatch(fake_env, tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg['prism']['risk_table'] = str(tmp_path / 'nope.json')
    with pytest.raises(FileNotFoundError):
        go(cfg, tmp_path / 'r.jsonl', conditions=('prism@2',))


def test_cli_run_exit_code_nonzero_on_failure(fake_env, tmp_path, monkeypatch):
    from prism_thesis import cli, llm
    FakeServer.fail = True
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test')
    monkeypatch.setattr(llm, 'OpenAICompatibleLLM', lambda *a, **k: PaidLLM())
    code = cli.main(['run', '--split', 'dev', '--limit', '2', '--conditions', 'nomem', '--out',
                     str(tmp_path / 'r.jsonl'), '--allow-paid', '--workers', '1'])
    assert code == 1


# ---- 第三轮：R3 代码与运行方式属于实验身份，R4 缺少 usage 时的费用 ----

def test_experiment_code_changes_refuse_resume(fake_env, tmp_path):
    out = tmp_path / 'r.jsonl'
    go(CFG, out)
    fake_env.setattr(runner, 'experiment_code_hash', lambda: 'NEW-CODE')
    with pytest.raises(ExperimentMismatch):
        go(CFG, out)


def test_experiment_modules_cover_behaviour_code():
    names = {f.name for f in runner.experiment_modules()}
    assert {'parser.py', 'env.py', 'evaluate.py', 'features.py', 'agent.py', 'prompts.py', 'prism.py', 'risk.py',
            'runner.py'} <= names
    assert not names & {'cli.py', 'checks.py', 'stats.py'}


def test_unpaid_records_do_not_count_for_paid_run(fake_env, tmp_path):
    out = tmp_path / 'r.jsonl'
    go(CFG, out, paid=False)
    with pytest.raises(ExperimentMismatch):
        go(CFG, out, paid=True)


class NoUsageLLM:
    model = 'meta-llama/llama-3.1-8b-instruct'

    def __init__(self, completion=None):
        self.completion = completion

    def chat(self, messages):
        return LLMResponse('Action: Operation\n```sql\nSELECT 1\n```', prompt_tokens=None,
                           completion_tokens=self.completion)


@pytest.mark.parametrize('completion', [None, 7])
def test_missing_usage_is_estimated_locally(fake_env, tmp_path, completion):
    def failing(task, policy, llm, session, counter, **kw):
        llm.chat([{'role': 'user', 'content': 'hello ' * 100}])
        raise ConnectionError('dropped')
    fake_env.setattr(runner, 'run_episode', failing)
    out = tmp_path / 'r.jsonl'
    state = run_experiment(CFG, TASKS, ['full'], 1, out, lambda t: NoUsageLLM(completion), paid=True, workers=1,
                           log=lambda m: None)
    recs = [json.loads(l) for l in out.read_text(encoding='utf-8').splitlines()]
    assert all(r['cost_complete'] is False and r['cost_usd_est'] > 0 for r in recs)
    assert state['cost'] > 0


def test_missing_usage_counts_toward_budget(fake_env, tmp_path):
    def failing(task, policy, llm, session, counter, **kw):
        llm.chat([{'role': 'user', 'content': 'hello ' * 100000}])
        raise ConnectionError('dropped')
    fake_env.setattr(runner, 'run_episode', failing)
    cfg = copy.deepcopy(CFG)
    cfg['run']['budget_usd'] = 1e-4                     # 一次约 10 万 token 的估计费用就超过上限
    calls = []
    state = run_experiment(cfg, TASKS, ['full'], 3, tmp_path / 'r.jsonl',
                           lambda t: calls.append(t) or NoUsageLLM(), paid=True, workers=1, log=lambda m: None)
    assert len(calls) == 1 and state['stopped'] and exit_code(state) == 1


def test_truncated_tail_is_repaired_on_resume(fake_env, tmp_path):
    """末行写入中断：续跑把残行移到旁边的文件并重跑该次，不与新记录拼接（第四轮 N3）。"""
    out = tmp_path / 'r.jsonl'
    go(CFG, out)
    lines = out.read_text(encoding='utf-8').splitlines()
    out.write_text(lines[0] + '\n' + lines[1][:25], encoding='utf-8')
    with pytest.raises(runner.CorruptResults):
        load_records([out])
    logs = []
    state = go(CFG, out, logs=logs)
    assert state['ran'] == 1 and exit_code(state) == 0
    assert len(load_records([out])) == 2
    assert [p.read_text(encoding='utf-8') for p in tmp_path.glob('r.jsonl.truncated-*')] == [lines[1][:25]]
    assert any('不完整' in m for m in logs)


def test_complete_tail_without_newline_is_kept(fake_env, tmp_path):
    out = tmp_path / 'r.jsonl'
    go(CFG, out)
    out.write_text(out.read_text(encoding='utf-8').rstrip('\n'), encoding='utf-8')
    state = go(CFG, out)
    assert state['ran'] == 0 and len(load_records([out])) == 2
    assert not list(tmp_path.glob('r.jsonl.truncated-*'))


def test_corrupt_middle_line_is_refused(fake_env, tmp_path):
    out = tmp_path / 'r.jsonl'
    go(CFG, out)
    lines = out.read_text(encoding='utf-8').splitlines()
    out.write_text(lines[0][:25] + '\n' + lines[1] + '\n', encoding='utf-8')
    with pytest.raises(runner.CorruptResults, match='第 1 行'):
        go(CFG, out)
    with pytest.raises(runner.CorruptResults):
        load_records([out])


def test_repair_tail_never_overwrites_backups(tmp_path, monkeypatch):
    """同一秒内两次修复各保留一份备份，已有同名备份不被覆盖（第五轮 N5）。"""
    monkeypatch.setattr(runner.time, 'strftime', lambda *a: '20261023T010101')
    out = tmp_path / 'r.jsonl'
    (tmp_path / 'r.jsonl.truncated-20261023T010101').write_bytes(b'old')
    for tail in (b'{"first', b'{"second'):
        out.write_bytes(b'{"a": 1}\n' + tail)
        runner.repair_tail(out, log=lambda m: None)
        assert out.read_bytes() == b'{"a": 1}\n'
    got = sorted(p.read_bytes() for p in tmp_path.glob('r.jsonl.truncated-*'))
    assert got == [b'old', b'{"first', b'{"second']


def test_repair_tail_keeps_log_when_backup_fails(tmp_path, monkeypatch):
    out = tmp_path / 'r.jsonl'
    out.write_bytes(b'{"a": 1}\n{"cut')

    def fail(*a, **k):
        raise OSError('disk full')
    monkeypatch.setattr(runner.os, 'fsync', fail)
    with pytest.raises(OSError):
        runner.repair_tail(out, log=lambda m: None)
    assert out.read_bytes() == b'{"a": 1}\n{"cut'
