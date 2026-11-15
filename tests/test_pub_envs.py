"""发表线：环境接口、BIRD-Agent（合成的小数据库）、操作系统环境（解析与规则；容器测试需 PRISM_TEST_DOCKER=1）、
测试集封存。全部不调用模型。"""
import json
import os
import random
import sqlite3
import subprocess
from pathlib import Path

import pytest

from prism_thesis import prompts
from prism_thesis.agent import build_messages, run_episode
from prism_thesis.config import load_config, make_policy, read_config
from prism_thesis.data import load_tasks
from prism_thesis.env import ExecResult
from prism_thesis.envs import default_env, get_env
from prism_thesis.envs import bird as bird_mod
from prism_thesis.envs import os_shell
from prism_thesis.llm import ScriptedLLM
from prism_thesis.memory import MemoryStore
from prism_thesis.policies.base import TurnState
from prism_thesis.policies.joint import JointPolicy
from prism_thesis.pub import bird_prep, cli as pub_cli, os_prep, seal
from prism_thesis.runner import load_records
from prism_thesis.tokens import TokenCounter

ROOT = Path(__file__).resolve().parents[1]
COUNTER = TokenCounter()


# ---------- db_bench：经环境接口得到的内容与原模块相同 ----------

def test_dbbench_env_matches_original_modules():
    env = default_env()
    load_config(ROOT / 'configs' / 'default.json')
    for task in load_tasks('development')[:5]:
        f = env.task_features(task)
        assert f.env is None and f.groups == prompts.group_units(f.predicted_functions)
        ids = env.all_ids(task)
        assert ids == prompts.ALL_IDS
        assert env.render_system(ids, task) == prompts.render_system(ids, task)
        assert env.render_first_user(ids, task) == prompts.render_first_user(ids, task)
    assert env.expert_drop('initial', 'query') == ('fn_other',)
    assert env.expert_drop('after_success', 'modify') == (
        'one_stmt', 'scope', 'fn_predicted', 'fn_other', 'schema', 'answer_query', 'empty_scalar')
    assert env.expert_drop('after_error', 'modify') == ()


def test_config_extends_merges_recursively():
    cfg = read_config(ROOT / 'configs' / 'pub' / 'bird.json')
    assert cfg['env'] == 'bird' and cfg['agent']['max_rounds'] == 6
    assert cfg['agent']['first_action_only'] is True          # 继承自 default.json
    assert cfg['memory']['k_examples'] == 3 and cfg['memory']['store'].startswith('results/pub/bird')


# ---------- BIRD-Agent：合成数据 ----------

def _make_bird(tmp_path: Path) -> dict:
    root = tmp_path / 'dev_databases'
    questions = []
    for d, db in enumerate(('shop', 'school')):
        (root / db / 'database_description').mkdir(parents=True)
        conn = sqlite3.connect(root / db / f'{db}.sqlite')
        conn.execute('CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, price REAL, cat TEXT)')
        conn.execute('CREATE TABLE "order lines" (item_id INTEGER, qty INTEGER)')
        conn.execute('CREATE TABLE unused (x INTEGER)')
        conn.executemany('INSERT INTO items VALUES (?,?,?,?)',
                         [(i, f'item{i}', 1.5 * i, 'a' if i % 2 else 'b') for i in range(1, 41)])
        conn.executemany('INSERT INTO "order lines" VALUES (?,?)', [(i % 40 + 1, i) for i in range(100)])
        conn.commit()
        conn.close()
        (root / db / 'database_description' / 'items.csv').write_text(
            'original_column_name,column_name,column_description,data_format,value_description\n'
            'id,id,unique id of the item,integer,\nprice,price,unit price,real,in euros\nname,name,name,text,\n',
            encoding='utf-8')
        for i in range(1, 31):
            questions.append({'question_id': d * 100 + i, 'db_id': db, 'difficulty': 'simple',
                              'question': f'What is the name of item number {i} in category list {i}?',
                              'evidence': 'item number refers to id' if i % 2 else '',
                              'SQL': f'SELECT name AS n{i} FROM items WHERE id = {i} AND price > {i - 1}.0 - {i % 7}'})
        questions.append({'question_id': d * 100 + 90, 'db_id': db, 'difficulty': 'simple',
                          'question': 'Which items cost more than 1000?', 'evidence': '',
                          'SQL': 'SELECT name FROM items WHERE price > 1000'})          # 空结果 → 排除
        questions.append({'question_id': d * 100 + 91, 'db_id': db, 'difficulty': 'simple',
                          'question': 'Broken gold', 'evidence': '', 'SQL': 'SELECT nope FROM items'})
    (tmp_path / 'dev.json').write_text(json.dumps(questions), encoding='utf-8')
    data_dir = tmp_path / 'data_bird'
    summary = bird_prep.prepare(tmp_path / 'dev.json', root, data_dir,
                                {'build': 20, 'search': 10, 'calib': 10, 'test': 16}, log=lambda *_: None)
    cfg = load_config(ROOT / 'configs' / 'pub' / 'bird.json')
    cfg['bird'].update({'data_dir': str(data_dir), 'db_root': str(root), 'timeout_s': 2})
    cfg['memory']['store'] = str(tmp_path / 'memory.json')
    cfg['prism']['js_table'] = str(tmp_path / 'js_table.json')
    cfg['run']['workers'] = 2
    return {'cfg': cfg, 'summary': summary, 'tmp': tmp_path}


@pytest.fixture
def bird(tmp_path):
    return _make_bird(tmp_path)


def test_bird_prepare_excludes_groups_and_splits(bird):
    s = bird['summary']
    assert s['excluded'] == {'gold_empty': 2, 'gold_error': 2}
    assert s['splits']['excluded'] == 4 and sum(s['splits'].values()) == 64
    env = get_env(bird['cfg'])
    tasks = env.load_tasks()
    assert len(tasks) == 60
    by = {}
    for t in tasks:
        by.setdefault(t.split, set()).add(t.group_id)
    assert {k: len([t for t in tasks if t.split == k]) for k in by} == \
        {'build': 20, 'search': 10, 'calib': 10, 'test': 16, 'reserve': 4}
    assert not any(by[a] & by[b] for a in by for b in by if a < b)
    # 两个库都出现在每个划分中（按库分层）
    assert all({t.db_id for t in tasks if t.split == k} == {'shop', 'school'} for k in ('build', 'test'))
    # 确定性：再次生成得到相同划分
    again = bird_prep.make_splits(
        [{'question_id': t.task_id, 'db_id': t.db_id, 'excluded': False} for t in tasks],
        {t.task_id: t.group_id for t in tasks}, {'build': 20, 'search': 10, 'calib': 10, 'test': 16})
    assert again == {t.task_id: t.split for t in tasks}


def test_bird_grouping_merges_same_skeleton_and_question():
    ts = [{'question_id': 1, 'db_id': 'a', 'question': 'How many X?', 'SQL': "SELECT COUNT(*) FROM t WHERE c = 'x'"},
          {'question_id': 2, 'db_id': 'a', 'question': 'Count the Y.', 'SQL': "SELECT COUNT(*) FROM t WHERE c = 'y'"},
          {'question_id': 3, 'db_id': 'a', 'question': 'count the y', 'SQL': 'SELECT 1'},
          {'question_id': 4, 'db_id': 'b', 'question': 'How many X?', 'SQL': "SELECT COUNT(*) FROM t WHERE c = 'x'"}]
    g = bird_prep.make_groups(ts)
    assert g[1] == g[2] == g[3] == 'a/g0000' and g[4] == 'b/g0000'


def test_bird_units_groups_and_rendering(bird):
    env = get_env(bird['cfg'])
    task = env.load_tasks('build')[0]
    f = env.task_features(task)
    ids = env.all_ids(task)
    assert f.env is env and env.core_ids <= ids
    assert {'tbl:items', 'tbl:order lines', 'tbl:unused', 'desc:items', 'smp:items'} <= ids
    assert 'desc:unused' not in ids and 'smp:unused' not in ids          # 没有列说明、没有数据
    assert f.groups['schema_pred'] == {'tbl:items'}                      # 问题提到 item / name
    assert f.groups['schema_other'] == {'tbl:order lines', 'tbl:unused'}
    covered = frozenset().union(*f.groups.values())
    assert covered == ids - env.core_ids and sum(map(len, f.groups.values())) == len(covered)
    system = env.render_system(ids, task)
    user = env.render_first_user(ids, task)
    assert system.startswith("You answer the user's question") and 'at most 6 rounds' in system
    assert 'Available SQL functions: COUNT, SUM' in system
    assert user.startswith(task.instruction) and f'Database: {task.db_id}' in user
    assert 'CREATE TABLE items' in user and '- price: unit price; in euros' in user and 'Sample rows of items' in user
    assert '- name:' not in user                                         # 与列名相同的说明略去
    # 省略一组后只少这一组的文本，核心内容不变
    less = env.render_first_user(ids - f.groups['samples_pred'] - f.groups['samples_other'], task)
    assert 'Sample rows' not in less and 'CREATE TABLE items' in less
    assert env.render_first_user(env.core_ids, task) == env.unit_text('task', task)


def test_bird_session_errors_readonly_timeout_and_truncation(bird):
    env = get_env(bird['cfg'])
    task = env.load_tasks('build')[0]
    with env.open_session(None, task, 0, bird['cfg']) as s:
        r = s.execute('SELECT id, name FROM items WHERE id <= 2')
        assert r.ok and r.text == "[(1, 'item1'), (2, 'item2')]" and s.last_rows == [(1, 'item1'), (2, 'item2')]
        for sql, kind in (('SELECT nope FROM items', 'no_column'), ('SELECT * FROM nope', 'no_table'),
                          ('SELEC 1', 'syntax'), ('SELECT 1; SELECT 2', 'multi_statement'), ('  ', 'empty'),
                          ('SELECT foo(1)', 'no_function'), ('DELETE FROM items', 'readonly'),
                          ('CREATE TABLE z (a)', 'readonly')):
            r = s.execute(sql)
            assert (r.ok, r.kind) == (False, kind), (sql, r)
            assert env.recovery_groups(r) and env.signature(r, sql).startswith(kind + '|')
        assert s.last_rows == [(1, 'item1'), (2, 'item2')]               # 出错不改变待提交的结果
        slow = ('WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 100000000) '
                'SELECT COUNT(*) FROM c')
        r = s.execute(slow)
        assert (r.ok, r.kind) == (False, 'timeout')
        r = s.execute('SELECT a.id, a.name, b.name FROM items a, items b')
        assert r.ok and len(r.text) < 2100 and 'of 1600 rows' in r.text and len(s.last_rows) == 1600
    assert env.recovery_groups(ExecResult(True, '', None, 'ok')) == frozenset()


def test_bird_grading_is_official_set_equality(bird):
    env = get_env(bird['cfg'])
    task = next(t for t in env.load_tasks('build'))
    with env.open_session(None, task, 0, bird['cfg']) as s:
        assert env.grade(task, s, 'answered', 'submit')[0] is False      # 没有执行过查询
        s.execute(task.reference_sql)
        assert env.grade(task, s, 'round_limit', None)[0] is False       # 没有提交
        assert env.grade(task, s, 'answered', 'submit')[0] is True
        i = task.reference_sql.split('id = ')[1].split()[0]
        s.execute(f'SELECT name FROM items WHERE id = {i} UNION ALL SELECT name FROM items WHERE id = {i}')
        assert env.grade(task, s, 'answered', 'submit')[0] is True       # 重复行不影响（集合比较）
        s.execute('SELECT name, id FROM items WHERE id = 1')
        ok, info = env.grade(task, s, 'answered', 'submit')
        assert ok is False and info['submitted_rows'] == 1


def test_bird_episode_with_wrong_then_right_query(bird):
    env = get_env(bird['cfg'])
    task = env.load_tasks('search')[0]
    replies = iter(['Action: Operation\n```sql\nSELECT nope FROM items\n```',
                    f'Action: Query\n```sql\n{task.reference_sql}\n```\nResult: made up',
                    'Action: Answer\nFinal Answer: submit'])
    policy = make_policy('nomem', bird['cfg'], COUNTER)
    with env.open_session(None, task, 0, bird['cfg']) as s:
        ep = run_episode(task, policy, ScriptedLLM(lambda m: next(replies)), s, COUNTER, max_rounds=6, env=env)
    assert ep.success and ep.status == 'answered'
    assert [t['phase'] for t in ep.turns] == ['initial', 'after_error', 'after_success']
    assert ep.turns[0]['exec']['kind'] == 'no_column' and ep.grade_info['submitted_rows'] == 1


def _dry(cfg, split, conditions, out, agent='gold', reps=1):
    out = Path(out)
    code = pub_cli.main(['--config', str(cfg), 'dry-run', '--split', split, '--conditions', conditions,
                         '--out', str(out), '--agent', agent, '--reps', str(reps)])
    return code, (load_records([out]) if out.exists() else [])


def _write_cfg(b) -> Path:
    p = b['tmp'] / 'cfg.json'
    p.write_text(json.dumps(b['cfg']), encoding='utf-8')
    return p


def test_bird_full_pipeline_memory_and_policies(bird, capsys):
    cfg_path = _write_cfg(bird)
    tmp = bird['tmp']
    assert pub_cli.main(['--config', str(cfg_path), 'check-data', '--hash-dbs']) == 0
    assert pub_cli.main(['--config', str(cfg_path), 'verify-env', '--split', 'calib']) == 0
    # 1. build 划分上的 nomem 演练 → 记忆库
    code, recs = _dry(cfg_path, 'build', 'nomem', tmp / 'nomem.jsonl', 'gold_after_error', reps=2)
    assert code == 0 and len(recs) == 40 and all(r['success'] for r in recs)
    assert all(r['db_id'] in ('shop', 'school') and r['grade_info']['submitted_rows'] == 1 for r in recs)
    assert pub_cli.main(['--config', str(cfg_path), 'build-memory', '--runs', str(tmp / 'nomem.jsonl')]) == 1
    assert pub_cli.main(['--config', str(cfg_path), 'build-memory', '--runs', str(tmp / 'nomem.jsonl'),
                         '--rehearsal']) == 0
    store = MemoryStore.load(tmp / 'memory.json')
    assert store.counts() == {'example': 20, 'repair': 20, 'insight': 0}
    ex = store.of_kind('example')[0]
    assert ex.text.startswith('Example question (database ') and ex.text.endswith('Final Answer: submit')
    assert store.of_kind('repair')[0].error_key == 'no_column'
    # 2. 各条件在 calib 划分上演练：脚本化智能体不看提示，成功率都应为 100%，token 数体现省略
    (tmp / 'js_table.json').write_text(json.dumps({'family': {'8': {
        'initial|query': ['fn_other', 'samples_other', 'desc_other'],
        'after_success|query': ['schema_other', 'schema_pred', 'samples_pred', 'exp_rest']}}}), encoding='utf-8')
    conds = 'full,nomem,phase_switch,random_drop,js-ref,js@8,js-abl:schema_other+insight@after_success,' \
            'ablate:exp_top1@any'
    code, recs = _dry(cfg_path, 'calib', conds, tmp / 'calib.jsonl', 'gold_after_error')
    assert code == 0 and len(recs) == 80 and all(r['success'] for r in recs)
    tok = {}
    for r in recs:
        tok[r['condition']] = tok.get(r['condition'], 0) + r['input_tokens']
    assert tok['js@8'] < tok['js-ref'] and tok['phase_switch'] < tok['js-ref'] and tok['nomem'] < tok['full']
    full = next(r for r in recs if r['condition'] == 'full')
    assert full['n_experiences'] >= 3 and full['turns'][0]['tokens']['memory'] > 0
    # 同库经验优先；同组排除
    env = get_env(bird['cfg'])
    task = next(t for t in env.load_tasks('calib') if t.task_id == full['task_id'])
    for u in full['turns'][0]['experiences']:
        e = store.by_id[u.split(':', 2)[2]]
        assert task.group_id not in e.source_groups
        if e.kind == 'example':
            assert e.source_groups[0].startswith(task.db_id + '/')
    # 出错后：全部单元 + 按错误签名检索到的修复经验
    js = next(r for r in recs if r['condition'] == 'js@8' and r['task_id'] == full['task_id'])
    assert js['turns'][1]['phase'] == 'after_error' and js['turns'][1]['plan_info']['n_repairs'] >= 1
    assert set(js['turns'][1]['units']) >= env.all_ids(task)
    # 成功执行后按表省略；“未知列”错误补回的建表语句之后一直保留
    assert 'smp:items' not in js['turns'][2]['units'] and 'tbl:items' in js['turns'][2]['units']
    assert 'schema_pred' in js['turns'][2]['plan_info']['reincluded']
    # 3. 只在 db_bench 实现的条件被拒绝；测试集不能进记忆库
    assert _dry(cfg_path, 'calib', 'prism@8', tmp / 'x.jsonl')[0] == 1
    code, _ = _dry(cfg_path, 'test', 'nomem', tmp / 'test_nomem.jsonl')
    assert code == 0                                                     # 不付费演练允许
    assert pub_cli.main(['--config', str(cfg_path), 'build-memory', '--runs', str(tmp / 'test_nomem.jsonl'),
                         '--rehearsal']) == 1
    capsys.readouterr()
    assert pub_cli.main(['--config', str(cfg_path), 'show-prompt', '--task-id', str(task.task_id),
                         '--condition', 'js@8', '--phase', 'after_success']) == 0
    shown = capsys.readouterr().out
    assert 'CREATE TABLE' not in shown and 'Database: ' in shown


def test_joint_policy_drops_env_groups_but_never_core(bird):
    env = get_env(bird['cfg'])
    task = env.load_tasks('build')[0]
    p = JointPolicy({('initial', 'query'): tuple(env.selectable_groups)}, 'js')
    p.start(task, env.task_features(task), random.Random(0))
    plan = p.plan(TurnState(0, 'initial', None))
    assert plan.prefix == env.core_ids
    msgs, _ = build_messages(task, plan, [], None, env)
    assert 'Action: Operation' in msgs[0]['content'] and msgs[1]['content'] == env.unit_text('task', task)


# ---------- 操作系统环境 ----------

def test_os_parser_and_first_action():
    raw = 'Let me do it.\nAct: x\nAction: bash\n```sh\nmkdir -p /a/b\n```\nAction: Answer\nFinal Answer: done'
    cut = os_shell.first_action(raw)
    assert cut == 'Action: Operation\n```bash\nmkdir -p /a/b\n```'
    assert os_shell.parse(cut).content == 'mkdir -p /a/b'
    assert os_shell.parse(os_shell.first_action('Action: finish')).kind == 'answer'
    assert os_shell.parse(os_shell.first_action('I am done.')).kind == 'invalid'
    assert os_shell.parse('Action: Operation\nmkdir x').reason == 'no_bash_block'
    assert os_shell.error_kind(127, '') == 'no_command' and os_shell.error_kind(124, '') == 'timeout'
    assert os_shell.error_kind(1, 'chmod: cannot access x: No such file or directory') == 'no_file'
    assert os_shell.first_command('sudo FOO=1 /usr/sbin/useradd -m bob') == 'useradd'


def _os_cfg(tmp_path, rows):
    cfg = load_config(ROOT / 'configs' / 'pub' / 'os.json')
    cfg['os'].update({'data_dir': str(tmp_path / 'data_os'), 'image': 'ubuntu:22.04', 'dockerfile': None,
                      'timeout_s': 3, 'container_prefix': f'prism-os-test-{os.getpid()}'})
    cfg['memory']['enabled'] = False
    cfg['run']['workers'] = 2
    raw = tmp_path / 'entry_dict.json'
    raw.write_text(json.dumps({str(i): r for i, r in enumerate(rows)}), encoding='utf-8')
    return cfg, raw


OS_ROWS = [
    {'instruction': f'Create the directory "/data/d{i}" and give it mode {m}.',
     'initialization_command_item': {'command_name': 'bash', 'script': 'mkdir -p /data'},
     'evaluation_info': str({'ground_truth_command_item': {'command_name': 'bash',
                                                           'script': f'mkdir -p /data/d{i} && chmod {m} /data/d{i}'},
                             'evaluation_command_item': {'command_name': 'bash',
                                                         'script': f'[ "$(stat -c %a /data/d{i})" = "{m}" ]'}}),
     'skill_list': "['mkdir', 'chmod']"}
    for i, m in enumerate((700, 750, 755, 770))
] + [
    {'instruction': 'Create the user alice with home directory.',
     'initialization_command_item': None,
     'evaluation_info': {'ground_truth_command_item': {'command_name': 'bash', 'script': 'useradd -m alice'},
                         'evaluation_command_item': {'command_name': 'bash', 'script': 'id alice && [ -d /home/alice ]'}},
     'skill_list': ['useradd']},
    {'instruction': 'Make sure /tmp exists.',                               # 不操作也通过 → 排除
     'initialization_command_item': None,
     'evaluation_info': {'ground_truth_command_item': {'command_name': 'bash', 'script': 'true'},
                         'evaluation_command_item': {'command_name': 'bash', 'script': '[ -d /tmp ]'}},
     'skill_list': []},
]


def test_os_convert_units_and_expert_rules(tmp_path):
    cfg, raw = _os_cfg(tmp_path, OS_ROWS)
    rows = os_prep.convert(os_prep.load_raw(raw))
    assert rows[0]['evaluation']['script'].startswith('[ "$(stat') and rows[0]['skills'] == ['mkdir', 'chmod']
    assert os_prep.group_key(OS_ROWS[0]['instruction']) == os_prep.group_key(OS_ROWS[1]['instruction'])
    os_prep.prepare(raw, cfg, {'build': 2, 'search': 1, 'calib': 1, 'test': 1}, skip_verify=True,
                    log=lambda *_: None)
    env = get_env(cfg)
    tasks = env.load_tasks()
    assert len(tasks) == 6 and len({t.group_id for t in tasks}) == 3
    task = tasks[0]
    f = env.task_features(task)
    assert {'cmd:mkdir', 'cmd:chmod'} <= f.groups['cmd_predicted'] and f.task_type == 'modify'
    assert frozenset().union(*f.groups.values()) == env.all_ids(task) - env.core_ids
    system = env.render_system(env.all_ids(task), task)
    assert 'Useful commands:\n- useradd' in system and 'at most 5 rounds' in system
    p = make_policy('phase_switch', cfg, COUNTER)
    p.start(task, f, random.Random(0))
    after = p.plan(TurnState(1, 'after_success', ExecResult(True, '', None, 'ok')))
    assert not any(u.startswith('cmd:') for u in after.units) and {'verify', 'observe', 'rounds'} <= after.units
    err = ExecResult(False, 'x', 127, 'no_command')
    assert p.plan(TurnState(2, 'after_error', err, 'foo --bar')).units == env.all_ids(task)
    assert env.signature(err, 'foo --bar') == 'no_command|foo:'


@pytest.mark.skipif(os.environ.get('PRISM_TEST_DOCKER') != '1', reason='需要 Docker（PRISM_TEST_DOCKER=1）')
def test_os_container_end_to_end(tmp_path):
    cfg, raw = _os_cfg(tmp_path, OS_ROWS)
    summary = os_prep.prepare(raw, cfg, {'build': 2, 'search': 1, 'calib': 1, 'test': 1}, log=lambda *_: None)
    assert summary['excluded'] == {'passes_without_action': 1}
    env = get_env(cfg)
    tasks = env.load_tasks()
    assert len(tasks) == 5
    task = next(t for t in tasks if 'alice' in t.instruction)
    backend = env.make_backend(cfg, 'x')
    backend.start()
    try:
        with env.open_session(backend, task, 'x', cfg) as s:
            r = s.execute('echo hi; nosuchcmd')
            assert (r.ok, r.kind, r.error_code) == (False, 'no_command', 127) and 'hi' in r.text
            assert s.execute('cd /tmp && pwd').text == 'The output of the OS:\n\n/tmp'
            assert s.execute('true').text == 'The output of the OS is empty.'
            assert s.execute('sleep 30').kind == 'timeout'
            assert s.execute('getent hosts example.com || echo offline').text.endswith('offline')
            assert env.grade(task, s, 'answered', 'done')[0] is False
            assert s.execute('useradd -m alice').ok
            assert env.grade(task, s, 'answered', 'done') == (True, {'evaluation_exit_code': 0})
    finally:
        backend.stop()
    cfg_path = tmp_path / 'cfg.json'
    cfg_path.write_text(json.dumps(cfg), encoding='utf-8')
    code, recs = _dry(cfg_path, 'all', 'nomem,phase_switch,js-ref', tmp_path / 'os.jsonl', 'gold_after_error')
    assert code == 0 and len(recs) == 15 and all(r['success'] for r in recs)
    assert all([t['phase'] for t in r['turns']] == ['initial', 'after_error', 'after_success'] for r in recs)
    left = subprocess.run(['docker', 'ps', '-aq', '--filter', f"name={cfg['os']['container_prefix']}"],
                          capture_output=True, text=True).stdout.strip()
    assert left == ''


# ---------- 测试集封存 ----------

def _repo(tmp_path) -> Path:
    root = tmp_path / 'repo'
    (root / 'src').mkdir(parents=True)
    (root / 'docs').mkdir()
    (root / 'src' / 'a.py').write_text('x = 1\n')
    (root / 'docs' / 'prereg.md').write_text('H1 ...\n')

    def git(*a):
        subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@example.com', '-c', 'commit.gpgsign=false',
                        *a], cwd=root, check=True, capture_output=True)
    git('init', '-q')
    git('add', '.')
    git('commit', '-q', '-m', 'init')
    return root


def test_seal_requires_committed_prereg_and_single_use(tmp_path):
    root = _repo(tmp_path)
    log = root / 'docs' / 'test_usage.jsonl'
    kw = dict(env='bird', split='test', model='m1', conditions=['full', 'js@8'], n_tasks=500, reps=10)
    with pytest.raises(seal.SealError, match='预先登记'):
        seal.authorize(log, root, out='r/a.jsonl', prereg=None, **kw)
    (root / 'docs' / 'draft.md').write_text('not committed\n')
    with pytest.raises(seal.SealError, match='尚未提交'):
        seal.authorize(log, root, out='r/a.jsonl', prereg='docs/draft.md', **kw)
    (root / 'src' / 'a.py').write_text('x = 2\n')
    with pytest.raises(seal.SealError, match='未提交的改动'):
        seal.authorize(log, root, out='r/a.jsonl', prereg='docs/prereg.md', **kw)
    (root / 'src' / 'a.py').write_text('x = 1\n')
    assert not log.exists()
    e = seal.authorize(log, root, out='r/a.jsonl', prereg='docs/prereg.md', **kw)
    assert e['resumed'] is False and len(e['commit']) == 40 and len(seal.read_log(log)) == 1
    assert seal.authorize(log, root, out='r/a.jsonl', prereg='docs/prereg.md', **kw)['resumed'] is True   # 续跑
    with pytest.raises(seal.SealError, match='只用一次'):
        seal.authorize(log, root, out='r/b.jsonl', prereg='docs/prereg.md', **kw)
    seal.authorize(log, root, out='r/b.jsonl', prereg='docs/prereg.md', rerun_reason='provider outage', **kw)
    seal.authorize(log, root, out='r/c.jsonl', prereg='docs/prereg.md', **{**kw, 'model': 'm2'})         # 另一个模型
    assert [x['rerun_reason'] for x in seal.read_log(log)] == [None, None, 'provider outage', None]
    (root / 'docs' / 'prereg.md').write_text('H1 changed\n')
    with pytest.raises(seal.SealError, match='未提交的改动'):
        seal.authorize(log, root, out='r/a.jsonl', prereg='docs/prereg.md', **kw)


def test_paid_test_run_refused_without_prereg(bird, monkeypatch, capsys):
    cfg_path = _write_cfg(bird)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test')
    base = ['--config', str(cfg_path), 'run', '--split', 'test', '--conditions', 'nomem', '--out',
            str(bird['tmp'] / 't.jsonl')]
    assert pub_cli.main(base) == 2                                        # 没有 --allow-paid
    assert pub_cli.main(base + ['--allow-paid']) == 1
    assert '预先登记' in capsys.readouterr().out
    assert pub_cli.main(base + ['--allow-paid', '--limit', '3', '--prereg', 'README.md']) == 1
    assert '整体运行' in capsys.readouterr().out
    assert not (bird['tmp'] / 't.jsonl').exists()
