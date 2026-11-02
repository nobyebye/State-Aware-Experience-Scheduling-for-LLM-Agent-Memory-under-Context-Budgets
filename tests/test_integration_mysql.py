"""需要 Docker 与 MySQL 镜像。设置环境变量 PRISM_TEST_MYSQL=1 时运行。"""
import os
import random

import pytest

from prism_thesis import scripted
from prism_thesis.agent import run_episode
from prism_thesis.data import load_tasks
from prism_thesis.env import MySQLServer, TaskSession
from prism_thesis.policies import FullPolicy, PhaseSwitchPolicy, PrismPolicy
from prism_thesis.tokens import TokenCounter
from test_policies import table

pytestmark = pytest.mark.skipif(os.environ.get('PRISM_TEST_MYSQL') != '1', reason='需要 Docker MySQL')


@pytest.fixture(scope='module')
def server():
    with MySQLServer(port=13399) as s:
        yield s


def test_isolation_and_single_statement(server):
    task = load_tasks('development')[0]
    with TaskSession(server, task) as s:
        assert not s.execute('SELECT 1; SELECT 2').ok
        r = s.execute('DROP TABLE `%s`' % task.table)
        assert not r.ok and r.error_code == 1142
        assert not s.execute('SELECT * FROM mysql.user').ok
        before = s.table_md5()
        assert s.execute(f'DELETE FROM `{task.table}`').ok and s.table_md5() == 'None' != before


def test_episodes_with_scripted_agents(server):
    counter = TokenCounter()
    tasks = load_tasks('development')[:12]
    for kind in ('gold', 'gold_after_error'):
        for policy in (FullPolicy(), PhaseSwitchPolicy(), PrismPolicy(table(0.0), 2.0, counter)):
            for t in tasks:
                with TaskSession(server, t) as s:
                    ep = run_episode(t, policy, scripted.make(kind, t), s, counter, rng=random.Random(0))
                assert ep.success, (kind, policy.name, t.task_id, ep.status)
                assert ep.input_tokens > 0
                if kind == 'gold_after_error':
                    assert ep.turns[1]['phase'] == 'after_error'


def test_wrong_answer_fails(server):
    t = next(t for t in load_tasks('development') if t.is_query and t.answer_direct)
    with TaskSession(server, t) as s:
        ep = run_episode(t, FullPolicy(), scripted.ScriptedLLM(lambda m: 'Action: Answer\nFinal Answer: [(123456789,)]'),
                         s, TokenCounter())
    assert not ep.success and ep.status == 'answered'


def test_killed_connection_is_infrastructure_error(server):
    from prism_thesis.env import InfrastructureError
    task = load_tasks('development')[0]
    with TaskSession(server, task) as s:
        assert s.execute("SELECT 'it\\'s;ok'").ok                     # 转义引号中的分号不拆分
        conn_id = s.conn.thread_id()
        root = server.connect()
        with root.cursor() as cur:
            cur.execute(f'KILL {conn_id}')
        root.close()
        with pytest.raises(InfrastructureError):
            s.execute('SELECT 1')


def test_line_comment_does_not_swallow_where_end_to_end(server):
    """first_action → parse → TaskSession.execute：注释后的 WHERE 仍然生效，只改一行。"""
    from prism_thesis.parser import first_action, parse
    task = next(t for t in load_tasks('development') if any(c.type.upper().startswith('INT') for c in t.columns))
    col = next(c.name for c in task.columns if c.type.upper().startswith('INT'))
    with TaskSession(server, task) as s:
        n = len(eval(s.execute(f'SELECT `{col}` FROM `{task.table}`').text))
        assert n > 1
        for comment in ('-- set to a sentinel', '# set to a sentinel'):
            sql = (f'UPDATE `{task.table}` SET `{col}` = -987 {comment}\n'
                   f'WHERE `{col}` = (SELECT m FROM (SELECT MAX(`{col}`) AS m FROM `{task.table}`) x)')
            r = s.execute(parse(first_action(f'Action: Operation\n```sql\n{sql}\n```')).content)
            assert r.ok, r.text
        changed = eval(s.execute(f'SELECT COUNT(*) FROM `{task.table}` WHERE `{col}` = -987').text)[0][0]
        assert 1 <= changed < n
        assert not s.execute(parse('Action: Operation\n```sql\nSELECT 1 -- a\n; SELECT 2\n```').content).ok
        assert s.execute(parse("Action: Operation\n```sql\nSELECT 'a\nb'\n```").content).text == "[('a\\nb',)]"


def test_episodes_with_memory(server):
    """带经验记忆的完整记忆注入、人工阶段切换与 PRISM：真实执行，出错后检索到修复经验。"""
    from prism_thesis.memory import Retriever
    from prism_thesis.policies import PhaseSwitchPolicy
    from test_memory import sample_store, table as mem_table
    counter = TokenCounter()
    store = sample_store(load_tasks('development')[20:30])
    tasks = load_tasks('development')[:8]
    for make in (FullPolicy, PhaseSwitchPolicy, lambda: PrismPolicy(mem_table(0.0), 2.0, counter)):
        policy = make()
        policy.memory = Retriever(store)
        for t in tasks:
            with TaskSession(server, t) as s:
                ep = run_episode(t, policy, scripted.make('gold_after_error', t), s, counter, rng=random.Random(0))
            assert ep.success and ep.memory_tokens > 0, (policy.name, t.task_id)
            if not isinstance(policy, FullPolicy):
                assert any(u.startswith('exp:r') for u in ep.turns[1]['experiences'])
