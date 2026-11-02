import itertools
import random

from prism_thesis.env import ExecResult, split_statements
from prism_thesis.features import phase, predict_functions, task_type
from prism_thesis.prism import knapsack, recovery
from prism_thesis.prism.risk import RiskTable, parse_ablation
from prism_thesis.tokens import PER_MESSAGE, PER_REQUEST, TokenCounter


def test_split_statements():
    assert split_statements("SELECT ';' FROM t; SELECT 2;") == ["SELECT ';' FROM t", 'SELECT 2']
    assert split_statements('SELECT 1') == ['SELECT 1']
    assert split_statements(' ; ') == []


def test_split_statements_escapes_and_comments():
    single = ["SELECT 'it\\'s;ok'", 'SELECT "a\\";b"', "SELECT 'a''b;c'", 'SELECT `a;b` FROM t',
              'SELECT 1 /* ; */', 'SELECT 1 -- a;b', 'SELECT 1 # x;y', 'SELECT 1;', 'SELECT 1; -- note',
              "SELECT 'x\\\\'"]
    for q in single:
        assert len(split_statements(q)) == 1, q
    assert split_statements("SELECT 'it\\'s;ok'") == ["SELECT 'it\\'s;ok'"]
    for q in ['SELECT 1; SELECT 2', "SELECT 'a\\'';' ; DROP TABLE t", 'SELECT 2--1;SELECT 3',
              'SELECT 1 /* c */; SELECT 2']:
        assert len(split_statements(q)) == 2, q


def test_connection_loss_is_infrastructure_error():
    import pymysql
    import pytest
    from prism_thesis.env import InfrastructureError, TaskSession

    class Cur:
        def __init__(self, exc):
            self.exc = exc

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql):
            raise self.exc

    class Conn:
        def __init__(self, exc):
            self.exc = exc

        def cursor(self):
            return Cur(self.exc)

    s = TaskSession.__new__(TaskSession)
    for exc in (pymysql.err.OperationalError(2013, 'Lost connection to MySQL server during query'),
                pymysql.err.OperationalError(2006, 'MySQL server has gone away'),
                pymysql.err.InterfaceError(0, '')):
        s.conn = Conn(exc)
        with pytest.raises(InfrastructureError):
            s.execute('SELECT 1')
    for exc, code in ((pymysql.err.ProgrammingError(1064, 'syntax'), 1064),
                      (pymysql.err.OperationalError(1142, 'denied'), 1142),
                      (pymysql.err.OperationalError(3024, 'max_execution_time exceeded'), 3024)):
        s.conn = Conn(exc)
        r = s.execute('SELECT 1')
        assert not r.ok and r.kind == 'mysql_error' and r.error_code == code


def test_features():
    assert task_type('Update the price of item 3') == 'modify'
    assert task_type('What is the average price?') == 'query'
    assert {'AVG'} <= predict_functions('What is the average price?')
    assert phase(0, None) == 'initial'
    assert phase(1, ExecResult(False, 'Error', 1064, 'mysql_error')) == 'after_error'
    assert phase(1, ExecResult(True, '[]', None)) == 'after_success'


def test_knapsack_matches_brute_force():
    rng = random.Random(0)
    for _ in range(300):
        n = rng.randint(0, 7)
        groups = [f'g{i}' for i in range(n)]
        values = {g: rng.randint(1, 50) for g in groups}
        risks = {g: rng.choice([-1.0, 0.0, 0.3, 0.5, 0.7, 1.2, 2.0, 3.6, 9.0]) for g in groups}
        eps = rng.choice([0, 0.5, 1, 2, 4, 8])
        best = max((sum(values[g] for g in s), -len(s)) for k in range(n + 1)
                   for s in itertools.combinations(groups, k)
                   if sum(knapsack.weight(risks[g]) for g in s) <= knapsack.math.floor(eps / 0.5 + 1e-9))
        got = knapsack.solve(values, risks, eps)
        assert (sum(values[g] for g in got), -len(got)) == best
        assert sum(knapsack.weight(risks[g]) for g in got) * 0.5 <= eps + 1e-9


def test_weight_rounds_up_and_clips_negative():
    assert knapsack.weight(-3) == 0 and knapsack.weight(0) == 0
    assert knapsack.weight(0.5) == 1 and knapsack.weight(0.51) == 2 and knapsack.weight(1.0) == 2


def test_recovery_map():
    assert recovery.groups_for(None) == frozenset()
    assert recovery.groups_for(ExecResult(True, '[]', None)) == frozenset()
    assert recovery.groups_for(ExecResult(False, 'x', 1054, 'mysql_error')) == {'schema'}
    assert recovery.groups_for(ExecResult(False, 'x', None, 'multi_statement')) == {'one_stmt'}
    assert recovery.groups_for(ExecResult(False, 'x', 9999, 'mysql_error')) == recovery.ALL


def test_risk_table_roundtrip(tmp_path):
    t = RiskTable({'schema|initial|query': 3.0}, [('a', 'b')], {'x': 1})
    t.save(tmp_path / 'r.json')
    u = RiskTable.load(tmp_path / 'r.json')
    assert u.lookup('schema', 'initial', 'query') == 3.0 and u.lookup('schema', 'initial', 'modify') is None
    assert u.merge_map() == {'a': 'a+b', 'b': 'a+b'}
    assert parse_ablation('ablate:a+b@after_success') == (('a', 'b'), 'after_success')
    assert parse_ablation('full') is None


def test_token_formula():
    c = TokenCounter()
    msgs = [{'role': 'system', 'content': 'hello world'}, {'role': 'user', 'content': 'hi'}]
    assert c.count_messages(msgs) == c.count('hello world') + c.count('hi') + 2 * PER_MESSAGE + PER_REQUEST
