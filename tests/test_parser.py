import random

from prism_thesis.parser import first_action, parse


def test_official_patterns():
    assert parse('Action: Operation\n```sql\nSELECT 1\n```').content == 'SELECT 1'
    assert parse('Action: Operation\n```sql\nSELECT *\nFROM t\n```').content == 'SELECT *\nFROM t'   # 保留换行（与官方的差别）
    assert parse('Action: Operation\nSELECT 1').reason == 'no_sql_block'
    assert parse('Action: Answer\nFinal Answer: [(1,)]').content == '[(1,)]'
    assert parse('Action: Answer Final Answer: x').reason == 'no_final_answer'
    assert parse('I think the answer is 3').reason == 'no_action'
    # 官方只看第一个 Action
    assert parse('Action: Answer\nFinal Answer: 1\nAction: Operation\n```sql\nX\n```').kind == 'answer'


def test_first_action_trims_guessed_continuation():
    text = 'Action: Operation\n```sql\nSELECT 1\n```\nResult: [(1,)]\nAction: Answer\nFinal Answer: [(1,)]'
    assert first_action(text) == 'Action: Operation\n```sql\nSELECT 1\n```'


def test_first_action_preserves_parse():
    rng = random.Random(1)
    pieces = ['Action: Operation', 'Action: Answer', '\n', '```sql\n', '\n```', 'SELECT 1', '\nFinal Answer: 2',
              'text ', 'Final Answer: 3']
    for _ in range(20000):
        s = ''.join(rng.choice(pieces) for _ in range(rng.randint(0, 9)))
        assert parse(first_action(s)) == parse(s), s


def test_sql_newlines_preserved_so_line_comments_keep_their_scope():
    """审查 N1：换行被替换为空格时，“-- ”注释会吞掉下一行的 WHERE。"""
    from prism_thesis.env import split_statements
    for sql in ('UPDATE t SET x=0 -- explanatory comment\nWHERE id=1', 'UPDATE t SET x=0 # note\nWHERE id=1'):
        p = parse(first_action(f'Action: Operation\n```sql\n{sql}\n```\nResult: ...'))
        assert p.content == sql
        assert split_statements(p.content) == [sql]
    two = parse('Action: Operation\n```sql\nSELECT 1 -- a\n; SELECT 2\n```').content
    assert len(split_statements(two)) == 2
    assert parse("Action: Operation\n```sql\nSELECT 'a\nb'\n```").content == "SELECT 'a\nb'"
