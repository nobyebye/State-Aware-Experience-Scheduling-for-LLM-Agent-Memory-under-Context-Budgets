import random

from prism_thesis.parser import Parsed, first_action, parse


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
    pieces = ['Action: Operation', 'Action: Answer', 'Action: Query', '\n', '```sql\n', '\n```', 'SELECT 1', '\nFinal Answer: 2',
              'text ', 'Final Answer: 3']
    for _ in range(20000):
        s = ''.join(rng.choice(pieces) for _ in range(rng.randint(0, 9)))
        before, after = parse(s), parse(first_action(s))
        if before.kind != 'invalid':          # 官方已合规的回复：规范化不改变解析结果
            assert after == before, s
        assert parse(first_action(first_action(s))) == after, s


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


def test_first_action_relabels_sql_verb_actions():
    """Llama-3.1-8B 常把操作标签写成 SQL 动词，后面跟合规代码块。"""
    for label in ('Query', 'Select', 'SELECT', 'Update', 'Insert', 'Delete'):
        text = f'Action: {label}\n\n```sql\nSELECT a\nFROM t;\n```\nFinal Answer: [(1,)]'
        out = first_action(text)
        assert out == 'Action: Operation\n```sql\nSELECT a\nFROM t;\n```'
        assert parse(out) == Parsed('operation', 'SELECT a\nFROM t;')
    assert parse(first_action('Action: Query\n```SQL \nSELECT 1\n```')).content == 'SELECT 1'
    assert parse(first_action('Action: Query\n```mysql\nSELECT 1\n```')).content == 'SELECT 1'


def test_first_action_keeps_invalid_without_block():
    """没有代码块的回复仍为无效动作（不猜测裸 SQL）。"""
    for text in ('Action: SELECT\nSELECT customer_id FROM t LIMIT 5;', 'Action: Operation\nSELECT 1',
                 'SELECT 1', 'Action: Query\nno sql here\nAction: Operation\nSELECT 2'):
        assert parse(first_action(text)).kind == 'invalid'


def test_first_action_relabel_uses_first_action_only():
    text = 'Action: Query\n```sql\nSELECT 1\n```\nAction: Update\n```sql\nUPDATE t SET a=1\n```'
    assert parse(first_action(text)).content == 'SELECT 1'
    # 第一个动作没有代码块时，不去拿后面动作的代码块
    text = 'Action: Query\nlet me think\nAction: Update\n```sql\nUPDATE t SET a=1\n```'
    assert parse(first_action(text)).kind == 'invalid'
    assert parse(first_action('Action: answer\nFinal Answer: [(2,)]\n')).content == '[(2,)]'


def test_reparse_command(tmp_path, capsys):
    import json
    from prism_thesis.cli import main
    recs = [{'run_key': 'a', 'status': 'invalid_action', 'success': False,
             'turns': [{'raw_response': 'Action: Query\n```sql\nSELECT 1\n```', 'action': 'invalid',
                        'parse_reason': 'no_action'}]},
            {'run_key': 'b', 'status': 'invalid_action', 'success': False,
             'turns': [{'raw_response': 'Action: SELECT\nSELECT 1;', 'action': 'invalid',
                        'parse_reason': 'no_action'}]}]
    f = tmp_path / 'r.jsonl'
    f.write_text(''.join(json.dumps(r) + '\n' for r in recs), encoding='utf-8')
    assert main(['reparse', '--runs', str(f)]) == 0
    out = capsys.readouterr().out
    assert "'operation': 1" in out and "'invalid:no_action': 1" in out
