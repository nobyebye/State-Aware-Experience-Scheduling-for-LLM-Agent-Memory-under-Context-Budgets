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


def test_first_action_keeps_invalid_without_sql():
    """没有可识别 SQL 的回复仍为无效动作。"""
    for text in ('SELECT 1', 'Action: Query\nno sql here\nAction: Operation\nSELECT 2',
                 'Action: Operation\nI will now look at the table.', 'Action: Operation\n'):
        assert parse(first_action(text)).kind == 'invalid', text


def test_first_action_takes_bare_sql():
    """Llama-3.1-8B 常省略代码块（VM 小试中约一半回复）；取动作行后的裸 SQL，截掉编造的结果表格。"""
    cases = [
        ("Action: Operation\nUPDATE media_submissions\nSET status = 'approved'\nWHERE word_count > 1000;",
         "UPDATE media_submissions\nSET status = 'approved'\nWHERE word_count > 1000;"),
        ("Action: Operation\nINSERT INTO p (a, b)\nVALUES (NULL, 102);\n\n+------+\n| a    |\n+-----",
         'INSERT INTO p (a, b)\nVALUES (NULL, 102);'),
        ('Action: Operation\nSELECT a, b\nFROM t\nWHERE b >= 4\nLIMIT 5\n;\n+----+\n| a  |\n',
         'SELECT a, b\nFROM t\nWHERE b >= 4\nLIMIT 5\n;'),
        ('Action: Operation\nSELECT region\nFROM t\nGROUP BY region', 'SELECT region\nFROM t\nGROUP BY region'),
        ('Action: SELECT\nSELECT customer_id FROM t LIMIT 5;', 'SELECT customer_id FROM t LIMIT 5;'),
        ("Action: UPDATE media_submissions SET status = 'approved' WHERE reviewed = 0",
         "UPDATE media_submissions SET status = 'approved' WHERE reviewed = 0"),
        ('Action: Operation\nSELECT 1\n\nThe result will show one row.', 'SELECT 1'),
        ('Action: Operation\nSELECT 1;\nAction: Answer\nFinal Answer: [(1,)]', 'SELECT 1;'),
    ]
    for text, sql in cases:
        out = first_action(text)
        assert parse(out) == Parsed('operation', sql), (text, out)


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
             'turns': [{'raw_response': 'Action: Operation\nLet me check the table first.', 'action': 'invalid',
                        'parse_reason': 'no_action'}]}]
    f = tmp_path / 'r.jsonl'
    f.write_text(''.join(json.dumps(r) + '\n' for r in recs), encoding='utf-8')
    assert main(['reparse', '--runs', str(f)]) == 0
    out = capsys.readouterr().out
    assert "'operation': 1" in out and "'invalid:no_sql_block': 1" in out
