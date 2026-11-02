"""evaluate.match_direct 与官方 DirectTypeAnswerValidator 逐例一致。"""
import random

from official_validator import DirectTypeAnswerValidator as Official

from prism_thesis.data import load_tasks
from prism_thesis.evaluate import match_direct

CASES = [
    ('[(1,)]', [(1,)]), ('1', [(1,)]), ("[('a', 2.0)]", [('a', 2)]), ("[('-55',)]", [(-55,)]),
    ('[]', []), ('', []), ('[(Decimal(\'3.50\'),)]', [(3.5,)]), ('[(1, "x"), (2, "y")]', [(1, 'x'), (2, 'y')]),
    ("[('it''s',)]", [("it's",)]), ('[(1.0000001,)]', [(1,)]), ('[(1.01,)]', [(1,)]), ('((1)', [(1,)]),
    ('[(1,), (2,)]', [(2,), (1,)]), ('"abc"', [('abc',)]), ("'abc", [('abc',)]), ('abc', [('abc', 1)]),
    ('[(None,)]', [('None',)]), ('[(Decimal(Decimal(1)),)]', [(1,)]), ('x)', [(1,)]),
]


def test_fixed_cases():
    for ans, truth in CASES:
        assert match_direct(ans, truth) == Official.validate(ans, truth), (ans, truth)


def test_random_against_official():
    rng = random.Random(0)
    atoms = ['1', '2.5', '-3', "'a'", '"b"', 'c', "Decimal('4.0')", 'None', '', ' ', '(', ')', ',', '[', ']', "'7'"]
    truths = [[(1,)], [(2.5, 'a')], [], [('c',), (1,)], [(-3, 'b', 4.0)], [('7',)], [(7,)]]
    for _ in range(20000):
        ans = ''.join(rng.choice(atoms) for _ in range(rng.randint(0, 8)))
        truth = rng.choice(truths)
        assert match_direct(ans, truth) == Official.validate(ans, truth), (ans, truth)


def test_reference_answers_validate():
    """每道查询题的标准答案按 str(行列表) 写出时，两种实现都判为成功。"""
    for t in load_tasks(include_excluded=True):
        if t.answer_direct is not None:
            s = str([tuple(r) for r in t.answer_direct])
            assert match_direct(s, t.answer_direct) == Official.validate(s, t.answer_direct) is True
