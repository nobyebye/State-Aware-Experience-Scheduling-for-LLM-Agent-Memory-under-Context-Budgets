"""提示调试选项：单元措辞替换与表信息写法；以及 show-runs 离线查看命令。"""
import json

import pytest

from prism_thesis import prompts
from prism_thesis.config import load_config, ROOT
from prism_thesis.data import load_tasks
from prism_thesis.runner import experiment_fingerprint


@pytest.fixture(autouse=True)
def reset_options():
    yield
    prompts.set_prompt_options(None, 'official')


def test_overrides_and_schema_styles():
    task = load_tasks('construction')[0]
    prompts.set_prompt_options({'role': 'ROLE TEXT'}, 'legacy')
    assert prompts.render_system(prompts.ALL_IDS, task).startswith('ROLE TEXT\n')
    first = prompts.render_first_user(prompts.ALL_IDS, task)
    assert first.startswith(task.instruction + '\nSchema: {') and first.endswith(f'\nTable: {task.table}')
    prompts.set_prompt_options(None, 'typed')
    assert all(c.type in prompts.render_first_user(prompts.ALL_IDS, task) for c in task.columns)
    prompts.set_prompt_options(None, 'official')
    assert prompts.render_first_user(prompts.ALL_IDS, task) == task.official_prompt


def test_bad_options_rejected():
    for bad in ({'task': 'x'}, {'schema': 'x'}, {'fn:COUNT': 'x'}, {'nope': 'x'}):
        with pytest.raises(ValueError):
            prompts.set_prompt_options(bad)
    with pytest.raises(ValueError):
        prompts.set_prompt_options(None, 'fancy')


def test_probe_configs_load_and_change_fingerprint():
    base = experiment_fingerprint(load_config(ROOT / 'configs' / 'default.json'), 'nomem', paid=False)
    seen = {base}
    for p in sorted((ROOT / 'configs' / 'probe').glob('*.json')):
        fp = experiment_fingerprint(load_config(p), 'nomem', paid=False)
        assert fp not in seen, p
        seen.add(fp)
    load_config(ROOT / 'configs' / 'default.json')


def test_show_runs(tmp_path, capsys):
    from prism_thesis.cli import main
    recs = [{'run_key': 'a', 'condition': 'nomem', 'task_id': 1, 'rep': 0, 'status': 'answered', 'success': False,
             'answer': 'done', 'turns': [
                 {'action': 'operation', 'content': 'SELECT 1', 'observation': '[(1,)]', 'exec': {'ok': True}},
                 {'action': 'operation', 'content': 'UPDATE t SET a=1', 'observation': '[]', 'exec': {'ok': True}},
                 {'action': 'answer', 'content': 'done'}]}]
    f = tmp_path / 'r.jsonl'
    f.write_text(''.join(json.dumps(r) + '\n' for r in recs), encoding='utf-8')
    assert main(['show-runs', '--runs', str(f)]) == 0
    out = capsys.readouterr().out
    assert '查询题中执行了修改语句: 1' in out and '查询题答案为 done/[]/[()]: 1' in out and '>> UPDATE t SET a=1' in out
