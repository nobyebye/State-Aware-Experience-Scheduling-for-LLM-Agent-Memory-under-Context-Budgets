"""db_bench（MySQL）环境：直接委托给原有模块，提示、解析、判分与原实现逐字节相同。"""
from __future__ import annotations

from typing import Optional

from .. import features as feat
from .. import parser, prompts
from ..data import DATA_DIR, load_tasks
from ..evaluate import match_direct
from ..memory import signature as sig
from ..policies.base import TaskFeatures
from ..prism import recovery
from .base import Environment

ANSWER_FORMAT = ('answer_query', 'empty_scalar')


class DbBenchEnv(Environment):
    name = 'db_bench'
    memory_splits = ('construction', 'development')
    obs_label = 'MySQL'
    core_ids = prompts.CORE_IDS
    selectable_groups = prompts.SELECTABLE_GROUPS
    insight_system = ('You analyse attempts of an agent that operates a MySQL database through single SQL '
                      'statements and then gives a final answer.')

    def all_ids(self, task) -> frozenset:
        return prompts.ALL_IDS

    def unit_text(self, uid, task):
        return prompts.unit_text(uid, task)

    def render_rules(self, unit_ids, task):
        return prompts.render_rules(unit_ids, task)

    def render_system(self, unit_ids, task):
        return prompts.render_system(unit_ids, task)

    def render_first_user(self, unit_ids, task):
        return prompts.render_first_user(unit_ids, task)

    def task_features(self, task) -> TaskFeatures:
        fns = feat.predict_functions(task.instruction)
        return TaskFeatures(feat.task_type(task.instruction), fns, prompts.group_units(fns))

    def phase(self, turn, last_exec):
        return feat.phase(turn, last_exec)

    def first_action(self, response):
        return parser.first_action(response)

    def parse(self, response):
        return parser.parse(response)

    def grade(self, task, session, status, answer):
        if task.answer_md5 is not None:
            md5 = session.table_md5()
            return md5 == task.answer_md5, {'final_md5': md5}
        return status == 'answered' and match_direct(answer, task.answer_direct), {}

    def recovery_groups(self, last_exec):
        return recovery.groups_for(last_exec)

    def error_key(self, result):
        return sig.error_key(result)

    def signature(self, result, action):
        return sig.signature(result, action)

    def expert_drop(self, phase, task_type):
        if phase == 'after_error':
            return ()
        drop = ('fn_other',) if phase == 'initial' else ('one_stmt', 'scope', 'fn_predicted', 'fn_other', 'schema')
        return drop + (ANSWER_FORMAT if task_type == 'modify' else ())

    def example_text(self, task, actions, answer, max_answer_chars):
        from ..memory.build import example_text
        return example_text(task, actions, answer, max_answer_chars)

    def repair_text(self, error_line, failed, fixed):
        from ..memory.build import repair_text
        return repair_text(error_line, failed, fixed)

    def load_tasks(self, split=None, include_excluded=False):
        return load_tasks(split, include_excluded)

    def data_files(self):
        return [DATA_DIR / 'db_bench.parquet', DATA_DIR / 'splits.json']

    def scripted(self, kind, task):
        from .. import scripted
        return scripted.make(kind, task)

    def record_extras(self, task, features) -> dict:
        return {'task_type_pred': features.task_type, 'is_query': task.is_query,
                'predicted_functions': sorted(features.predicted_functions)}
