from __future__ import annotations

from functools import lru_cache

from ..memory.store import render_memory
from ..prompts import ALL_IDS, render_system
from ..prompts import schema_text as prompt_schema_text
from .base import Plan, Policy, TurnState


PROSE_PREFIXES = ('Rule:', 'Example task:')


class LLMLingua2Policy(Policy):
    """LLMLingua-2 token 级压缩基线：对可压缩内容（系统指令、表信息句与任务开始时检索的经验）按固定压缩率
    压缩，任务指令原样保留；每题压缩一次，各轮相同（官方布局）。压缩率在开发集上确定后冻结。
    scope='memory'（条件 llmlingua2-mem）：只压缩经验块中的规则与示例题目描述，系统指令、表信息、
    代码块、Action 与 Final Answer 行保持原样，避免 token 级删除破坏动作格式与 SQL。
    需要额外安装 llmlingua 与 torch（见 requirements-llmlingua.txt）。"""

    def __init__(self, rate: float, model_name: str, revision: str | None = None, device: str = 'cpu',
                 scope: str = 'all', compressor=None):
        if scope not in ('all', 'memory'):
            raise ValueError(f'未知压缩范围：{scope}')
        self.rate, self.scope = rate, scope
        self.name = f'llmlingua2@{rate:g}' if scope == 'all' else f'llmlingua2-mem@{rate:g}'
        self._compress = lru_cache(maxsize=4096)(self._compress_raw)
        if compressor is not None:       # 测试用
            self.compressor = compressor
            return
        from llmlingua import PromptCompressor
        kwargs = {'model_name': model_name, 'use_llmlingua2': True, 'device_map': device}
        if revision:
            kwargs['model_config'] = {'revision': revision}
        self.compressor = PromptCompressor(**kwargs)

    def _compress_raw(self, text: str) -> str:
        out = self.compressor.compress_prompt(text, rate=self.rate, force_tokens=['\n', '`', ':'])
        return out['compressed_prompt']

    def compress_memory_prose(self, memory: str) -> str:
        """只压缩经验块中的自然语言：规则（Rule:）与示例题目描述（Example task:）的正文。
        标题行、Action 行、Final Answer 行、代码块、修复经验中的错误信息与 SQL 一律原样保留。"""
        out, fenced = [], False
        for line in memory.split('\n'):
            if line.lstrip().startswith('```'):
                fenced = not fenced
                out.append(line)
                continue
            prefix = next((p for p in PROSE_PREFIXES if line.startswith(p)), None)
            if fenced or prefix is None or not line[len(prefix):].strip():
                out.append(line)
            else:
                out.append(prefix + ' ' + self._compress(line[len(prefix):].strip()).strip())
        return '\n'.join(out)

    def start(self, task, features, rng) -> None:
        super().start(task, features, rng)
        units = self.full_units()
        memory = render_memory(units, self.store)
        if self.scope == 'memory':
            self.fixed = Plan(prefix=units, memory_text=self.compress_memory_prose(memory) if memory else None)
            return
        self.fixed = Plan(prefix=units, system_text=self._compress(render_system(ALL_IDS, task)),
                          schema_text=self._compress(prompt_schema_text(task)),
                          memory_text=self._compress(memory) if memory else None)

    def plan(self, state: TurnState) -> Plan:
        return self.fixed
