from __future__ import annotations

from functools import lru_cache

from ..memory.store import render_memory
from ..prompts import ALL_IDS, render_system
from ..prompts import schema_text as prompt_schema_text
from .base import Plan, Policy, TurnState


class LLMLingua2Policy(Policy):
    """LLMLingua-2 token 级压缩基线：对可压缩内容（系统指令、表信息句与任务开始时检索的经验）按固定压缩率
    压缩，任务指令原样保留；每题压缩一次，各轮相同（官方布局）。压缩率在开发集上确定后冻结。
    需要额外安装 llmlingua 与 torch（见 requirements-llmlingua.txt）。"""

    def __init__(self, rate: float, model_name: str, revision: str | None = None, device: str = 'cpu'):
        from llmlingua import PromptCompressor
        self.rate = rate
        self.name = f'llmlingua2@{rate:g}'
        kwargs = {'model_name': model_name, 'use_llmlingua2': True, 'device_map': device}
        if revision:
            kwargs['model_config'] = {'revision': revision}
        self.compressor = PromptCompressor(**kwargs)
        self._compress = lru_cache(maxsize=4096)(self._compress_raw)

    def _compress_raw(self, text: str) -> str:
        out = self.compressor.compress_prompt(text, rate=self.rate, force_tokens=['\n', '`', ':'])
        return out['compressed_prompt']

    def start(self, task, features, rng) -> None:
        super().start(task, features, rng)
        units = self.full_units()
        memory = render_memory(units, self.store)
        self.fixed = Plan(prefix=units, system_text=self._compress(render_system(ALL_IDS, task)),
                          schema_text=self._compress(prompt_schema_text(task)),
                          memory_text=self._compress(memory) if memory else None)

    def plan(self, state: TurnState) -> Plan:
        return self.fixed
