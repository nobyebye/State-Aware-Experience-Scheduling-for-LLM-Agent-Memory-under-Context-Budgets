"""输入 token 计数（Llama-3.1 分词器）。

一次请求的输入 token = 各消息正文 token 之和 + 每条消息 5 个格式 token + 5 个固定 token
（begin_of_text 与 assistant 消息头）。第一步问题分析中，这一计数方法在 709 次请求上
与服务商返回的 prompt_tokens 完全一致；正式实验中两者都会记录，并在分析时再次核对。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .data import DATA_DIR

DEFAULT_TOKENIZER = DATA_DIR / 'tokenizer' / 'llama31_8b_tokenizer.json'
PER_MESSAGE = 5
PER_REQUEST = 5


class TokenCounter:
    def __init__(self, path: Path = DEFAULT_TOKENIZER):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(str(path))
        self.count = lru_cache(maxsize=200_000)(self._count)

    def _count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tok.encode(text, add_special_tokens=False).ids)

    def count_messages(self, messages: list[dict]) -> int:
        return sum(self.count(m['content']) for m in messages) + PER_MESSAGE * len(messages) + PER_REQUEST
