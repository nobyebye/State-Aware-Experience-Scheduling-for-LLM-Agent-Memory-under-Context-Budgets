"""模型调用：OpenAI 兼容的 chat/completions 接口（默认 OpenRouter），只用标准库。

ScriptedLLM 按预先给定的函数生成回复，用于离线测试与不花钱的流程演练。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    raw: dict = field(default_factory=dict)


class LLMError(RuntimeError):
    pass


class OpenAICompatibleLLM:
    def __init__(self, model: str, api_base: str, api_key_env: str, temperature: float = 0.2,
                 top_p: float = 1.0, max_tokens: int = 2048, timeout: float = 120,
                 extra_body: Optional[dict] = None, max_retries: int = 5, seed: Optional[int] = None):
        self.model, self.temperature, self.top_p, self.max_tokens = model, temperature, top_p, max_tokens
        self.url = api_base.rstrip('/') + '/chat/completions'
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise LLMError(f'环境变量 {api_key_env} 未设置')
        self.timeout, self.extra_body, self.max_retries, self.seed = timeout, extra_body or {}, max_retries, seed

    def chat(self, messages: list[dict]) -> LLMResponse:
        body = {'model': self.model, 'messages': messages, 'temperature': self.temperature,
                'top_p': self.top_p, 'max_tokens': self.max_tokens, 'usage': {'include': True},
                **self.extra_body}
        if self.seed is not None:
            body['seed'] = self.seed
        data = json.dumps(body).encode('utf-8')
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.url, data=data, method='POST', headers={
                'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode('utf-8'))
                if 'error' in payload:
                    raise LLMError(str(payload['error']))
                choice = payload['choices'][0]
                usage = payload.get('usage') or {}
                details = usage.get('prompt_tokens_details') or {}
                return LLMResponse(content=choice['message'].get('content') or '',
                                   prompt_tokens=usage.get('prompt_tokens'),
                                   completion_tokens=usage.get('completion_tokens'),
                                   cached_tokens=details.get('cached_tokens'),
                                   model=payload.get('model'), provider=payload.get('provider'),
                                   raw={'id': payload.get('id'), 'finish_reason': choice.get('finish_reason')})
            except urllib.error.HTTPError as e:
                if e.code not in (408, 429, 500, 502, 503, 504) or attempt == self.max_retries:
                    raise LLMError(f'HTTP {e.code}: {e.read()[:500]!r}') from e
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, LLMError) as e:
                if attempt == self.max_retries:
                    raise LLMError(str(e)) from e
            time.sleep(delay)
            delay *= 2
        raise LLMError('unreachable')


class ScriptedLLM:
    """reply_fn(messages) -> 回复文本。不产生费用，prompt_tokens 为空。"""

    def __init__(self, reply_fn: Callable[[list[dict]], str], model: str = 'scripted'):
        self.reply_fn, self.model = reply_fn, model

    def chat(self, messages: list[dict]) -> LLMResponse:
        return LLMResponse(content=self.reply_fn(messages), model=self.model)
