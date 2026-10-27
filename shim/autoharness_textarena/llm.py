from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from .config import LLMConfig


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class LLMResponse:
    content: str
    usage: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    """Chat-completions client for the solver's vLLM server.

    The content is returned exactly as generated. The solver server runs without
    a reasoning parser, so any thinking text stays in `content`, matching what
    the verl rollout workers see during training.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        api_key = os.environ.get(config.api_key_env or "", "") or "EMPTY"
        self._client = OpenAI(base_url=config.api_base, api_key=api_key, timeout=config.timeout_s)

    def _extra_body(self) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if self.config.top_k is not None:
            extra["top_k"] = self.config.top_k
        if self.config.chat_template_kwargs:
            extra["chat_template_kwargs"] = dict(self.config.chat_template_kwargs)
        return extra

    def complete_response(self, messages: list[ChatMessage], max_tokens: int | None = None) -> LLMResponse:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        attempts = max(1, int(self.config.max_retries))
        for attempt in range(1, attempts + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=payload,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_tokens=int(max_tokens or self.config.max_tokens),
                    extra_body=self._extra_body(),
                )
            except (APIConnectionError, APITimeoutError, RateLimitError):
                if attempt == attempts:
                    raise
                time.sleep(min(60.0, 2.0**attempt) + random.random())
                continue
            except APIStatusError as exc:
                if exc.status_code < 500 or attempt == attempts:
                    raise
                time.sleep(min(60.0, 2.0**attempt) + random.random())
                continue
            choice = resp.choices[0]
            usage = resp.usage.model_dump() if resp.usage is not None else {}
            return LLMResponse(content=choice.message.content or "", usage=usage)
        raise RuntimeError("unreachable")

    def complete(self, messages: list[ChatMessage], max_tokens: int | None = None) -> str:
        return self.complete_response(messages, max_tokens=max_tokens).content
