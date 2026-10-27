from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMConfig:
    """Fields read from the `models` entries of WHALE's chess meta-harness config.

    runner._make_llm keeps only keys that are dataclass fields, so any field
    added here becomes configurable from that JSON.
    """

    provider: str = "openai-compatible"
    model: str = ""
    api_base: str = "http://127.0.0.1:8044/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    max_tokens: int = 8129
    batch_concurrency: int = 1
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    timeout_s: float = 900.0
    max_retries: int = 5
