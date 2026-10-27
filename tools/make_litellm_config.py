"""Print the LiteLLM proxy config for the current rung.

Claude Code speaks the Anthropic Messages API; LiteLLM translates it to the
backend's API. Every model name Claude Code asks for ("*") is routed to the one
proposer model of this rung, so background calls (titles, summaries) cannot
reach a different model than the one being tested.

Reads: LADDER_RUNG, PROPOSER_BACKEND, PROPOSER_PORT, LITELLM_MASTER_KEY,
DEEPSEEK_PROVIDER, FIREWORKS_DEEPSEEK_MODEL, DEEPSEEK_MODEL.
"""

from __future__ import annotations

import json
import os
import sys


def backend_params() -> dict:
    backend = os.environ["PROPOSER_BACKEND"]
    if backend == "local":
        return {
            "model": "hosted_vllm/proposer",
            "api_base": f"http://127.0.0.1:{os.environ['PROPOSER_PORT']}/v1",
            "api_key": "EMPTY",
        }
    provider = os.environ.get("DEEPSEEK_PROVIDER", "fireworks")
    if provider == "fireworks":
        model_id = os.environ.get("FIREWORKS_DEEPSEEK_MODEL", "").strip()
        if not model_id:
            sys.exit("FIREWORKS_DEEPSEEK_MODEL is not set (the model id shown on fireworks.ai/models)")
        return {"model": f"fireworks_ai/{model_id}", "api_key": "os.environ/FIREWORKS_API_KEY"}
    if provider == "deepseek":
        return {
            "model": f"deepseek/{os.environ.get('DEEPSEEK_MODEL', 'deepseek-flash')}",
            "api_key": "os.environ/DEEPSEEK_API_KEY",
        }
    sys.exit(f"unknown DEEPSEEK_PROVIDER={provider}")


def main() -> None:
    params = backend_params()
    config = {
        "model_list": [
            {"model_name": "ladder-proposer", "litellm_params": params},
            {"model_name": "*", "litellm_params": params},
        ],
        # Claude Code sends Anthropic-only fields (e.g. cache_control, effort);
        # drop what the backend does not accept instead of failing the call.
        "litellm_settings": {"drop_params": True, "num_retries": 3, "request_timeout": 900},
        "general_settings": {"master_key": os.environ["LITELLM_MASTER_KEY"]},
    }
    # JSON is valid YAML, and avoids a PyYAML dependency.
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
