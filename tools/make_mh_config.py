"""Write this run's copy of WHALE's chess meta-harness config.

Changes from domains/chess_puzzles/meta_harness/config-chess-puzzle.json:
  - the solver endpoint points at this job's port instead of the fixed 8044,
    so two jobs on one shared node cannot collide;
  - the proposer is the rung's model behind LiteLLM ("ladder-proposer");
  - proposer_effort is null, so no Anthropic-only --effort flag is sent;
  - proposals_per_iter comes from the profile.
Everything else (puzzles, sampling, turn limits, metric) is left as released.

Usage: python make_mh_config.py SOURCE_JSON OUT_JSON
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    config = json.loads(src.read_text())
    meta = config.setdefault("meta", {})
    meta["proposer_model"] = "ladder-proposer"
    meta["proposer_effort"] = None
    meta["proposals_per_iter"] = int(os.environ.get("PROPOSALS_PER_ITER", meta.get("proposals_per_iter", 3)))
    meta["propose_timeout_s"] = int(os.environ.get("PROPOSER_TIMEOUT", meta.get("propose_timeout_s", 2400)))
    for model in config.get("models", []):
        model["api_base"] = f"http://127.0.0.1:{os.environ['SOLVER_PORT']}/v1"
        model["model"] = os.environ["SOLVER_SERVED_NAME"]
        model["api_key_env"] = "LADDER_SOLVER_KEY"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
