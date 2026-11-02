"""Layer 2: can this rung's proposer write valid candidates? No training, no scoring.

    python tools/proposer_format_check.py SAVED_RUN_DIR OUT_DIR SESSIONS

SAVED_RUN_DIR is a finished WHALE search run directory (harnesses/, logs/,
e.g. Sumanth's B1 search records). For each session it copies that directory,
asks the proposer for one iteration of candidates with exactly the real
search's prompt, then applies the real search's validation rules
(patches/candidate-validation.patch) and records the outcome. Run from the
chess domain directory with ladder/arm_env.sh loaded; ladder/
proposer_format_check.sh does both and starts the proposer's servers.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder_prompt  # noqa: E402
from meta_harness import meta_harness_chess_puzzle as mh  # noqa: E402

saved, out, sessions = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), int(sys.argv[3])
assert (saved / "logs" / "accepted_harness.txt").exists(), f"not a WHALE search run directory: {saved}"
proposals = int(os.environ.get("PROPOSALS_PER_ITER", "3"))
rows = []
# Each session wraps WHALE's prompt for its own copy of the run directory.
original_render, original_system = mh.render_task_prompt, mh.PROPOSER_SYSTEM_PROMPT
for n in range(1, sessions + 1):
    run_dir = out / f"session{n}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(saved, run_dir, ignore=shutil.ignore_patterns("claude_sessions", "pending_eval.json"))
    mh.render_task_prompt, mh.PROPOSER_SYSTEM_PROMPT = original_render, original_system
    ladder_prompt.install(mh, run_dir)
    names = mh.next_harness_names(run_dir, proposals)
    iter_dir = run_dir / "logs" / "format_check"
    iter_dir.mkdir(parents=True, exist_ok=True)
    prompt = mh.render_task_prompt(1, names)
    (iter_dir / "proposer_prompt.txt").write_text(prompt)
    started = time.monotonic()
    result = mh.propose_claude_with_retries(
        task_prompt=prompt, iteration=1, run_dir=run_dir, next_names=names,
        proposer_model="ladder-proposer", proposer_effort=None,
        timeout_seconds=int(os.environ.get("PROPOSER_TIMEOUT", "1200")), use_api_key=False)
    candidates = mh.read_pending_candidates(run_dir, names, iter_dir)
    valid, _, invalid, _ = mh.validate_proposals(run_dir, candidates, names, 1)
    rows.append({
        "session": n, "exit_code": result.exit_code, "seconds": round(time.monotonic() - started, 1),
        "slots": names, "valid": valid, "invalid": [{"harness": r["harness"], "reason": r["reason"]} for r in invalid],
        "manifest_written": (iter_dir / "pending_eval.json").exists(),
        "tool_calls": len(result.tool_calls), "tool_errors": sum(1 for t in result.tool_calls if t.is_error),
        "tokens": result.token_usage,
    })
    r = rows[-1]
    print(f"session {n}: exit {r['exit_code']}, {len(valid)}/{proposals} valid, manifest "
          f"{'yes' if r['manifest_written'] else 'no'}, {r['seconds']}s"
          + "".join(f"\n    rejected {x['harness']}: {x['reason']}" for x in r["invalid"]), flush=True)

total_valid = sum(len(r["valid"]) for r in rows)
summary = {"rung": os.environ.get("LADDER_RUNG"), "sessions": rows, "proposals_requested": proposals * sessions,
           "valid": total_valid, "valid_rate": round(total_valid / (proposals * sessions), 3)}
(out / "format_check.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"\n{summary['rung']}: {total_valid}/{proposals * sessions} candidates valid ({summary['valid_rate']:.0%}); "
      f"details in {out / 'format_check.json'}")
