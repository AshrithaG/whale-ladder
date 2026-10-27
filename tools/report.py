"""Turn one smoke run into measured unit costs and a budget for the real runs.

Reads what the run already writes:
  CKPT_DIR/alternation.log      phase start/end times (WHALE's driver)
  CKPT_DIR/logs/*.log           verl step metrics (timing_s/step, val scores)
  RUN_OUT/timing.jsonl          hook timings (checkpoint merge, server start/stop)
  RUN_OUT/proposer_cost.jsonl   one row per proposer session: time, tokens, USD
  MH_RUNS/mh-RUN_NAME-round*/   candidate harnesses and their scores

Writes RUN_OUT/report.json and RUN_OUT/report.md. The extrapolation is linear:
weight-phase step time is scaled by batch size, harness-phase costs by the
number of iterations and candidates. Treat it as a planning number, and replace
it once a real multi-cycle run exists.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (.*)$")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def alternation_phases(log: Path) -> list[dict]:
    """[{cycle, weight_s, harness_s}] from alternate.sh's timestamped log."""
    if not log.exists():
        return []
    events = []
    for line in log.read_text().splitlines():
        m = TS.match(line)
        if m:
            events.append((datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp(), m.group(2).strip()))
    cycles, cur = [], None
    for t, msg in events:
        if msg.startswith("--- cycle"):
            cur = {"cycle": int(msg.split()[2]), "t_cycle": t}
            cycles.append(cur)
        elif cur is None:
            continue
        elif msg.startswith("weight-update phase"):
            cur["t_weight"] = t
        elif msg.startswith("checkpoint already present"):
            cur["weight_skipped"] = True
        elif msg.startswith("harness-search phase") and "already complete" not in msg:
            cur["t_harness"] = t
        elif msg.startswith("accepted harness:"):
            cur["t_accepted"] = t
            cur["accepted"] = msg.split(":", 1)[1].strip()
        elif msg.startswith("budget exhausted") or msg.startswith("=== done"):
            cur.setdefault("t_end", t)
    out = []
    for c in cycles:
        row = {"cycle": c["cycle"], "accepted": c.get("accepted")}
        if "t_weight" in c:
            end = c.get("t_harness", c.get("t_end"))
            row["weight_s"] = (end - c["t_weight"]) if end else None
        if "t_harness" in c and "t_accepted" in c:
            row["harness_s"] = c["t_accepted"] - c["t_harness"]
        out.append(row)
    return out


def verl_metrics(log_dir: Path) -> dict:
    step_times, vals = [], {}
    for log in sorted(log_dir.glob("*.log")):
        text = log.read_text(errors="replace")
        step_times += [float(x) for x in re.findall(r"timing_s/step:([0-9.]+)", text)]
        for key, value in re.findall(r"(val-core/[^\s:]+):([0-9.]+)", text):
            vals[key] = float(value)
    return {"step_times_s": step_times, "val": vals}


def hook_seconds(timing: list[dict]) -> dict:
    def spans(start: str, end: str) -> list[float]:
        starts = [r["t"] for r in timing if r.get("event") == start]
        ends = [r["t"] for r in timing if r.get("event") == end]
        return [e - s for s, e in zip(starts, ends)]

    merge = []
    for r in timing:
        if r.get("event") == "merge_done":
            prior = [x["t"] for x in timing if x.get("event") == "pre_harness_start" and x["t"] <= r["t"]]
            if prior:
                merge.append(r["t"] - prior[-1])
    return {
        "pre_hook_s": spans("pre_harness_start", "pre_harness_end"),
        "merge_s": merge,
        "post_hook_s": spans("post_harness_start", "post_harness_end"),
    }


def candidate_scores(mh_runs: Path, run_name: str) -> list[dict]:
    rows = []
    for summary in sorted(mh_runs.glob(f"mh-{run_name}-round*/logs/evolution_summary.jsonl")):
        rows += read_jsonl(summary)
    return rows


def mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--run-out", required=True, type=Path)
    ap.add_argument("--ckpt-dir", required=True, type=Path)
    ap.add_argument("--mh-runs", required=True, type=Path)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--smoke-batch", type=int, default=64)
    ap.add_argument("--smoke-steps", type=int, default=2)
    ap.add_argument("--smoke-proposals", type=int, default=3)
    # Plan to extrapolate to. Defaults: WHALE's chess batch, its best (0.2, 6)
    # schedule per the author's issue #2 (13 steps = 0.2 epoch, 6 iterations),
    # 4 cycles, 3 candidates per iteration, 5 arms, 1 seed.
    ap.add_argument("--plan-batch", type=int, default=256)
    ap.add_argument("--plan-round-steps", type=int, default=13)
    ap.add_argument("--plan-cycles", type=int, default=4)
    ap.add_argument("--plan-mh-iters", type=int, default=6)
    ap.add_argument("--plan-proposals", type=int, default=3)
    ap.add_argument("--plan-whale-arms", type=int, default=4)
    ap.add_argument("--plan-seeds", type=int, default=1)
    a = ap.parse_args()

    phases = alternation_phases(a.ckpt_dir / "alternation.log")
    verl = verl_metrics(a.ckpt_dir / "logs")
    hooks = hook_seconds(read_jsonl(a.run_out / "timing.jsonl"))
    ledger = read_jsonl(a.run_out / "proposer_cost.jsonl")
    cands = candidate_scores(a.mh_runs, a.run_name)

    weight_wall = mean([p.get("weight_s") for p in phases])
    harness_wall = mean([p.get("harness_s") for p in phases])
    step_s = mean(verl["step_times_s"]) or (weight_wall / a.smoke_steps if weight_wall else None)
    weight_overhead = (weight_wall - a.smoke_steps * step_s) if (weight_wall and step_s) else None
    proposer_s = mean([r.get("duration_s") for r in ledger])
    pre_s, post_s = mean(hooks["pre_hook_s"]) or 0.0, mean(hooks["post_hook_s"]) or 0.0
    # Scoring = harness window minus hooks and proposer time, shared over the
    # incoming harness plus the candidates.
    score_per_cand = None
    if harness_wall and proposer_s is not None:
        scoring = harness_wall - pre_s - post_s - sum(r.get("duration_s", 0) for r in ledger)
        score_per_cand = max(scoring, 0) / (a.smoke_proposals + 1)

    measured = {
        "weight_phase_wall_s": weight_wall,
        "rsft_step_s_at_smoke_batch": step_s,
        "weight_phase_fixed_overhead_s": weight_overhead,
        "harness_phase_wall_s": harness_wall,
        "pre_hook_s": pre_s,
        "checkpoint_merge_s": mean(hooks["merge_s"]),
        "post_hook_s": post_s,
        "proposer_session_s": proposer_s,
        "scoring_s_per_harness_256_puzzles": score_per_cand,
        "proposer_sessions": len(ledger),
        "proposer_failed_sessions": sum(1 for r in ledger if r.get("exit_code") not in (0, None)),
        "proposer_tool_errors": sum(r.get("n_tool_errors", 0) for r in ledger),
        "proposer_input_tokens": sum(r.get("input_tokens", 0) for r in ledger),
        "proposer_cached_input_tokens": sum(r.get("cached_input_tokens", 0) for r in ledger),
        "proposer_output_tokens": sum(r.get("output_tokens", 0) for r in ledger),
        "proposer_usd": round(sum(r.get("cost_usd", 0.0) for r in ledger), 4),
        "val_metrics": verl["val"],
        "cycles": phases,
        "candidates": cands,
    }

    plan = None
    if step_s and weight_overhead is not None:
        weight_cycle = weight_overhead + a.plan_round_steps * step_s * (a.plan_batch / a.smoke_batch)
        harness_cycle = None
        if score_per_cand is not None and proposer_s is not None:
            harness_cycle = pre_s + post_s + score_per_cand + a.plan_mh_iters * (
                proposer_s + a.plan_proposals * score_per_cand
            )
        whale_arm_h = a.plan_cycles * (weight_cycle + (harness_cycle or 0)) / 3600 * a.gpus
        rsft_arm_h = a.plan_cycles * weight_cycle / 3600 * a.gpus
        usd_per_session = (measured["proposer_usd"] / len(ledger)) if ledger else 0.0
        plan = {
            "assumptions": vars(a) | {"run_out": str(a.run_out), "ckpt_dir": str(a.ckpt_dir), "mh_runs": str(a.mh_runs)},
            "weight_phase_s_per_cycle": round(weight_cycle),
            "harness_phase_s_per_cycle": round(harness_cycle) if harness_cycle else None,
            "gpu_hours_one_whale_arm": round(whale_arm_h, 1),
            "gpu_hours_rsft_only_arm": round(rsft_arm_h, 1),
            "gpu_hours_total": round((a.plan_whale_arms * whale_arm_h + rsft_arm_h) * a.plan_seeds, 1),
            "proposer_usd_one_arm_this_rung": round(usd_per_session * a.plan_cycles * a.plan_mh_iters, 2),
            "note": "Every whale arm is priced with THIS arm's harness costs; a 27B or self arm "
                    "has different proposer speed, so smoke-test one local rung too before trusting the total.",
        }

    report = {"run": a.run_name, "measured": measured, "plan": plan}
    (a.run_out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")

    def fmt(x, unit=""):
        if x is None:
            return "n/a"
        return f"{x:,.1f}{unit}" if isinstance(x, float) else f"{x}{unit}"

    lines = [f"# Smoke report: {a.run_name}", "", "## Measured", ""]
    for key in ("weight_phase_wall_s", "rsft_step_s_at_smoke_batch", "weight_phase_fixed_overhead_s",
                "harness_phase_wall_s", "checkpoint_merge_s", "pre_hook_s", "proposer_session_s",
                "scoring_s_per_harness_256_puzzles", "proposer_sessions", "proposer_failed_sessions",
                "proposer_tool_errors", "proposer_input_tokens", "proposer_output_tokens", "proposer_usd"):
        lines.append(f"- {key}: {fmt(measured[key])}")
    if measured["val_metrics"]:
        lines += ["", "Validation after the weight phase:"] + [f"- {k}: {v}" for k, v in measured["val_metrics"].items()]
    if cands:
        lines += ["", "Candidates (success rate on the 256 harness-search puzzles):"]
        lines += [f"- {c.get('harness')}: {c.get('avg_success_rate')} (axis {c.get('axis')}, status {c.get('status')})"
                  for c in cands]
    lines += ["", "## Extrapolated plan", ""]
    if plan:
        for key, value in plan.items():
            if key != "assumptions":
                lines.append(f"- {key}: {value}")
    else:
        lines.append("Not enough measurements to extrapolate (did both phases finish?).")
    (a.run_out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
