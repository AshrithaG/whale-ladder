"""Summarize one arm's run (ladder/run_cycles.sh) into results.json and a table.

    python tools/summarize_cycles.py STATE_DIR --arm ARM --profile PROFILE

Per cycle: training steps (rollouts, successful rollouts used for SFT), the
harness training used, and for proposer arms the incoming harness's score,
every candidate's score or rejection reason, and the accepted harness. Then the
final held-out test score. It also checks the hand-offs: each cycle trained with
the harness the previous cycle accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else None


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def score_of(val_dir: Path, harness: str) -> float | None:
    """Success rate WHALE recorded for one harness in a search run's logs."""
    for val in sorted(val_dir.glob(f"*/{harness}/*/val.json")):
        return float(json.loads(val.read_text()).get("success_rate", 0.0))
    return None


def val_score(metrics: dict) -> dict[str, float]:
    return {k: v for k, v in metrics.items() if k.startswith("val-core/") and isinstance(v, (int, float))}


def timing_report(state):
    phases = jsonl(state / 'phase-timing.jsonl')
    details = {}
    for directory in sorted(state.glob('c*-train')) + [state / 'final-test']:
        steps = {}
        for row in jsonl(directory / 'training-metrics.jsonl'):
            steps.setdefault(str(row['step']), {}).update({k: v for k, v in row.get('data', {}).items()
                if k.startswith('timing_s/') or k in ('actor/sft_time', 'actor/sft_updates',
                    'online_rsft/total', 'online_rsft/accepted')})
        if steps:
            details[directory.name] = steps
    search = {}
    for directory in sorted(state.glob('*-search')):
        search[directory.name] = dict(events=jsonl(directory / 'timing.jsonl'),
                                      proposer_sessions=jsonl(directory / 'proposer_cost.jsonl'))
    starts = [r['started_unix'] for r in phases]
    ends = [r['ended_unix'] for r in phases]
    wall = max(ends) - min(starts) if phases else None
    measured = sum(r['elapsed_seconds'] for r in phases)
    return dict(controller_wall_seconds=wall, sequential_phase_seconds=measured,
        controller_gap_seconds=max(0, wall - measured) if wall is not None else None,
        phases=phases, trainer_details=details, search_details=search,
        note='Sequential controller phases can be summed. Trainer and search details are nested and may overlap; do not add them to phase totals. Candidate attempt execution includes model calls and reward checking.')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("state", type=Path)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--profile", required=True)
    a = ap.parse_args()
    state = a.state

    cycles, problems = [], []
    c = 1
    while (state / f"c{c}-train").exists():
        train = state / f"c{c}-train"
        rows = jsonl(train / "training-metrics.jsonl")
        steps = [
            {"step": r["step"], "rollouts": r["data"].get("online_rsft/total"),
             "successful": r["data"].get("online_rsft/accepted"),
             "sft_updates": r["data"].get("actor/sft_updates"), "val": val_score(r["data"])}
            for r in rows if "online_rsft/total" in r.get("data", {})
        ]
        cycle = {"cycle": c, "train_passed": (train / "passed.json").exists(), "steps": steps,
                 "trained_with_harness": sha(state / f"c{c}-trained-with.py")}
        search = state / f"{a.arm}-c{c}-search"
        if search.exists():
            passed = json.loads((search / "passed.json").read_text()) if (search / "passed.json").exists() else {}
            run_dir = Path((search / "harness-run-dir").read_text().strip()) if (search / "harness-run-dir").exists() else None
            evo = jsonl(run_dir / "logs" / "evolution_summary.jsonl") if run_dir else []
            cycle["search"] = {
                "passed": bool(passed),
                "incoming_harness_score": score_of(run_dir / "logs", "h0") if run_dir else None,
                "candidates": [
                    {"harness": r.get("harness"), "status": r.get("status"),
                     "score": r.get("avg_success_rate") if r.get("status") == "evaluated" else None,
                     "reason": r.get("reason", "")}
                    for r in evo
                ],
                "evaluated": passed.get("candidates_evaluated"),
                "rejected": passed.get("candidates_rejected"),
                "accepted": passed.get("accepted_harness"),
                "accepted_harness": sha(state / f"c{c}-accepted.py"),
                "proposer_sessions": len(jsonl(search / "proposer_cost.jsonl")),
                "proposer_usd": round(sum(r.get("cost_usd", 0.0) for r in jsonl(search / "proposer_cost.jsonl")), 4),
            }
            nxt = state / f"c{c + 1}-trained-with.py"
            if nxt.exists() and sha(nxt) != cycle["search"]["accepted_harness"]:
                problems.append(f"cycle {c + 1} trained with a different harness than cycle {c} accepted")
        cycles.append(cycle)
        c += 1

    final = state / "final-test" / "passed.json"
    test = None
    if final.exists():
        data = json.loads(final.read_text())
        scores = {}
        for m in (data.get("metrics") or {}).values():
            scores.update(val_score(m))
        test = {"rollouts": data.get("validation_rollouts"), "scores": scores,
                "harness": sha(state / "final-harness.py")}
    elif cycles:
        problems.append("final test missing")
    if (state / "final-harness.py").exists() and cycles and a.arm != "rsft_only":
        last = cycles[-1].get("search", {}).get("accepted_harness")
        if last and sha(state / "final-harness.py") != last:
            problems.append("final test used a different harness than the last cycle accepted")

    timings = timing_report(state)
    (state / 'timing-report.json').write_text(json.dumps(timings, indent=2) + '\n')
    result = {"arm": a.arm, "profile": a.profile, "cycles": cycles, "final_test": test, "problems": problems,
              'timings': timings}
    (state / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    print(f"\n{a.arm} / {a.profile}")
    for cy in cycles:
        tr = ", ".join(f"step {s['step']}: {s['successful']}/{s['rollouts']}" for s in cy["steps"]) or "no metrics"
        print(f"  cycle {cy['cycle']}: train [{tr}]")
        s = cy.get("search")
        if s:
            cands = "; ".join(
                f"{x['harness']} {x['score']:.3f}" if x["score"] is not None else f"{x['harness']} {x['status']}"
                for x in s["candidates"]
            )
            h0 = f"{s['incoming_harness_score']:.3f}" if s["incoming_harness_score"] is not None else "n/a"
            print(f"           search: incoming {h0}; {cands or 'no candidates'}; accepted {s['accepted']}")
    print(f"  final test: {test['scores'] if test else 'missing'}")
    for p in problems:
        print(f"  PROBLEM: {p}")
    if timings['phases']:
        print('  Controller phase timings (seconds):')
        for phase in timings['phases']:
            print(f"    {phase['phase']}: {phase['elapsed_seconds']:.1f} ({phase['status']})")


if __name__ == "__main__":
    main()
