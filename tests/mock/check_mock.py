"""PASS/FAIL checks after ladder/mock_run.sh has run every arm.

    python tests/mock/check_mock.py CYCLES_ROOT MOCK_STATE_DIR

Checks what the real runs depend on and what broke before: every arm finishes
every cycle and the final test; checkpoints and accepted harnesses are handed
from phase to phase; bad proposals are recorded and skipped, never crashing a
run; each rung's proposer is the model it should be; an interrupted run
resumes without redoing finished phases; nothing is left running.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

root, state = Path(sys.argv[1]), Path(sys.argv[2])
ARMS = ["rsft_only", "whale_deepseek", "whale_qwen27b", "whale_qwen4b", "whale_self"]
CYCLES = 3
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'' if ok or not detail else ': ' + detail}")


def lineage(arm_dir: Path, c: int, steps: int) -> dict:
    run = next((arm_dir / f"c{c}-train" / "checkpoints").iterdir())
    return json.loads((run / f"global_step_{c * steps}" / "mock_lineage.json").read_text())


for arm in ARMS:
    arm_dir = root / f"{arm}-mock-s7"
    res_path = arm_dir / "results.json"
    if not res_path.exists():
        check(f"{arm}: finished with results.json", False, f"see {state / (arm + '.log')}")
        continue
    res = json.loads(res_path.read_text())
    cycles = res["cycles"]
    check(f"{arm}: {CYCLES} cycles trained", len(cycles) == CYCLES and all(c["train_passed"] for c in cycles),
          f"{len(cycles)} cycles")
    check(f"{arm}: final test scored", bool(res["final_test"] and res["final_test"]["scores"]))
    check(f"{arm}: hand-offs consistent (summary found no problems)", not res["problems"], "; ".join(res["problems"]))
    # Each cycle resumed from the previous cycle's last checkpoint.
    chain_ok = True
    for c in range(2, CYCLES + 1):
        run = next((arm_dir / f"c{c}-train" / "checkpoints").iterdir())
        step1 = json.loads((run / f"global_step_{(c - 1) * 2 + 1}" / "mock_lineage.json").read_text())
        if not str(step1["resumed_from"]).endswith(f"c{c - 1}-train/checkpoints/{run.name}/global_step_{(c - 1) * 2}"):
            chain_ok = False
    check(f"{arm}: each cycle resumed from the previous cycle's checkpoint", chain_ok)

    if arm == "rsft_only":
        check("rsft_only: never searched", all("search" not in c for c in cycles))
        continue

    searches = [c["search"] for c in cycles]
    check(f"{arm}: every search phase passed", all(s["passed"] for s in searches))
    # The accepted harness is the best of the incoming one and the valid candidates.
    sel_ok = True
    for s in searches:
        scores = [x["score"] for x in s["candidates"] if x["score"] is not None]
        best = max([s["incoming_harness_score"] or 0.0] + scores)
        acc = s["accepted"]
        acc_score = s["incoming_harness_score"] if acc == "h0" else next(
            (x["score"] for x in s["candidates"] if x["harness"] == acc), None)
        if acc_score is None or acc_score < best - 1e-9:
            sel_ok = False
    check(f"{arm}: accepted harness is the best scored one every cycle", sel_ok)
    # The next cycle trained with exactly the harness this cycle accepted.
    for c in range(1, CYCLES):
        trained = lineage(arm_dir, c + 1, 2)["harness_sha"]
        accepted = cycles[c - 1]["search"]["accepted_harness"]
        check(f"{arm}: cycle {c + 1} trained with cycle {c}'s accepted harness", trained == accepted,
              f"{trained} != {accepted}")

    # Server logs from every search phase, plus the arm's driver output, where
    # WHALE's search loop prints its recoveries and rejections.
    log_files = list(arm_dir.glob(f"{arm}-c*-search/*.log")) + [state / f"{arm}.log"]
    logs = "\n".join(p.read_text(errors="replace") for p in log_files if p.exists())
    rejected = sum(s["rejected"] or 0 for s in searches)
    reasons = " | ".join(x["reason"] for s in searches for x in s["candidates"] if x["status"] != "evaluated")
    if arm == "whale_deepseek":
        check("deepseek: proposer routed to the API backend", "routing ladder-proposer -> fireworks_ai/" in logs)
        check("deepseek: no local proposer server", "served_name=proposer" not in logs)
        check("deepseek: proposer cost recorded at API prices", sum(s["proposer_usd"] for s in searches) > 0)
        check("deepseek: no rejected proposals", rejected == 0, reasons)
    if arm == "whale_qwen27b":
        check("qwen27b: proposer server is the 27B model", re.search(r"model=\S*27B\S* as served_name=proposer", logs) is not None)
        check("qwen27b: 'slot' instead of 'name' accepted, nothing rejected", rejected == 0, reasons)
    if arm == "whale_qwen4b":
        check("qwen4b: proposer server is the base 4B model",
              re.search(r"model=\S*Qwen3\.5-4B\S* as served_name=proposer", logs) is not None)
        for label, pattern in [("broken Python", r"SyntaxError|invalid syntax|was never closed|expected"),
                               ("missing function", r"Missing required function"),
                               ("unassigned slot", r"not one of this iteration's slots")]:
            check(f"qwen4b: {label} rejected and recorded", re.search(pattern, reasons) is not None, reasons)
        check("qwen4b: unreadable manifest recovered from harness files", "pending_eval.json unreadable" in logs)
        cyc_log = (arm_dir / "cycles.log").read_text()
        check("qwen4b: injected training failure was set aside", "moved unfinished" in cyc_log)
        check("qwen4b: rerun resumed without redoing finished phases", "training already passed" in cyc_log)
    if arm == "whale_self":
        served = re.findall(r"model=(\S+) as served_name=proposer", logs)
        check("self: proposer served the checkpoint being searched",
              bool(served) and all(re.search(r"global_step_\d+/hf$", m) for m in served), str(served))
        check("self: a different checkpoint each cycle", len(set(served)) == CYCLES, str(served))
        check("self: empty and crashed proposals kept the incoming harness", "no valid candidates" in logs)

# Layer-2 tool, run by mock_run.sh on the deepseek arm's first search run.
for arm, want in [("whale_deepseek", "all valid"), ("whale_qwen4b", "some rejected")]:
    path = root.parent / f"{arm}-mock-s7" / "format_check" / "format_check.json"
    if not path.exists():
        check(f"format check {arm}: ran", False, f"see {state / ('format-' + arm + '.log')}")
        continue
    fc = json.loads(path.read_text())
    rejected = sum(len(s["invalid"]) for s in fc["sessions"])
    ok = fc["valid"] == fc["proposals_requested"] if want == "all valid" else (rejected > 0 and fc["valid"] > 0)
    check(f"format check {arm}: {want} ({fc['valid']}/{fc['proposals_requested']} valid)", ok)

left = subprocess.run(["pgrep", "-f", str(Path(__file__).parent / "bin")], capture_output=True, text=True).stdout.split()
check("no mock servers left running", not left, f"pids {left}")

failed = results.count(False)
print(f"\n{len(results) - failed}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
