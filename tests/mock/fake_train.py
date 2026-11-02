"""Fake verl training/evaluation run for the CPU mock (called via fake_train.sh).

Takes the same Hydra overrides ladder/hybrid_test.sh passes to the real
launcher and writes exactly what hybrid_test.sh's validator and
ladder/run_cycles.sh read: per-step metrics in VERL_FILE_LOGGER_PATH, rollout
and validation dumps, and global_step_N checkpoints with actor/ and data.pt.

What it checks for real: the harness handed to it (HARNESS_PATH) loads with
WHALE's own loader, and a resume checkpoint exists and was produced by the
previous phase. Each checkpoint records the harness and the checkpoint it came
from (mock_lineage.json) so the mock can verify every hand-off.

MOCK_FAIL_ONCE_TRAIN=<run name>:<target step> makes the first attempt at that
phase fail, to exercise resume.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path


def overrides(argv: list[str]) -> dict[str, str]:
    out = {}
    for arg in argv:
        if "=" in arg and not arg.startswith("~"):
            key, value = arg.split("=", 1)
            out[key.lstrip("+")] = value
    return out


def main() -> None:
    if "--cfg" in sys.argv:
        print("mock composed config")
        return
    o = overrides(sys.argv[1:])
    run = Path(os.environ["OUTPUT_ROOT"]) / os.environ["RUN_NAME"]
    run.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(os.environ["VERL_FILE_LOGGER_PATH"])
    harness = Path(os.environ["HARNESS_PATH"])

    from autoharness_chess_puzzle.harness import load_harness  # WHALE's loader, real check
    load_harness(harness)
    harness_sha = hashlib.sha256(harness.read_bytes()).hexdigest()[:12]
    skill = 0.1 + 0.5 * (int(harness_sha[:6], 16) % 1000) / 1000.0

    resume = o.get("trainer.resume_from_path")
    start = 0
    if resume:
        m = re.search(r"global_step_(\d+)$", resume)
        assert m, f"resume path must end in global_step_N: {resume}"
        start = int(m.group(1))
        assert (Path(resume) / "actor").is_dir() and (Path(resume) / "data.pt").is_file(), f"bad resume checkpoint {resume}"
        assert (Path(resume) / "mock_lineage.json").is_file(), "resume checkpoint was not produced by a previous phase"

    import pyarrow.parquet as pq
    val_puzzles = pq.read_table(os.environ["VAL_FILES"]).num_rows
    val_n = int(os.environ["VAL_ROLLOUT_N"])

    def write_val(step: int) -> float:
        (run / "val_rollouts").mkdir(exist_ok=True)
        rows = [json.dumps({"output": "<move>a1a2</move>", "assistant_generated_tokens_tokenized": 8, "score": 0})
                for _ in range(val_puzzles * val_n)]
        (run / "val_rollouts" / f"{step}.jsonl").write_text("\n".join(rows) + "\n")
        return round(skill + 0.01 * step, 4)

    def log_metrics(step: int, data: dict) -> None:
        with metrics_path.open("a") as fh:
            fh.write(json.dumps({"step": step, "data": data}) + "\n")

    if o.get("trainer.val_only") == "True":
        score = write_val(start)
        log_metrics(start, {f"val-core/chess-puzzle-v0/reward/mean@{val_n}": score})
        print(f"mock eval: step {start}, {val_puzzles}x{val_n} rollouts, score {score}")
        return

    target = int(o["trainer.total_training_steps"])
    fail = os.environ.get("MOCK_FAIL_ONCE_TRAIN", "")
    # Outside the phase directory: run_cycles.sh moves a failed phase aside.
    marker = Path(os.environ.get("MOCK_STATE_DIR", run.parent)) / f"failed-once-{os.environ['RUN_NAME']}-{target}"
    if fail == f"{os.environ['RUN_NAME']}:{target}" and not marker.exists():
        marker.write_text("failed once\n")
        sys.exit(f"mock training: injected failure before step {target}")

    batch, n = int(os.environ["TRAIN_BATCH_SIZE"]), int(os.environ["ROLLOUT_N"])
    previous = resume or None
    for step in range(start + 1, target + 1):
        total = batch * n
        accepted = max(1, int(total * (0.05 + 0.01 * step)))
        (run / "rollouts").mkdir(exist_ok=True)
        (run / "rollouts" / f"{step}.jsonl").write_text("\n".join("{}" for _ in range(total)) + "\n")
        ckpt = run / f"global_step_{step}"
        (ckpt / "actor" / "huggingface").mkdir(parents=True, exist_ok=True)
        (ckpt / "actor" / "model_world_size_1_rank_0.pt").write_bytes(b"mock")
        (ckpt / "actor" / "huggingface" / "config.json").write_text('{"mock": true}\n')
        (ckpt / "actor" / "huggingface" / "tokenizer_config.json").write_text("{}\n")
        (ckpt / "data.pt").write_bytes(b"mock")
        (ckpt / "mock_lineage.json").write_text(json.dumps(
            {"step": step, "resumed_from": previous, "harness_sha": harness_sha}) + "\n")
        previous = str(ckpt)
        data = {"online_rsft/total": total, "online_rsft/accepted": accepted,
                "actor/sft_updates": (accepted + 63) // 64, "timing_s/update_weights": 0.1, "timing_s/step": 0.2}
        if step == target:
            data[f"val-core/chess-puzzle-v0/reward/mean@{val_n}"] = write_val(step)
        log_metrics(step, data)
        time.sleep(float(os.environ.get("MOCK_TRAIN_SLEEP_S", "0")))
    print(f"mock training: steps {start + 1}-{target} with harness {harness_sha}")


if __name__ == "__main__":
    main()
