# whale-ladder

Runs WHALE's chess experiment (Qwen3.5-4B, Lichess puzzles) with the harness
proposer swapped across a ladder of models, to test whether a small model can
improve its own agent. Everything else (data, training, scoring, the Claude Code
proposer scaffold and its skill files) stays as WHALE released it.

| Arm | Proposer model | Where it runs |
|---|---|---|
| `rsft_only` | none (weight updates only, the baseline) | PSC |
| `whale_deepseek` | DeepSeek Flash (strong) | provider API |
| `whale_qwen27b` | Qwen3.8-27B (medium) | PSC, 2 of the 4 GPUs |
| `whale_qwen4b` | Qwen3.5-4B base weights, frozen (weak) | PSC, 1 GPU |
| `whale_self` | the checkpoint just trained (self-improvement) | PSC, 1 GPU |

This first milestone is the **one-cycle smoke test**: one short weight phase and
one harness-search iteration per arm. It proves the pipeline runs and measures
real per-step and per-candidate costs, so the full runs can be sized from data
instead of guesses.

## How one cycle runs on 4 H100s

```text
weight phase   GPU 0-1  verl FSDP trainer (full fine-tune, online RSFT)
               GPU 2-3  vLLM rollout workers (8 attempts per puzzle)
               -> checkpoint global_step_N

harness phase  pre-hook: merge checkpoint to HF format, serve it as the solver,
               serve the proposer model (local rungs), start LiteLLM
               Claude Code proposer -> LiteLLM -> rung's model; writes 3 candidates
               each candidate scored on 256 held-out puzzles by the solver
               post-hook: stop all servers, free the GPUs
```

GPU split in the harness phase: `deepseek` solver on 4 GPUs; `qwen27b` solver
on 2, proposer on 2; `qwen4b` and `self` solver on 3, proposer on 1.

## Before you start

1. **Use your own API keys.** The course credits are per student and must not
   be shared. The DeepSeek rung bills to whoever's key is in `ladder/secrets.env`.
2. **Check the GPU spec string**: `sinfo -p GPU-shared -o "%P %G"`. The Slurm
   scripts request `h100-80`; change `--gpus` in `slurm/*.sbatch` if it differs.
3. **Check internet from compute nodes** (the DeepSeek rung calls an API from
   inside the job). Preflight tests this; if it fails, run the local rungs only.
4. Know your account: every `sbatch` below takes `-A <account>`. Check the
   balance with `projects`.

## Steps

```bash
# 1. Put the repo on PSC project storage and configure it
cd $PROJECT && git clone <this repo> whale-ladder && cd whale-ladder
cp ladder/secrets.env.example ladder/secrets.env     # add your keys
vi ladder/config.env    # FIREWORKS_DEEPSEEK_MODEL and DEEPSEEK_PRICE_* for the DeepSeek rung

# 2. Install (login node, no GPU, about 1-2 h, about 90 GB with the 27B)
bash setup/install.sh

# 3. Check each proposer rung (about 10-30 min each)
sbatch -A <account> slurm/proposer_check.sbatch whale_qwen4b
sbatch -A <account> slurm/proposer_check.sbatch whale_qwen27b
sbatch -A <account> slurm/proposer_check.sbatch whale_deepseek

# 4. Smoke test, one arm at a time (up to 8 h each)
sbatch -A <account> slurm/smoke.sbatch whale_deepseek
sbatch -A <account> slurm/smoke.sbatch whale_self
```

Start with `whale_self`: it needs no internet or API key, and it exercises the
local proposer path and the trained-checkpoint handoff. Then run
`whale_deepseek` (after preflight confirms compute nodes have internet) and
`whale_qwen27b`.

## Running the arms across several people

Each arm is an independent run, so arms can go on different people's
allocations. The split we use:

| Person | Arms | Install |
|---|---|---|
| A | `whale_qwen27b` | full install (needs the 27B) |
| B | `whale_qwen4b`, `whale_self` | `SKIP_27B=1 bash setup/install.sh` |
| C | `whale_deepseek`, `rsft_only` | `SKIP_27B=1 bash setup/install.sh`, own API key |

For the arms to be comparable, everyone must run identical inputs:

1. **Same code:** the same commit of this repo, with no local edits to `ladder/profiles/`.
2. **Same weights:** `config.env` pins both models to exact Hugging Face revisions; nothing to do.
3. **Same puzzles:** the install builds them from a Lichess dataset that changes over time. So **B installs first**. B's install prints a data fingerprint; B puts it in `CHESS_DATA_FINGERPRINT` in `ladder/config.env`, pushes, and sends the three `.parquet` files to A and C. A and C set `CHESS_DATA_FROM` to the folder holding those files before running install. From then on, preflight refuses to run on different data.
4. **Same GPUs:** H100-80GB for everyone.

**Order:** B runs the `whale_self` smoke test first and alone: it needs no internet or API key and exercises the most code. A and C start once it passes.

Two jobs from one person may be submitted at the same time, even if Slurm puts them on the same node. Each job uses only its own GPUs, its own ports and its own Ray instance, and stops only its own processes.

## What to send back

- `slurm-ladder-smoke-<jobid>.out`
- `$PROJECT/whale-ladder-work/runs/<arm>-smoke-s42/report.md` (and `report.json`)
- for a proposer check: `$PROJECT/whale-ladder-work/runs/<arm>-check-s0/check_workdir/check_result.json`

A smoke run **passed** when `report.md` shows both phases, `proposer_sessions` ≥ 1
with 0 failed sessions, and at least one candidate with a success rate. The
"Extrapolated plan" section is the GPU-hour and dollar estimate for the full runs.

## Watching and stopping

```bash
squeue -u $USER
tail -f slurm-ladder-smoke-<jobid>.out
tail -f $PROJECT/whale-ladder-work/runs/<arm>-smoke-s42/logs/*.log
scancel <jobid>        # stops every server and Ray worker the job started
```

Check the API provider's dashboard after every DeepSeek run. Spending per run
is capped at `LADDER_MAX_USD` (default $25): the cap is checked before each
proposer session, so the last session can overshoot it slightly.

## Where things are written

| Path (under `$PROJECT/whale-ladder-work/`) | Contents |
|---|---|
| `runs/ckpt/<run>/` | checkpoints, `alternation.log`, verl logs |
| `runs/<run>/` | timing, proposer cost ledger, server logs, report |
| `WHALE/domains/chess_puzzles/meta_harness/runs/mh-<run>-round<N>/` | candidate harnesses, scores, proposer sessions |

Relaunching the same arm resumes: finished weight and harness phases are skipped.

## Changes from WHALE as released (commit 63faf4d)

WHALE's README says its launchers were verified only with stubbed training and
harness search. Running them for real needed these changes:

1. **Missing package.** The chess runner imports `autoharness_textarena`, which is
   not in the repo or on PyPI. `shim/` provides the client it uses.
2. **No solver server in the harness phase.** The release never serves the new
   checkpoint on the port its scoring config expects. `ladder/hooks/` merges the
   FSDP checkpoint and serves it; `patches/whale-ladder.patch` adds hook points
   to `run/lib/alternate.sh`.
3. **Fixed port 8044.** Ports are now per job, so jobs sharing a node don't collide.
4. **Proposer backend.** Claude Code is pointed at LiteLLM via
   `ANTHROPIC_BASE_URL`; the `--effort` flag (Anthropic-only) is not sent. The
   patch lets an empty `PROPOSER_EFFORT` mean "no flag".
5. **Install.** `pip install -e verl` from WHALE's docs fails for chess (no
   setup.py); dependencies are installed directly, without WHALE's `numpy<2` pin.
6. **Layout.** The paper used 16 H200s (4 trainer + 12 rollout); this runs on one
   4-GPU job (2 + 2).
7. **Proposer cost.** The cost ledger and dollar cap are added to `claude_wrapper.py`.

Also note:
- **Iteration count:** per the author (WHALE issue #2), `ROUND_MH_ITERS=5` in the launchers is a mistake; the paper's schedule uses 6.
- **Training algorithm:** the release leaves `CHESS_PUZZLE_ALGO=grpo` with RSFT enabled by override. That is what runs here; it is printed as `algo=` in the training log.

## Troubleshooting

- **Trainer out of memory (4B full fine-tune on 2 GPUs):** add `FSDP_OPTIM_OFFLOAD=True` to the profile.
- **Ray hangs at startup:** too few CPUs for the agent workers. Lower `AGENT_NUM_WORKERS` in the profile.
- **Proposer check fails on tool calls:** check the vLLM tool-call / reasoning parser names for the model and set `TOOL_CALL_PARSER` / `REASONING_PARSER` in `config.env`.
- **Proposer check fails with a context error (small models):** lower `PROPOSER_MAX_MODEL_LEN` only if the model cannot serve it. Claude Code's system prompt plus the skill file is long, and a 4B failing here is itself a result: it is logged separately from bad edits.
- **Preflight "outbound internet" fails:** compute nodes have no internet. Run the local rungs; the DeepSeek rung needs a different setup.
