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

## The pipeline: three stages

| Stage | How | Size | Answers |
|---|---|---|---|
| 1. Tiny test | interactive, 1 hour | 1 training step on 16 puzzles, 1 proposal, scoring on 32 puzzles | does every stage run end to end? |
| 2. Smoke test | `sbatch`, up to 8 h; one arm is enough | real sizes, one cycle | what does a real cycle cost (GPU hours, dollars)? |
| 3. Real runs | `sbatch`, all arms, split across people | sized from stage 2 | the results |

Move to the next stage only when the current one passes. Stage 1 catches
crashes in minutes. Its timings are mostly startup and do not predict real
costs, which is why stage 2 exists.

**Stage 1, interactive:**

```bash
tmux new -s ladder                        # on the login node; survives a dropped connection
srun -A <account> -p GPU-shared --gpus=h100-80:4 -t 01:00:00 --pty bash
cd $PROJECT/whale-ladder
bash ladder/interactive_test.sh whale_self
```

It takes about 30-45 minutes and ends with a PASS/FAIL line per stage. It
starts fresh each time, so rerunning after a fix is safe. Then, still in the
same session if time remains, check the API rung:
`bash ladder/proposer_check.sh whale_deepseek`.

## Steps

```bash
# 1. Put the repo on PSC project storage and configure it
cd $PROJECT && git clone <this repo> whale-ladder && cd whale-ladder
cp ladder/secrets.env.example ladder/secrets.env     # add your keys
vi ladder/config.env    # FIREWORKS_DEEPSEEK_MODEL and DEEPSEEK_PRICE_* for the DeepSeek rung

# 2. Install on a compute allocation with internet access.
#    On PSC, select the compiler module before installation and future launches:
export LADDER_GCC_MODULE=gcc/13.3.1-p20240614
bash setup/install.sh

# 3. Stage 1: the interactive tiny test above. Then check each proposer rung
#    (about 10-30 min each; can also run inside the interactive session)
sbatch -A <account> slurm/proposer_check.sbatch whale_qwen4b
sbatch -A <account> slurm/proposer_check.sbatch whale_qwen27b
sbatch -A <account> slurm/proposer_check.sbatch whale_deepseek

# 4. Stage 2: smoke test (up to 8 h); one arm is enough to size the real runs
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

### PSC runtime compatibility

Setup restores the missing chess `verl.tools` package from the Math domain in
the same pinned WHALE checkout, and restores `data/legacy_data.yaml` from that
release's generated trainer configuration. Conflicting local files are rejected.
Four runtime patches adapt vLLM expert imports, streamed tied-weight validation,
and the generation-pause API, and provide existing PyTorch padding helpers when
FlashAttention is absent. Padding fallback does not supply attention kernels.

The NCCL weight-transfer path needs CuPy matching PyTorch's CUDA runtime. Its
installer constrains installed packages while adding the missing dependency.
The kernel installer obtains matching precompiled kernels through vLLM's helper.
`ladder/config.env` selects Conda's C++ runtime and optionally loads the PSC GCC
module specified by `LADDER_GCC_MODULE`; persist that setting in your job environment
or site configuration so subsequent launches use the same compiler.

On the allocated GPU node, run `bash setup/preflight.sh whale_self tiny` before
the tiny pipeline. Preflight checks imports, declared dependencies, the actual
launcher configuration, compiler support, padding gradients, tied-weight loading,
PyTorch/CuPy buffer copies, and whether transfer buckets fit the largest tensor.
It does not establish collective-transfer correctness, full training memory fit,
or successful generated harnesses; those require the end-to-end tiny test.

The fixes were exercised during development on Python 3.11, PyTorch 2.13.0 with
CUDA 13, vLLM 0.32.0, Transformers 5.19.0, CuPy 14.2.0, and GCC 13.3.1. Those
GPU runs also contained separate two-GPU memory adaptations and recovery code.
This compatibility-only extraction still needs a clean installation and an
end-to-end run with the repository's original four-GPU profiles before it can
be considered independently validated. Installation retains the existing broad
dependency ranges and can resolve newer versions; record `pip-freeze.txt` and
avoid reinstalling into an experiment's active environment.

Dependency-free regression tests: `python3 -m unittest discover -s tests -v`.

### Runtime diagnostics

- **Trainer out of memory:** the profiles in this branch allocate four GPUs.
  Two-GPU full fine-tuning needs a separately validated memory configuration;
  the legacy `FSDP_OPTIM_OFFLOAD` flag alone did not prevent Adam initialization
  from exhausting GPU memory in development trials.
- **Ray hangs at startup:** too few CPUs for the agent workers. Lower `AGENT_NUM_WORKERS` in the profile.
- **Proposer check fails on tool calls:** check the vLLM tool-call / reasoning parser names for the model and set `TOOL_CALL_PARSER` / `REASONING_PARSER` in `config.env`.
- **Proposer check fails with a context error (small models):** lower `PROPOSER_MAX_MODEL_LEN` only if the model cannot serve it. Claude Code's system prompt plus the skill file is long, and a 4B failing here is itself a result: it is logged separately from bad edits.
- **Preflight "outbound internet" fails:** compute nodes have no internet. Run the local rungs; the DeepSeek rung needs a different setup.


## Shared two-GPU rollout/training validation (draft)

`ladder/hybrid_test.sh` is an opt-in one/two-step runner for two H100 80 GB GPUs,
24 CPU cores and 256 GiB host memory. The same two GPUs run independent vLLM
replicas during generation and FSDP2 workers during training. It selects WHALE's
existing `main_ppo` entrypoint in private launcher copies. The regular ladder
profiles and alternating driver retain their original behavior.

Defaults: 256 training puzzles x 8 attempts (2,048 rollouts per step), eight
agent workers, CUDA graphs (`ENFORCE_EAGER=false`), 75% inference memory, Triton
GDN prefill, native CPU offload, and one SFT epoch with a minibatch of 32 per
GPU (64 globally). Each training step saves weights and runs a small validation
of 16 test puzzles x 2 attempts. The saved checkpoint resumes model/dataloader
state; the upstream model+extra checkpoint does not preserve Adam state.

After installing this branch's patches, inside an existing two-H100 allocation:

```bash
export LADDER_ROOT=/path/to/your/whale-ladder-work
export LADDER_GCC_MODULE=gcc/13.3.1-p20240614
# Use separate, fresh output directories for composition and execution.
bash ladder/hybrid_test.sh "$LADDER_ROOT/review/shared-config" --dry-run
HYBRID_TRAINING_STEPS=1 bash ladder/hybrid_test.sh "$LADDER_ROOT/review/shared-step"
# Or evaluate 256 test puzzles x 8 attempts from a completed checkpoint:
HYBRID_MODE=eval HYBRID_SOURCE_CHECKPOINT=/path/to/global_step_1 \
  bash ladder/hybrid_test.sh "$LADDER_ROOT/review/shared-eval"
```

`--dry-run` prepares launcher/data files and composes Hydra configuration; it
starts no training or inference. Execution uses an existing allocation and
cleans up its own Ray processes; this runner does not submit or cancel jobs.
`HYBRID_AGENT_NUM_WORKERS` and `HYBRID_GPU_MEM_UTIL` remain explicit overrides
for matched comparisons. Use a fresh output directory for every setting.

Recorded development measurements are in `benchmarks/shared-two-gpu.json`.
On the same checkpoint and 2,048 evaluation rollouts, four versus eight workers
took 1,342.768 versus 1,064.366 seconds for generation and output processing,
with similar output-token totals. A subsequent training run generated 2,048
rollouts in 1,056.85 seconds and performed two SFT updates in 292.87 seconds.
The small eager/graph comparison used 128 rollouts and a single rollout GPU;
its speedup should not be extrapolated as a universal multiplier. The 75%
memory setting worked, but a matched 60% control was not completed.

This is a review extraction from development runs. It removes optional live
progress instrumentation and persistent-server integration, and makes the
measured eight-worker/75% configuration the opt-in runner's defaults. Its exact
extracted revision still needs a GPU generation/update validation before
merging. It is not a complete harness-search or full research schedule.

## Two-GPU candidate scoring and sleep/wake switching (opt-in)

`ladder/search_two_gpu.sh` runs a standalone harness-search phase from a saved
solver checkpoint. It uses two H100 80GB GPUs, a frozen Qwen3.5-4B proposer,
and the original 256-puzzle harness-search dataset. Use an idle allocation;
the development runs requested 24 CPU cores and 256 GiB host memory.

GPU 0 keeps one solver replica. GPU 1 alternates between a second solver replica
and the proposer using vLLM level-1 sleep/wake; both processes stay alive after
initialization. A loopback router distributes scoring requests between the
solver replicas, with capacity 128 requests per backend. It drains GPU 1 before
sleeping that solver, propagates HTTP failures without retrying, and records
request counters separately from WHALE's completed-puzzle counts.

The search runner scores the incoming baseline, proposes three harnesses per
iteration, and evaluates each on 256 puzzles with one attempt per puzzle. The
next iteration receives evaluation feedback. It defaults to one iteration and
accepts 1–6; the historical switching validation completed two real iterations.
All servers use CUDA graphs and 75% inference memory. Proposing and scoring
are sequential; an overlap benchmark showed no improvement.

After installing this branch, with the same environment setup as above:

```bash
# Checkpoint must contain actor/ and data.pt. Output must not already exist.
bash ladder/search_two_gpu.sh "$LADDER_ROOT/review/search" /path/to/global_step_1 2

# Optional matched comparison: replay the same saved h1/h2/h3 candidates
# with restart, sleep/wake, and overlapped sleep/wake; then run two real searches.
SWITCH_MODE=benchmark SWITCH_REFERENCE_RUN=/path/to/previous/harness-run \
  bash ladder/search_two_gpu.sh "$LADDER_ROOT/review/switch-benchmark" /path/to/global_step_1
```

The checkpoint merger publishes `hf/` only after successful conversion and
validation, and marks it `.merge_complete`. An older unmarked merge is preserved
before replacement. The runner writes server logs, `timing.jsonl`, router/GPU
metrics, and `harness-run-dir`; `passed.json` requires all expected candidates
to have 256-puzzle scores with the original 8,129-token assistant budget. Failure
stops the run and preserves the evidence. Cleanup signals only subprocess groups
started by this invocation; it does not submit or cancel an allocation.

`benchmarks/two-gpu-search.json` records the historical comparison: the same
three-candidate round, including switching, took 559.71 seconds with server
restarts and 393.70 seconds with sleep/wake. Each switch fell from approximately
64–72 seconds to less than one second. These measurements exclude initial model
startup and proposing, and do not establish a matched one-GPU/two-GPU speedup.

This extracted branch still requires GPU validation before merging. It changes
the baseline evaluation to use both GPUs and selects serialized switching for
new searches; those exact extracted defaults have not been rerun on H100s.
It does not fix proposer-generated invalid Python or missing harness functions,
and it does not introduce automatic candidate repair. Keeping servers resident
across weight updates, checkpoint hot reload, and the full alternating trial
controller remain separate experimental work.
