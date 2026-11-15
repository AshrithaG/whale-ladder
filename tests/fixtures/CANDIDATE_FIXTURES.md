Candidate timeout fixtures:

- `c4-h1.py` is the unchanged failed cycle-4 candidate from the recorded parser-stall regression. The test calls its parser with a synthetic 11,356-character response that exercises the same pathological regex.
- `pinned_harness_loader.py` comes unchanged from WHALE commit `63faf4deb2dd2938775af6e2b020a9030f3aa697`, `domains/chess_puzzles/autoharness_chess_puzzle/harness.py`.
- `pinned_meta_harness_before_validation.py` is that pinned WHALE revision's `domains/chess_puzzles/meta_harness/meta_harness_chess_puzzle.py` after this repository's existing `whale-ladder.patch`, before `candidate-validation.patch`. Tests apply the current validation patch to this fixture and execute its actual function bodies with CPU-only inference boundaries.
- `candidate_test_worker.py` supplies controlled subprocess evaluation behavior, retaining the pinned source loader for real generated-module execution.

- `pinned_runner_before_memory_errors.py` and `pinned_benchmark_before_memory_errors.py` are the unchanged pinned WHALE runner and benchmark sources from `63faf4deb2dd2938775af6e2b020a9030f3aa697`. Tests apply `candidate-memory-errors.patch` and execute the actual formatter, prompt, parser, legality, and sweep bodies.
- `patches/compatibility/candidate-validation-d5af0e6.patch` preserves the original validation patch exactly from ladder commit `d5af0e6ac09a13cb6a796496d8043fbf7b31ab8e`, for deterministic upgrade testing and installation.
