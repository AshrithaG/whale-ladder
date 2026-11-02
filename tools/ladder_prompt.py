"""Instructions every proposer gets on top of WHALE's own prompt, identical for
every rung. Used by tools/two_gpu_search.py (the real search) and
tools/proposer_format_check.py (layer 2), so both test the same prompt.
"""
from pathlib import Path


def preamble(run_dir: Path) -> str:
    # The chess skill never states the manifest format (the Math and SearchQA
    # skills do), so it is stated here.
    return (
        f'Your actual working directory is {run_dir}. Use relative paths from here. '
        'Write valid Python without backslash-escaped identifiers. '
        'Harness code must not import sys or modify sys.path. Allowed imports are '
        '__future__, collections, itertools, math, random, re, statistics. '
        'Write pending_eval.json at the working-directory root as JSON of the form '
        '{"candidates": [{"name": "<slot directory name, e.g. h1>", "axis": "<edit axis>", '
        '"hypothesis": "<why it should help>", "changes": "<what changed>"}]}, '
        'one entry per harness file you wrote.\n'
    )


def install(mh, run_dir: Path) -> None:
    """Wrap WHALE's task prompt and proposer system prompt for this run."""
    original = mh.render_task_prompt
    mh.render_task_prompt = lambda iteration, names: preamble(run_dir) + original(iteration, names)
    mh.PROPOSER_SYSTEM_PROMPT += f' Your working directory is exactly {run_dir}.'
