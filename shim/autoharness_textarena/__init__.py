"""Stand-in for the `autoharness_textarena` package that WHALE's chess domain imports.

WHALE's release imports `autoharness_textarena.config.LLMConfig` and
`autoharness_textarena.llm.{ChatMessage, LLMClient}` from
`domains/chess_puzzles/autoharness_chess_puzzle/runner.py`, but the package is
not in the repository or on PyPI. This module implements only the surface that
runner uses: an OpenAI-compatible chat client that returns the raw completion
text and its usage dict. The textarena game runners are not provided; the chess
puzzle path does not call them.
"""
