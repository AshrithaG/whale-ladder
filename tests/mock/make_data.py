"""Write small mock chess-puzzle splits in WHALE's real format.

    python tests/mock/make_data.py DOMAIN_DIR

Positions come from random legal play; each puzzle's single solution move is the
alphabetically smallest legal move, which the mock solver (tests/mock/bin/vllm)
knows. Rows go through WHALE's own convert_lichess_row, so the files have the
exact columns the real pipeline reads. Uses the real file names, with 64/16/16
rows instead of 16,384/256/256.
"""
import importlib.util
import random
import sys
from pathlib import Path

import chess
import pandas as pd

domain = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("prep", domain / "scripts/prepare_chess_puzzle_data.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

sizes = {"train": 64, "mh_val": 16, "test": 16}
names = {"train": "lichess_puzzles_train_16384.parquet", "mh_val": "lichess_puzzles_mh_val_256.parquet",
         "test": "lichess_puzzles_test_256.parquet"}
rng = random.Random(7)
rows, seen = [], set()
while len(rows) < sum(sizes.values()):
    board = chess.Board()
    for _ in range(rng.randint(2, 24)):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    start = board.fen()
    triggers = list(board.legal_moves)
    if not triggers:
        continue
    trigger = rng.choice(triggers)
    board.push(trigger)
    replies = sorted(m.uci() for m in board.legal_moves)
    if len(replies) < 2 or board.fen() in seen:
        continue
    seen.add(board.fen())
    row = prep.convert_lichess_row({"FEN": start, "Moves": f"{trigger.uci()} {replies[0]}",
                                    "PuzzleId": f"mock{len(rows):04d}", "Rating": 1500, "Themes": "mock"})
    if row:
        rows.append(row)

out = domain / "data" / "chess_puzzle"
out.mkdir(parents=True, exist_ok=True)
i = 0
for split in ("train", "mh_val", "test"):
    part = rows[i:i + sizes[split]]
    i += sizes[split]
    for r in part:
        r["extra_info"]["split"] = split
    pd.DataFrame(part).to_parquet(out / names[split], index=False)
    print(f"mock data: {len(part)} {split} puzzles -> {out / names[split]}")
