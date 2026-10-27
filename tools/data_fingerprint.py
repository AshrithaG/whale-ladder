"""Print a fingerprint of the three chess parquet files, to compare across people.

    python tools/data_fingerprint.py DATA_DIR

Prints one sha256 per file, then a short combined fingerprint on the last line.
Two runs used identical puzzles exactly when their fingerprints match.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

FILES = (
    "lichess_puzzles_train_16384.parquet",
    "lichess_puzzles_mh_val_256.parquet",
    "lichess_puzzles_test_256.parquet",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    data = Path(sys.argv[1])
    lines = [f"{sha256(data / name)}  {name}" for name in FILES]
    print("\n".join(lines))
    print(hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16])


if __name__ == "__main__":
    main()
