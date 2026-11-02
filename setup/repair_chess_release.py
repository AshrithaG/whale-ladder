#!/usr/bin/env python3
"""Restore the tools package omitted from WHALE's pinned chess release.

Use the canonical files vendored by the Math domain in the same checkout.
Existing differing files are never overwritten.
"""
import argparse
from pathlib import Path
import shutil
import textwrap


def legacy_data_config(root):
    reference = root / "domains/chess_puzzles/verl/trainer/config/_generated_ppo_trainer.yaml"
    lines = reference.read_text().splitlines(keepends=True)
    start = lines.index("data:\n") + 1
    section = []
    for line in lines[start:]:
        if line.strip() and not line[0].isspace() and not line.startswith("#"):
            break
        section.append(line)
    if not section:
        raise RuntimeError("The generated reference has no data defaults")
    return ("# Restored from this release's _generated_ppo_trainer.yaml data section.\n"
            + textwrap.dedent("".join(section))).encode()


def repair(whale_dir):
    root = Path(whale_dir)
    source = root / "domains/math_reasoning/verl/tools"
    destination = root / "domains/chess_puzzles/verl/tools"
    if not (source / "schemas.py").is_file():
        raise RuntimeError("Canonical tools package is missing from the WHALE checkout")
    files = [p for p in source.rglob("*") if p.is_file()
             and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    data_target = root / "domains/chess_puzzles/verl/trainer/config/data/legacy_data.yaml"
    data_content = legacy_data_config(root)
    if data_target.exists() and data_target.read_bytes() != data_content:
        raise RuntimeError("Existing data defaults differ; left untouched: " + str(data_target))
    for path in files:
        target = destination / path.relative_to(source)
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise RuntimeError("Existing file differs; left untouched: " + str(target))
    copied = 0
    for path in files:
        target = destination / path.relative_to(source)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(target))
            copied += 1
    if not data_target.exists():
        data_target.parent.mkdir(parents=True, exist_ok=True)
        data_target.write_bytes(data_content)
        copied += 1
    return copied


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("whale_dir", type=Path)
    args = parser.parse_args()
    print("Restored {} missing chess release files from the same WHALE checkout".format(repair(args.whale_dir)))
