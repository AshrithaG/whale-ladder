"""Fake FSDP-to-Hugging-Face merger for the CPU mock (LADDER_MERGE_CMD).

Same arguments as `python -m verl.model_merger merge`; writes the files
lib.sh's merge_ckpt checks before it publishes the serving directory.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--local_dir", required=True, type=Path)
ap.add_argument("--target_dir", required=True, type=Path)
a = ap.parse_args()
assert (a.local_dir / "model_world_size_1_rank_0.pt").exists(), f"no actor shards in {a.local_dir}"
a.target_dir.mkdir(parents=True, exist_ok=True)
# Architecture/tokenizer stay fixed while the simulated weights change.
(a.target_dir / "config.json").write_text('{"mock": true}\n')
lineage = a.local_dir.parent / 'mock_lineage.json'
weights = lineage.read_bytes() if lineage.exists() else b'mock'
(a.target_dir / "model.safetensors").write_bytes(weights)
print(f"mock merge: {a.local_dir} -> {a.target_dir}")
