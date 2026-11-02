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
(a.target_dir / "config.json").write_text(json.dumps({"mock": True, "merged_from": str(a.local_dir)}) + "\n")
(a.target_dir / "model.safetensors").write_bytes(b"mock")
print(f"mock merge: {a.local_dir} -> {a.target_dir}")
