"""Check FP32 FSDP transfer capacity using safetensors headers, not weights."""
import argparse
import json
import math
from pathlib import Path
import struct


def largest_tensor(model_dir):
    root = Path(model_dir)
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise ValueError("No safetensors model shards found at " + str(root))
    for index in root.glob("*.safetensors.index.json"):
        for shard in set(json.loads(index.read_text())["weight_map"].values()):
            if not (root / shard).is_file():
                raise ValueError("Missing model shard: " + shard)
    largest = (0, "")
    for path in files:
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > 100 * 1024 * 1024:
                raise ValueError("Unexpectedly large safetensors header: " + str(path))
            header = json.loads(stream.read(length))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            # Actor parameters and exported FSDP state are FP32, even though
            # the downloaded checkpoint and mixed-precision forwards are BF16.
            # DisaggregatedAsyncActorWorker sends state without a dtype cast.
            stored = info["data_offsets"][1] - info["data_offsets"][0]
            largest = max(largest, (max(stored, math.prod(info["shape"]) * 4), name))
    return largest


def check(model_dir, bucket_mb):
    size, name = largest_tensor(model_dir)
    if size > bucket_mb * 1024**2:
        raise ValueError("Bucket {} MiB cannot hold {} ({:.1f} MiB)".format(
            bucket_mb, name, size / 1024**2))
    return "largest tensor {}: {:.1f} MiB; bucket {} MiB".format(name, size / 1024**2, bucket_mb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("bucket_mb", type=int)
    args = parser.parse_args()
    print(check(args.model_dir, args.bucket_mb))
