import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_transfer_buffer", ROOT / "setup/check_transfer_buffer.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TransferBuffer(unittest.TestCase):
    def test_rejects_undersized_bucket_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            # Exact pinned 4B embedding dimensions. Only the header is read.
            header = json.dumps({"embedding": {"shape": [248320, 2560], "dtype": "BF16",
                                                "data_offsets": [0, 1271398400]}}).encode()
            (Path(temp) / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
            with self.assertRaisesRegex(ValueError, "cannot hold"):
                module.check(temp, 512)
            with self.assertRaisesRegex(ValueError, "cannot hold"):
                module.check(temp, 1536)
            self.assertIn("2425.0 MiB", module.check(temp, 3072))

    def test_missing_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "part1.safetensors").touch()
            (root / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {"a": "part1.safetensors", "b": "part2.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "Missing model shard"):
                module.check(temp, 1536)
