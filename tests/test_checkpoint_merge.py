import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CheckpointMerge(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "domains/chess_puzzles").mkdir(parents=True)
        (self.root / "bin").mkdir()
        self.ckpt = self.root / "checkpoint"
        self.ckpt.mkdir()
        fake = self.root / "bin/python"
        fake.write_text('''#!/usr/bin/env bash
set -eu
while [[ $# -gt 0 ]]; do
  if [[ "$1" == --target_dir ]]; then target="$2"; break; fi
  shift
done
echo call >> "$CALLS"
mkdir -p "$target"
echo '{}' > "$target/config.json"
echo weights > "$target/model.safetensors"
if [[ "${MISSING_TOKENIZER:-0}" == 0 ]]; then
  echo '{}' > "$target/tokenizer_config.json"
fi
exit "${MERGE_EXIT:-0}"
''')
        fake.chmod(0o755)
        self.calls = self.root / "calls"
        self.env = os.environ | {
            "WHALE_DIR": str(self.root), "CKPT": str(self.ckpt),
            "LIB": str(ROOT / "ladder/lib.sh"), "CALLS": str(self.calls),
            "PATH": str(self.root / "bin") + ":" + os.environ["PATH"],
        }

    def merge(self, **settings):
        # Calling from an `if` disables Bash errexit inside the function;
        # errors must still never publish an incomplete serving directory.
        return subprocess.run(["bash", "-c", 'source "$LIB"; if merge_ckpt "$CKPT"; then exit 0; else exit $?; fi'],
                              env=self.env | settings, capture_output=True, text=True)

    def test_interrupted_merge_is_not_reused_and_old_output_is_preserved(self):
        hf = self.ckpt / "hf"
        hf.mkdir()
        (hf / "config.json").write_text("{}")
        (hf / "model.safetensors").write_text("partial")
        self.assertEqual(self.merge(MERGE_EXIT="7").returncode, 7)
        self.assertEqual((hf / "model.safetensors").read_text(), "partial")
        self.assertFalse((hf / ".merge_complete").exists())
        self.assertEqual(self.merge().returncode, 0)
        self.assertTrue((hf / ".merge_complete").exists())
        previous = list(self.ckpt.glob(".hf-merge.*.previous"))
        self.assertEqual(len(previous), 1)
        self.assertEqual((previous[0] / "model.safetensors").read_text(), "partial")
        calls = self.calls.read_text()
        self.assertEqual(self.merge().returncode, 0)
        self.assertEqual(self.calls.read_text(), calls)

    def test_incomplete_successful_output_is_not_published(self):
        self.assertEqual(self.merge(MISSING_TOKENIZER="1").returncode, 1)
        self.assertFalse((self.ckpt / "hf").exists())
