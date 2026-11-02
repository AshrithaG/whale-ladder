import importlib.util
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "setup/repair_chess_release.py"
spec = importlib.util.spec_from_file_location("repair_chess_release", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReleaseRepair(unittest.TestCase):
    def source(self, root):
        source = root / "domains/math_reasoning/verl/tools"
        source.mkdir(parents=True)
        (source / "schemas.py").write_text("canonical schema\n")
        (source / "__init__.py").write_text("license notice\n")
        config = root / "domains/chess_puzzles/verl/trainer/config"
        config.mkdir(parents=True)
        (config / "_generated_ppo_trainer.yaml").write_text(
            "actor_rollout_ref:\n  enabled: true\ndata:\n  train_batch_size: 1024\n"
            "  sampler:\n    class_path: null\ncritic:\n  enabled: false\n")
        return source

    def test_restore_is_exact_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.source(root)
            (source / "__pycache__").mkdir()
            (source / "__pycache__/schemas.pyc").write_bytes(b"cache")
            self.assertEqual(module.repair(root), 3)
            destination = root / "domains/chess_puzzles/verl/tools"
            self.assertEqual((destination / "schemas.py").read_bytes(), (source / "schemas.py").read_bytes())
            self.assertFalse((destination / "__pycache__").exists())
            data = root / "domains/chess_puzzles/verl/trainer/config/data/legacy_data.yaml"
            self.assertIn("train_batch_size: 1024\nsampler:\n  class_path: null", data.read_text())
            self.assertNotIn("critic:", data.read_text())
            self.assertEqual(module.repair(root), 0)

    def test_conflict_does_not_overwrite_or_partially_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.source(root)
            destination = root / "domains/chess_puzzles/verl/tools"
            destination.mkdir(parents=True)
            (destination / "schemas.py").write_text("user modification\n")
            with self.assertRaises(RuntimeError):
                module.repair(root)
            self.assertEqual((destination / "schemas.py").read_text(), "user modification\n")
            self.assertFalse((destination / "__init__.py").exists())


if __name__ == "__main__":
    unittest.main()
