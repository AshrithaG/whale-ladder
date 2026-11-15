"""Known original candidate patch upgrades atomically without fetching a repository."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('candidate_migration', ROOT / 'setup/apply_candidate_validation.py')
migration = importlib.util.module_from_spec(spec); spec.loader.exec_module(migration)


class CandidateMigrationTests(unittest.TestCase):
    def test_original_patch_migrates_to_current_and_current_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); target = root / migration.TARGET
            target.parent.mkdir(parents=True)
            target.write_bytes((ROOT / 'tests/fixtures/pinned_meta_harness_before_validation.py').read_bytes())
            original = ROOT / 'patches/compatibility/candidate-validation-d5af0e6.patch'
            subprocess.run(['git', 'apply', str(original)], cwd=root, check=True)
            self.assertEqual(migration.apply(root, ROOT), 'migrated-original')
            updated = target.read_bytes()
            subprocess.run(['git', 'apply', '--reverse', '--check', str(ROOT / 'patches/candidate-validation.patch')],
                           cwd=root, check=True)
            self.assertEqual(migration.apply(root, ROOT), 'current')
            self.assertEqual(target.read_bytes(), updated)

    def test_unpatched_pinned_source_applies_current_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); target = root / migration.TARGET
            target.parent.mkdir(parents=True)
            target.write_bytes((ROOT / 'tests/fixtures/pinned_meta_harness_before_validation.py').read_bytes())
            self.assertEqual(migration.apply(root, ROOT), 'applied')
            self.assertEqual(migration.apply(root, ROOT), 'current')

    def test_unknown_modifications_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); target = root / migration.TARGET
            target.parent.mkdir(parents=True); target.write_text('private unknown source\n')
            with self.assertRaisesRegex(RuntimeError, 'Unknown candidate-validation state'):
                migration.apply(root, ROOT)
            self.assertEqual(target.read_text(), 'private unknown source\n')


if __name__ == '__main__': unittest.main()
