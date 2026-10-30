import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('prepare',Path(__file__).resolve().parents[1]/'setup/prepare_hybrid_launcher.py')
prepare=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class HybridLauncher(unittest.TestCase):
    def test_private_copy_preserves_source_and_selects_existing_hybrid_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            scripts=root/'upstream/scripts'
            scripts.mkdir(parents=True)
            launcher=scripts/'train_chess_puzzle_multinode_disagg.sh'
            pre=scripts/'chess_puzzle_preamble_disagg_rsft.py'
            original='python3 "$PROJECT_DIR/scripts/chess_puzzle_preamble_disagg_rsft.py" "$@"\n'
            original_pre='from verl.trainer.main_textarena_disagg_rsft import main as _main\n'
            launcher.write_text(original);pre.write_text(original_pre)
            new_sh,new_pre=prepare.prepare(root/'upstream',root/'out')
            self.assertEqual(launcher.read_text(),original)
            self.assertEqual(pre.read_text(),original_pre)
            self.assertIn('LADDER_HYBRID_PREAMBLE',new_sh.read_text())
            self.assertIn('from verl.trainer.main_ppo import main',new_pre.read_text())
            self.assertTrue((root/'out/source.json').exists())

    def test_changed_upstream_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);scripts=root/'scripts';scripts.mkdir()
            (scripts/'train_chess_puzzle_multinode_disagg.sh').write_text('unknown entrypoint')
            (scripts/'chess_puzzle_preamble_disagg_rsft.py').write_text('unknown entrypoint')
            with self.assertRaises(ValueError):prepare.prepare(root,root/'out')
            self.assertFalse((root/'out').exists())


if __name__=='__main__':unittest.main()
