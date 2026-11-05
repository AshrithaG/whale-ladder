import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

P=Path(__file__).resolve().parents[1]/'setup/check_install_reuse.py'
spec=importlib.util.spec_from_file_location('reuse',P);reuse=importlib.util.module_from_spec(spec);spec.loader.exec_module(reuse)


class ReuseTests(unittest.TestCase):
    def test_logging_changes_reuse_environment_but_dependency_changes_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def git(*args):return subprocess.check_output(['git','-c','user.name=Test','-c','user.email=test@example.invalid',*args],cwd=root,text=True).strip()
            git('init','-q')
            (root/'setup').mkdir();(root/'setup/install.sh').write_text('install pinned dependencies\n')
            git('add','.');git('commit','-qm','prepared');base=git('rev-parse','HEAD')
            (root/'observe.py').write_text('logging only\n')
            git('add','.');git('commit','-qm','logging');runtime=git('rev-parse','HEAD')
            self.assertTrue(reuse.check(root,base,runtime)['reused_environment'])
            (root/'setup/install.sh').write_text('different dependencies\n')
            git('add','.');git('commit','-qm','dependency change')
            with self.assertRaisesRegex(RuntimeError,'inputs changed'):
                reuse.check(root,base,git('rev-parse','HEAD'))


if __name__=='__main__':unittest.main()
