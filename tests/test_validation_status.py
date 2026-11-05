"""Exercise the actual PSC wrapper's finalization with CPU-only phase commands."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ValidationStatusTests(unittest.TestCase):
    def run_wrapper(self, cycle_body, interrupt=False, dirty=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code, work, fake = root / 'code', root / 'work', root / 'bin'
            for folder in (code / 'ladder', code / 'setup', code / 'tools', work / 'venv/bin', fake):
                folder.mkdir(parents=True, exist_ok=True)
            for name in ('ladder/validate_candidate.sh', 'setup/load_candidate_config.py'):
                (code / name).write_bytes((ROOT / name).read_bytes())
            (code / 'ladder/config.env').write_text('OUT_ROOT="$LADDER_ROOT/runs"\n')
            (work / 'venv/bin/activate').write_text(':\n')
            (work / 'installed-revision').write_text('fixture\n')
            (fake / 'git').write_text('''#!/bin/sh
if [ "$1" = rev-parse ]; then echo fixture; fi
if [ "$1" = diff ]; then
  if [ "$TEST_DIRTY" = unstaged ] && [ "$2" = --quiet ]; then exit 1; fi
  if [ "$TEST_DIRTY" = staged ] && [ "$2" = --cached ]; then exit 1; fi
fi
exit 0
''')
            (fake / 'git').chmod(0o755)
            (fake / 'python').symlink_to(sys.executable)
            (fake / 'python3').symlink_to(sys.executable)
            (code / 'tools/observe_run.py').write_text('import time\ntime.sleep(30)\n')
            for name in ('setup/install_gpu_kernels.sh', 'setup/preflight.sh'):
                (code / name).write_text('exit 0\n')
            (code / 'ladder/run_cycles.sh').write_text(cycle_body)
            config = root / 'config.json'
            config.write_text(json.dumps(dict(LADDER_ROOT=str(work), LADDER_CODE_DIR=str(code),
                                             EXPECTED_REVISION='fixture')))
            ready = root / 'ready'
            env = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ['PATH'],
                       LADDER_JOB_GPUS='0,1', TEST_READY=str(ready), TEST_DIRTY=dirty or '')
            with (root / 'console.log').open('w') as log:
                proc = subprocess.Popen(['bash', str(code / 'ladder/validate_candidate.sh'), str(config)],
                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    if interrupt:
                        deadline = time.monotonic() + 8
                        while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
                            time.sleep(.02)
                        self.assertTrue(ready.exists(), (root / 'console.log').read_text())
                        os.killpg(proc.pid, signal.SIGTERM)
                    returncode = proc.wait(timeout=8)
                    path = work / 'runs/modal-control/status.json'
                    status = json.loads(path.read_text()) if path.exists() else None
                    return returncode, status, (root / 'console.log').read_text()
                finally:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=3)

    def test_normal_completion_is_recorded(self):
        code, status, _ = self.run_wrapper('exit 0\n')
        self.assertEqual(code, 0)
        self.assertEqual(status['state'], 'completed')
        self.assertEqual(status['returncode'], 0)

    def test_failed_cycle_is_recorded(self):
        code, status, _ = self.run_wrapper('exit 17\n')
        self.assertEqual(code, 17)
        self.assertEqual(status['state'], 'failed')
        self.assertEqual(status['returncode'], 17)

    def test_termination_during_training_cannot_report_completion(self):
        code, status, _ = self.run_wrapper('touch "$TEST_READY"\nsleep 30\n', interrupt=True)
        self.assertEqual(status['state'], 'failed')
        self.assertEqual(code, 143)
        self.assertEqual(status['returncode'], 143)

    def test_dirty_tracked_changes_fail_with_actionable_message(self):
        for dirty in ('unstaged', 'staged'):
            with self.subTest(dirty=dirty):
                code, status, console = self.run_wrapper('exit 0\n', dirty=dirty)
                self.assertEqual(code, 2)
                self.assertIsNone(status, 'Experiment phases must not start')
                self.assertIn('checkout is dirty; commit or stash', console)


if __name__ == '__main__':
    unittest.main()
