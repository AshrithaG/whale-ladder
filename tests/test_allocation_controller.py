"""Exercise allocation release, bounded diagnosis and manual retries in Bash."""
from contextlib import contextmanager
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ControllerTests(unittest.TestCase):
    def wait_for(self, predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate(): return
            time.sleep(.03)
        self.fail('Controller did not reach the expected state')

    @contextmanager
    def controller(self, worker_result=17, missing_config=False, hold_seconds='0'):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / 'bin'
            fake.mkdir()
            counter, result = root / 'worker-calls', root / 'worker-result'
            result.write_text(str(worker_result))
            (fake / 'srun').write_text('#!/bin/bash\nprintf "called\\n" >> "$TEST_COUNTER"\nexit "$(cat "$TEST_RESULT_FILE")"\n')
            (fake / 'srun').chmod(0o755)
            config = root / 'job-config.json'
            if not missing_config:
                config.write_text('{}')
            env = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ['PATH'],
                       SLURM_SUBMIT_DIR=str(ROOT), SLURM_JOB_ID='controller-test',
                       TEST_COUNTER=str(counter), TEST_RESULT_FILE=str(result),
                       LADDER_FAILURE_HOLD_SECONDS=hold_seconds)
            with (root / 'output').open('w') as log:
                proc = subprocess.Popen(['bash', str(ROOT / 'slurm/candidate_validate.sbatch'), str(config)],
                                        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    yield root, proc, counter
                finally:
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=3)

    def status(self, root):
        return root / 'jobs/candidate-controller-test/status'

    def test_success_releases_even_when_failure_hold_is_enabled(self):
        with self.controller(worker_result=0, hold_seconds='1200') as (root, proc, counter):
            self.assertEqual(proc.wait(timeout=3), 0)
            self.assertIn('PASSED', self.status(root).read_text())
            self.assertIn('allocation released', self.status(root).read_text())
            self.assertEqual(counter.read_text().splitlines(), ['called'])

    def test_worker_failure_releases_and_preserves_exit_code_by_default(self):
        with self.controller() as (root, proc, counter):
            self.assertEqual(proc.wait(timeout=3), 17)
            self.assertIn('exit=17; allocation released', self.status(root).read_text())
            self.assertEqual(counter.read_text().splitlines(), ['called'])

    def test_bootstrap_failure_releases_by_default(self):
        with self.controller(missing_config=True) as (root, proc, counter):
            self.assertEqual(proc.wait(timeout=3), 2)
            self.assertIn('exit=2; allocation released', self.status(root).read_text())
            self.assertFalse(counter.exists())

    def test_opt_in_failure_hold_expires_without_automatic_retry(self):
        with self.controller(hold_seconds='2') as (root, proc, counter):
            status = self.status(root)
            self.wait_for(lambda: status.exists() and 'allocation retained' in status.read_text())
            self.assertIsNone(proc.poll())
            self.assertEqual(proc.wait(timeout=4), 17)
            self.assertIn('hold expired', status.read_text())
            self.assertEqual(counter.read_text().splitlines(), ['called'])

    def test_bootstrap_diagnosis_is_also_bounded(self):
        with self.controller(missing_config=True, hold_seconds='1') as (root, proc, counter):
            self.assertEqual(proc.wait(timeout=3), 2)
            self.assertIn('hold expired', self.status(root).read_text())
            self.assertFalse(counter.exists())

    def test_manual_retry_can_succeed_and_releases_immediately(self):
        with self.controller(hold_seconds='1200') as (root, proc, counter):
            status = self.status(root)
            self.wait_for(lambda: status.exists() and 'allocation retained' in status.read_text())
            (root / 'worker-result').write_text('0')
            (status.parent / 'retry').touch()
            self.assertEqual(proc.wait(timeout=3), 0)
            self.assertEqual(counter.read_text().splitlines(), ['called', 'called'])
            self.assertIn('PASSED', status.read_text())

    def test_failed_retry_does_not_reset_diagnosis_deadline(self):
        with self.controller(hold_seconds='3') as (root, proc, counter):
            status = self.status(root)
            self.wait_for(lambda: status.exists() and 'allocation retained' in status.read_text())
            first = status.read_text().split('until elapsed=')[1].split('s;')[0]
            (status.parent / 'retry').touch()
            self.wait_for(lambda: len(counter.read_text().splitlines()) == 2)
            self.wait_for(lambda: status.exists() and 'allocation retained' in status.read_text())
            self.assertEqual(status.read_text().split('until elapsed=')[1].split('s;')[0], first)
            self.assertEqual(proc.wait(timeout=4), 17)
            self.assertEqual(counter.read_text().splitlines(), ['called', 'called'])

    def test_invalid_or_excessive_hold_is_rejected(self):
        for value in ('-1', '1201', 'forever'):
            with self.subTest(value=value), self.controller(hold_seconds=value) as (root, proc, counter):
                self.assertEqual(proc.wait(timeout=3), 2)
                self.assertIn('0 to 1200', (root / 'output').read_text())
                self.assertFalse(counter.exists())

    def test_explicit_configuration_quotes_paths_without_shell_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / 'should-not-exist'
            value = str(root / ('path with spaces $(touch ' + str(marker) + ')'))
            path = root / 'job.json'
            path.write_text(json.dumps(dict(LADDER_ROOT=value, LADDER_CODE_DIR=str(ROOT), EXPECTED_REVISION='abc')))
            exports = subprocess.check_output([sys.executable, str(ROOT / 'setup/load_candidate_config.py'), str(path)], text=True)
            result = subprocess.check_output(['bash', '-c', exports + '\nprintf "%s" "$LADDER_ROOT"'], text=True)
            self.assertEqual(result, value)
            self.assertFalse(marker.exists())


if __name__ == '__main__':
    unittest.main()
