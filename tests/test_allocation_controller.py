"""Prove controller lifetime is independent of startup/worker success."""
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

    def check_hold(self, worker_result, missing_config=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = root / 'bin'
            fake.mkdir()
            counter = root / 'worker-calls'
            (fake / 'srun').write_text('#!/bin/bash\nprintf "called\\n" >> "$TEST_COUNTER"\nexit "$TEST_RESULT"\n')
            (fake / 'srun').chmod(0o755)
            config = root / 'job-config.json'
            if not missing_config: config.write_text('{}')
            env = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ['PATH'],
                       SLURM_SUBMIT_DIR=str(ROOT), SLURM_JOB_ID='controller-test',
                       TEST_COUNTER=str(counter), TEST_RESULT=str(worker_result))
            env.pop('LADDER_ROOT', None)
            with (root / 'output').open('w') as log:
                proc = subprocess.Popen(['bash', str(ROOT / 'slurm/candidate_validate.sbatch'), str(config)],
                                        env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    status = root / 'jobs/candidate-controller-test/status'
                    self.wait_for(lambda: status.exists() and 'allocation retained' in status.read_text())
                    self.assertIsNone(proc.poll(), 'Controller exited and would release the allocation')
                    self.assertIn('exit=' + str(2 if missing_config else worker_result), status.read_text())
                    if not missing_config:
                        self.assertEqual(counter.read_text().splitlines(), ['called'])
                        time.sleep(.1)
                        self.assertEqual(counter.read_text().splitlines(), ['called'], 'Unexpected automatic retry')
                finally:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=3)

    def test_worker_failure_retains_allocation_without_automatic_retry(self):
        self.check_hold(17)

    def test_worker_success_retains_allocation(self):
        self.check_hold(0)

    def test_missing_startup_configuration_retains_allocation(self):
        self.check_hold(0, missing_config=True)

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
