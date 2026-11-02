"""Validate the one-step path against recorded full-size training metrics."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'ladder/hybrid_test.sh').read_text()
validator=source.split('"$training_steps" <<\'PY\'\n',1)[1].split('\nPY\n',1)[0]


class TrainingSteps(unittest.TestCase):
    def test_evaluation_uses_requested_checkpoint_step(self):
        with tempfile.TemporaryDirectory() as name:
            p=Path(name);(p/'val_rollouts').mkdir()
            (p/'metrics.jsonl').write_text(json.dumps({'step':7,'data':{'val-core/acc':0.1}})+'\n')
            (p/'val_rollouts/7.jsonl').write_text('{}\n'*2048)
            command=[sys.executable,'-c',validator,str(p/'metrics.jsonl'),str(p),str(p/'passed.json'),'eval','1']
            env=dict(os.environ,HYBRID_EVAL_STEP='7')
            subprocess.run(command,env=env,check=True,capture_output=True)
            self.assertEqual(json.loads((p/'passed.json').read_text())['validation_rollouts'],2048)
            (p/'val_rollouts/7.jsonl').write_text('{}\n'*2047)
            self.assertNotEqual(subprocess.run(command,env=env,capture_output=True).returncode,0)

    def test_resumed_step_checks_only_new_rollouts_and_correct_final_validation(self):
        rows=[json.loads(s) for s in (ROOT/'tests/fixtures/shared_training_metrics.jsonl').read_text().splitlines()]
        metric=next(r for r in rows if r['step']==2)
        for start in [1,2]:
            with self.subTest(start=start),tempfile.TemporaryDirectory() as name:
                p=Path(name);(p/'rollouts').mkdir();(p/'val_rollouts').mkdir()
                target=start+1;metric['step']=target
                (p/'metrics.jsonl').write_text(json.dumps(metric)+'\n')
                (p/'rollouts'/f'{target}.jsonl').write_text('{}\n'*2048)
                (p/'val_rollouts'/f'{target}.jsonl').write_text('{}\n'*32)
                command=[sys.executable,'-c',validator,str(p/'metrics.jsonl'),str(p),str(p/'passed.json'),'train','1']
                env=dict(os.environ,HYBRID_VALIDATION_START_STEP=str(start))
                subprocess.run(command,env=env,check=True,capture_output=True)
                self.assertEqual(json.loads((p/'passed.json').read_text())['target_step'],target)
                (p/'rollouts'/f'{target}.jsonl').unlink()
                self.assertNotEqual(subprocess.run(command,env=env,capture_output=True).returncode,0)

    def test_one_and_two_step_completion_and_missing_update(self):
        recorded=[json.loads(s) for s in (ROOT/'tests/fixtures/shared_training_metrics.jsonl').read_text().splitlines()]
        for steps in [1,2]:
            with self.subTest(steps=steps),tempfile.TemporaryDirectory() as name:
                p=Path(name);(p/'rollouts').mkdir();(p/'val_rollouts').mkdir()
                metrics=p/'metrics.jsonl';passed=p/'passed.json'
                metrics.write_text(''.join(json.dumps(x)+'\n' for x in recorded if x['step']<=steps))
                for step in range(1,steps+1):(p/'rollouts'/f'{step}.jsonl').write_text('{}\n'*2048)
                (p/'val_rollouts'/f'{steps}.jsonl').write_text('{}\n'*32)
                command=[sys.executable,'-c',validator,str(metrics),str(p),str(passed),'train',str(steps)]
                subprocess.run(command,check=True,capture_output=True)
                self.assertEqual(json.loads(passed.read_text())['steps'],steps)
                rows=[json.loads(x) for x in metrics.read_text().splitlines()]
                rows[-1]['data']['actor/sft_updates']=0
                metrics.write_text(''.join(json.dumps(x)+'\n' for x in rows))
                self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)


if __name__=='__main__':unittest.main()
