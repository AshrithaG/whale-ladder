"""Check that progress and timing reports preserve their measurement meanings."""
import json
import http.client
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,str(ROOT/'setup'))
from observe_run import snapshot, parse_metrics, inference_metrics
from summarize_cycles import timing_report
from ladder_rollout_progress import RolloutProgress
from modal_gpu_status import describe, load_snapshot


class ObservabilityTests(unittest.TestCase):
    def test_non_http_port_does_not_hide_the_real_metrics_port(self):
        proc=SimpleNamespace(pid=1,info={'name':'vLLMHttpServer'},
            net_connections=lambda **_: [SimpleNamespace(status='LISTEN',
                laddr=SimpleNamespace(ip='127.0.0.1',port=p)) for p in (100,200)])
        psutil=SimpleNamespace(process_iter=lambda _: [proc],Error=RuntimeError)
        response=Mock()
        response.__enter__=Mock(return_value=SimpleNamespace(read=lambda:b'vllm:generation_tokens_total 900\n'))
        response.__exit__=Mock(return_value=False)
        opener=Mock()
        opener.open.side_effect=[http.client.BadStatusLine('HTTP/2 binary preface'),response]
        values,errors=inference_metrics(psutil,opener)
        self.assertEqual(values[0]['port'],200)
        self.assertEqual(values[0]['vllm:generation_tokens_total'],900)
        self.assertEqual(errors[0]['type'],'BadStatusLine')

    def test_partial_live_snapshot_falls_back_to_last_complete_history_row(self):
        valid=dict(time=10,checked_utc='sample',rollouts={})
        volume=SimpleNamespace(read_file=lambda name: [b'\x00'*20] if name.endswith('live.json')
            else [(json.dumps(valid)+'\n').encode()+b'{"time":'])
        result=load_snapshot(volume)
        self.assertEqual(result['time'],10)
        self.assertIn('history',result['snapshot_source'])
        self.assertEqual(result['rollouts'],{})

    def test_incomplete_observations_do_not_become_zero_rollout_counts(self):
        volume=SimpleNamespace(read_file=lambda _: [b'\x00'])
        with self.assertRaisesRegex(RuntimeError,'no rollout count inferred'):
            load_snapshot(volume)

    def test_live_attempts_remain_distinct_from_accepted_training_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            train=root/'cycles/trial/c1-train'
            progress=train/'rollout-progress'
            for worker in ('a','b'):
                tracker=RolloutProgress(progress,target=128,batch_total=256,expected_workers=2,
                    sequence=1,validation=False,global_step=1,worker_id=worker)
                for _ in range(64):tracker.record('completed')
            (train/'training-metrics.jsonl').write_text(json.dumps(dict(step=1,data={
                'online_rsft/total':256,'online_rsft/accepted':5,'actor/sft_updates':1}))+'\n')
            psutil=SimpleNamespace(virtual_memory=lambda:SimpleNamespace(total=256*2**30,
                available=200*2**30,used=56*2**30),cpu_percent=lambda:25.0,process_iter=lambda fields:[])
            with patch('observe_run.subprocess.check_output',return_value='0,GPU-a,100,81559,50,200\n'):
                row=snapshot(root,psutil,None)
            self.assertEqual(next(iter(row['rollouts'].values()))['completed'],128)
            self.assertEqual(next(iter(row['training'].values()))['accepted_samples'],5)
            text=describe(row)
            self.assertIn('completed 128/256',text)
            self.assertIn('accepted 5',text)

    def test_nested_training_timers_are_not_added_to_phase_wall_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);train=root/'c1-train';train.mkdir()
            rows=[dict(phase='c1-train',started_unix=10,ended_unix=110,elapsed_seconds=100,status='completed'),
                  dict(phase='c1-search',started_unix=115,ended_unix=135,elapsed_seconds=20,status='failed')]
            (root/'phase-timing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (train/'training-metrics.jsonl').write_text(json.dumps(dict(step=1,data={'timing_s/gen':80}))+'\n')
            result=timing_report(root)
            self.assertEqual(result['sequential_phase_seconds'],120)
            self.assertEqual(result['controller_wall_seconds'],125)
            self.assertEqual(result['controller_gap_seconds'],5)
            self.assertEqual(result['trainer_details']['c1-train']['1']['timing_s/gen'],80)
            self.assertEqual(result['phases'][-1]['status'],'failed')

    def test_http_metrics_are_labelled_and_aggregated_as_requests_and_tokens(self):
        body='''# HELP vllm:generation_tokens_total tokens
vllm:generation_tokens_total{engine="0"} 300
vllm:generation_tokens_total{engine="1"} 200
vllm:request_success_total{model="x"} 7
vllm:num_requests_running{model="x"} 2
unrelated_metric 999
'''
        metrics=parse_metrics(body)
        self.assertEqual(metrics['vllm:generation_tokens_total'],500)
        self.assertNotIn('unrelated_metric',metrics)
        text=describe(dict(time=1,inference=[dict(pid=1,port=2,**metrics)]))
        self.assertIn('HTTP model responses=7',text)
        self.assertNotIn('completed 7',text)


if __name__=='__main__':unittest.main()
