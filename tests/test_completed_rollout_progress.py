import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SETUP=Path(__file__).resolve().parents[1]/'setup'
sys.path.insert(0,str(SETUP))
from ladder_rollout_progress import RolloutProgress,aggregate
from install_rollout_progress import transform
from install_rollout_progress import transform_search


class Progress(unittest.TestCase):
    def test_out_of_order_tasks_keep_gather_order_and_exact_total(self):
        with tempfile.TemporaryDirectory() as directory:
            async def run():
                tracker=RolloutProgress(directory,target=3,batch_total=3,expected_workers=1,sequence=1,validation=False,global_step=1)
                async def work(n):await asyncio.sleep((3-n)*.005);return n
                tasks=[asyncio.create_task(work(n)) for n in range(3)]
                for task in tasks:task.add_done_callback(tracker.done)
                result=await asyncio.gather(*tasks)
                self.assertEqual(result,[0,1,2])
            asyncio.run(run())
            d=aggregate(directory);self.assertEqual(d['completed'],3);self.assertTrue(d['all_finished'])

    def test_errors_and_cancellations_are_not_successful_rollouts(self):
        with tempfile.TemporaryDirectory() as directory:
            async def run():
                tracker=RolloutProgress(directory,target=3,batch_total=3,expected_workers=1,sequence=1,validation=True,global_step=2)
                loop=asyncio.get_running_loop()
                good=loop.create_future();good.set_result(None)
                bad=loop.create_future();bad.set_exception(ValueError('test'))
                cancel=loop.create_future();cancel.cancel()
                for f in [good,bad,cancel]:tracker.done(f)
            asyncio.run(run());d=aggregate(directory)
            self.assertEqual((d['completed'],d['failed'],d['cancelled']),(1,1,1))

    def test_writes_are_throttled_and_worker_totals_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            class Done:
                def cancelled(self):return False
                def exception(self):return None
            for worker in ['a','b']:
                t=RolloutProgress(directory,target=128,batch_total=256,expected_workers=2,sequence=1,validation=False,global_step=1,worker_id=worker)
                for _ in range(128):t.done(Done())
                self.assertLessEqual(t.writes,4)
            d=aggregate(directory);self.assertEqual(d['completed'],256);self.assertTrue(d['all_finished'])

    def test_patch_is_opt_in_idempotent_and_keeps_gather(self):
        source='        outputs = await asyncio.gather(*tasks)\n        chunkes = prompts.chunk(len(self.agent_loop_workers))\n'
        updated=transform(source)
        self.assertIn('task.add_done_callback(progress.done)',updated)
        self.assertIn('outputs = await asyncio.gather(*tasks)',updated)
        self.assertIn('trainer.rollout_progress_dir',updated)
        self.assertEqual(transform(updated),updated)
        with self.assertRaises(ValueError):transform('unexpected source')

    def test_threaded_search_completion_has_no_lost_updates(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as directory:
            tracker=RolloutProgress(directory,target=2048,batch_total=2048,expected_workers=1,
                sequence=1,validation=True,global_step=None)
            with ThreadPoolExecutor(max_workers=32) as pool:
                list(pool.map(lambda _:tracker.record('completed'),range(2048)))
            self.assertEqual(aggregate(directory)['completed'],2048)
            self.assertTrue(aggregate(directory)['all_finished'])

    def test_search_patch_preserves_submission_and_result_order(self):
        source='''def evaluate_harness():
    max_workers = 2
    if max_workers == 1:
        outputs = [run_puzzle_rollout(example=example) for example in examples]
    else:
        futures = [pool.submit(
                        run_puzzle_rollout,
                        example=example) for example in examples]
        for idx, future in futures:
            outputs[idx] = future.result()
    metadata = {
        "test": True,
    }
    return summary

def append_trace(): pass
'''
        updated=transform_search(source)
        compile(updated,'search.py','exec')
        self.assertIn('outputs[idx] = future.result()',updated)
        self.assertIn('progress.record("completed")',updated)
        self.assertIn('progress.record("failed")',updated)
        self.assertEqual(transform_search(updated),updated)


if __name__=='__main__':unittest.main()
