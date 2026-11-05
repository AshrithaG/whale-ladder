"""Install opt-in counters in the runtime checkout, preserving a source backup."""
import argparse
import hashlib
import json
from pathlib import Path


def transform(source):
    if '# LADDER_ROLLOUT_PROGRESS' in source:return source
    gather='        outputs = await asyncio.gather(*tasks)\n'
    chunk='        chunkes = prompts.chunk(len(self.agent_loop_workers))\n'
    if source.count(gather)!=1 or source.count(chunk)!=1:
        raise ValueError('Unexpected agent-loop source; refusing an ambiguous patch')
    tracking='''        # LADDER_ROLLOUT_PROGRESS: callbacks retain gather's original output order.
        progress_dir = OmegaConf.select(self.config, "trainer.rollout_progress_dir", default=None)
        if progress_dir:
            from verl.experimental.agent_loop.ladder_rollout_progress import RolloutProgress
            self._ladder_progress_sequence = getattr(self, "_ladder_progress_sequence", 0) + 1
            progress = RolloutProgress(progress_dir, target=len(tasks),
                batch_total=batch.meta_info.get("_ladder_progress_total", len(tasks)),
                expected_workers=batch.meta_info.get("_ladder_progress_workers", 1),
                sequence=self._ladder_progress_sequence,
                validation=bool(batch.meta_info.get("validate", False)),
                global_step=batch.meta_info.get("global_steps"))
            for task in tasks:
                task.add_done_callback(progress.done)
'''
    metadata='''        if OmegaConf.select(self.config, "trainer.rollout_progress_dir", default=None):
            prompts.meta_info["_ladder_progress_total"] = len(prompts)
            prompts.meta_info["_ladder_progress_workers"] = len(self.agent_loop_workers)
'''
    return source.replace(gather,tracking+gather).replace(chunk,metadata+chunk)


def transform_search(source):
    """Observe each completed attempt without changing submission/result order."""
    if '# LADDER_SEARCH_PROGRESS' in source:return source
    start=source.index('def evaluate_harness(')
    end=source.index('\ndef append_trace(',start)
    body=source[start:end]
    marker='    if max_workers == 1:\n'
    if (body.count(marker)!=1 or body.count('run_puzzle_rollout(')!=1
            or body.count('                        run_puzzle_rollout,')!=1):
        raise ValueError('Unexpected search evaluation source')
    tracking='''    # LADDER_SEARCH_PROGRESS: only logging; original output order is retained.
    import os
    import time
    from autoharness_chess_puzzle.ladder_rollout_progress import RolloutProgress
    progress = RolloutProgress(output / "rollout-progress", target=len(examples),
        batch_total=len(examples), expected_workers=1, sequence=1,
        validation=True, global_step=None) if os.environ.get("LADDER_ROLLOUT_PROGRESS") == "1" else None
    rollout_started = time.monotonic()
    def tracked_rollout(**kwargs):
        try:
            result = run_puzzle_rollout(**kwargs)
        except BaseException:
            if progress: progress.record("failed")
            raise
        else:
            if progress: progress.record("completed")
            return result
'''
    body=body.replace('run_puzzle_rollout(', 'tracked_rollout(').replace(
        '                        run_puzzle_rollout,','                        tracked_rollout,')
    body=body.replace(marker,tracking+marker)
    body=body.replace('    metadata = {\n','    rollout_finished = time.monotonic()\n    metadata = {\n')
    body=body.replace('    return summary\n','''    if progress:
        (output / "evaluation-timing.json").write_text(json.dumps({
            "attempt_execution_seconds": rollout_finished - rollout_started,
            "aggregation_and_write_seconds": time.monotonic() - rollout_finished,
            "note": "Attempt execution includes model calls, harness work and reward checking."
        }, indent=2) + "\\n")
    return summary
''')
    return source[:start]+body+source[end:]


def install(domain,backup):
    target=domain/'verl/experimental/agent_loop/agent_loop.py'
    if 'WHALE_Pilot' in target.resolve().parts:raise ValueError('Reference WHALE_Pilot is read-only')
    source=target.read_text();updated=transform(source)
    search=domain/'autoharness_chess_puzzle/runner.py'
    search_source=search.read_text();search_updated=transform_search(search_source)
    compile(updated,str(target),'exec');compile(search_updated,str(search),'exec')
    backup.mkdir(parents=True,exist_ok=True)
    old=backup/('agent_loop-'+hashlib.sha256(source.encode()).hexdigest()[:12]+'.py')
    if not old.exists():old.write_text(source)
    helper=target.with_name('ladder_rollout_progress.py')
    search_old=backup/('search-runner-'+hashlib.sha256(search_source.encode()).hexdigest()[:12]+'.py')
    if not search_old.exists():search_old.write_text(search_source)
    helper_source=Path(__file__).with_name('ladder_rollout_progress.py').read_text()
    for p,s in [(helper,helper_source),(search.with_name(helper.name),helper_source),(target,updated),(search,search_updated)]:
        compile(s,str(p),'exec')
        t=p.with_name(p.name+'.progress-staging');t.write_text(s)
        t.chmod(p.stat().st_mode if p.exists() else 0o644);t.replace(p)
    (backup/'rollout-progress-install.json').write_text(json.dumps(dict(
        source_before_sha256=hashlib.sha256(source.encode()).hexdigest(),
        source_after_sha256=hashlib.sha256(updated.encode()).hexdigest(),
        helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
        search_before_sha256=hashlib.sha256(search_source.encode()).hexdigest(),
        search_after_sha256=hashlib.sha256(search_updated.encode()).hexdigest(),
        backup=str(old), behavior='task-completion callbacks; gather ordering retained'),indent=2)+'\n')
    print('Installed opt-in rollout counters; backup:',old)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('domain',type=Path);p.add_argument('backup',type=Path);a=p.parse_args();install(a.domain,a.backup)
