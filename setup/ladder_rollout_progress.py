"""Small, throttled local snapshots of completed chess-rollout tasks."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import time
import threading


class RolloutProgress:
    def __init__(self,directory,*,target,batch_total,expected_workers,sequence,validation,global_step,worker_id=None):
        self.directory=Path(directory)
        self.worker_id=str(os.getpid() if worker_id is None else worker_id)
        self.data=dict(worker_id=self.worker_id,target=target,batch_total=batch_total,
            expected_workers=expected_workers,batch_sequence=sequence,validation=validation,
            global_step=global_step,completed=0,failed=0,cancelled=0,
            started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        self.last_write=0.;self.writes=0;self.disabled=False
        self.lock=threading.Lock()
        self.flush()

    def done(self,task):
        kind='cancelled' if task.cancelled() else 'failed' if task.exception() is not None else 'completed'
        self.record(kind)

    def record(self,kind):
        if kind not in ('completed','failed','cancelled'):raise ValueError(kind)
        with self.lock:
            self.data[kind]+=1
            finished=sum(self.data[k] for k in ['completed','failed','cancelled'])
            if finished==self.data['target'] or finished%64==0 or time.monotonic()-self.last_write>=5:
                self.flush()

    def flush(self):
        if self.disabled:return
        d=self.data|{'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
        d['finished']=sum(d[k] for k in ['completed','failed','cancelled'])==d['target']
        try:
            self.directory.mkdir(mode=0o700,parents=True,exist_ok=True)
            path=self.directory/f'worker-{self.worker_id}.json'
            temp=path.with_suffix('.tmp');body=json.dumps(d,default=str)+'\n';temp.write_text(body);temp.replace(path)
            if d['finished']:
                final=self.directory/'finished';final.mkdir(exist_ok=True)
                (final/f'worker-{self.worker_id}-batch-{d["batch_sequence"]}.json').write_text(body)
            self.writes+=1;self.last_write=time.monotonic()
        except OSError as exc:
            # Observability must not turn a valid rollout into a failed rollout.
            self.disabled=True
            print(f'WARNING: rollout progress logging disabled: {exc}', flush=True)


def aggregate(directory):
    rows=[]
    for p in Path(directory).glob('worker-*.json'):
        try:rows.append(json.loads(p.read_text()))
        except (OSError,ValueError):continue
    return aggregate_rows(rows)


def aggregate_rows(rows):
    """Use only the newest batch; never add training and validation together."""
    if not rows:return {'available':False}
    sequence=max(r['batch_sequence'] for r in rows)
    rows=[r for r in rows if r['batch_sequence']==sequence]
    result={k:rows[0].get(k) for k in ['batch_sequence','validation','global_step','batch_total','expected_workers']}
    result.update(available=True,workers_reporting=len(rows))
    for k in ['completed','failed','cancelled']:result[k]=sum(r[k] for r in rows)
    result['all_finished']=len(rows)==result['expected_workers'] and all(r['finished'] for r in rows)
    result['updated_utc']=max(r['updated_utc'] for r in rows)
    result['oldest_worker_update_utc']=min(r['updated_utc'] for r in rows)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('output',type=Path,nargs='?');a=p.parse_args()
    body=json.dumps(aggregate(a.directory),indent=2)+'\n'
    if a.output:a.output.write_text(body)
    else:print(body,end='')
