"""GPU-container checks for the Modal wrapper; run using the experiment venv."""
import argparse
import csv
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def inspect_host():
    import torch

    if Path(sys.prefix).resolve() != Path('/opt/whale-work/venv').resolve():
        raise RuntimeError(f'Wrong experiment Python: {sys.prefix}')
    raw = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,name,memory.total,driver_version',
        '--format=csv,noheader,nounits'], text=True)
    rows = [dict(zip(('index', 'uuid', 'name', 'memory_mib', 'driver'),
                    (v.strip() for v in row))) for row in csv.reader(raw.splitlines())]
    normalize = lambda value: str(value).lower().removeprefix('gpu-')
    by_uuid = {normalize(row['uuid']): row for row in rows}
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError('Expected exactly two CUDA-visible GPUs')
    devices = []
    for i in range(2):
        prop = torch.cuda.get_device_properties(i)
        row = by_uuid[normalize(prop.uuid)]
        if 'H100' not in row['name'] or not 79000 <= int(row['memory_mib']) <= 83000:
            raise RuntimeError(f'Expected H100 80GB, got {row}')
        devices.append(dict(row, cuda_ordinal=i, cuda_capability=list(torch.cuda.get_device_capability(i))))
    if devices[0]['uuid'] == devices[1]['uuid']:
        raise RuntimeError('GPU UUIDs are not distinct')
    cgroup = {}
    for name in ('cpu.max', 'memory.max', 'cpuset.cpus.effective'):
        p = Path('/sys/fs/cgroup') / name
        cgroup[name] = p.read_text().strip() if p.exists() else None
    if len(os.sched_getaffinity(0)) < 24:
        raise RuntimeError('Fewer than 24 CPUs available in affinity mask')
    if cgroup['cpu.max']:
        quota, period = cgroup['cpu.max'].split()
        if quota != 'max' and int(quota) / int(period) < 24:
            raise RuntimeError('CPU quota is below the requested 24 cores')
    if cgroup['memory.max'] not in (None, 'max') and int(cgroup['memory.max']) < 256 * 1024**3:
        raise RuntimeError('Container memory limit is below 256 GiB')
    free = shutil.disk_usage('/opt/whale-work').free
    if free < 200 * 1024**3:
        raise RuntimeError(f'Less than 200 GiB scratch available: {free}')
    return dict(devices=devices, ladder_job_gpus=','.join(d['uuid'] for d in devices),
        original_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        original_nvidia_visible_devices=os.environ.get('NVIDIA_VISIBLE_DEVICES'),
        cpu_affinity=sorted(os.sched_getaffinity(0)), cgroup=cgroup,
        meminfo=Path('/proc/meminfo').read_text(), scratch_free_bytes=free,
        torch=torch.__version__, torch_cuda=torch.version.cuda, nccl=torch.cuda.nccl.version(),
        python_prefix=sys.prefix)


def check_nccl():
    import torch
    import torch.distributed as dist

    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=90))
    try:
        value = torch.tensor([rank + 1.0], device=f'cuda:{rank}')
        dist.all_reduce(value)
        torch.cuda.synchronize()
        if value.item() != 3.0:
            raise RuntimeError(f'NCCL all-reduce returned {value.item()}')
        print(f'NCCL rank {rank}: all-reduce passed', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--nccl', action='store_true')
    args = parser.parse_args()
    if args.nccl:
        check_nccl()
    else:
        if args.output is None:
            parser.error('--output is required for host inspection')
        args.output.write_text(json.dumps(inspect_host(), indent=2) + '\n')
