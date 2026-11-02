"""Read-only telemetry for one isolated run; never starts work or retries it."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'setup'))
from ladder_rollout_progress import aggregate

COUNTERS = {'vllm:num_requests_running', 'vllm:num_requests_waiting',
            'vllm:kv_cache_usage_perc', 'vllm:num_preemptions_total',
            'vllm:generation_tokens_total', 'vllm:request_success_total'}


def parse_metrics(body):
    values = {}
    for line in body.splitlines():
        if not line or line.startswith('#'):
            continue
        name = line.split('{', 1)[0].split(' ', 1)[0]
        if name in COUNTERS:
            values[name] = values.get(name, 0) + float(line.rsplit(' ', 1)[1])
    return values


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def progress(root):
    result = {}
    paths = list((root / 'cycles').glob('*/c*-train/rollout-progress'))
    paths += list((root / 'cycles').glob('*/final-test/rollout-progress'))
    paths += list((root / 'harness-runs').glob('*/logs/**/rollout-progress'))
    for path in paths:
        result[str(path.relative_to(root))] = aggregate(path)
    return result


def training_totals(root):
    rows = {}
    for path in (root / 'cycles').glob('*/c*-train/training-metrics.jsonl'):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue  # An active writer may not have finished its last row.
            data = row.get('data', {})
            if 'online_rsft/total' in data:
                rows[str(path.parent.relative_to(root))] = dict(step=row['step'],
                    generated_rollouts=data['online_rsft/total'],
                    accepted_samples=data.get('online_rsft/accepted'),
                    optimizer_updates=data.get('actor/sft_updates'))
    return rows


def inference_metrics(psutil, opener):
    result, errors = [], []
    for process in psutil.process_iter(['pid', 'name']):
        try:
            name = process.info['name'] or ''
            if 'vLLMHttpServer' not in name and name not in ('vllm', 'vllm serve'):
                continue
            for conn in process.net_connections(kind='tcp'):
                if conn.status != 'LISTEN' or ':' in conn.laddr.ip:
                    continue
                port = conn.laddr.port
                with opener.open(f'http://127.0.0.1:{port}/metrics', timeout=1) as response:
                    metrics = parse_metrics(response.read().decode())
                if metrics:
                    result.append(dict(pid=process.pid, port=port, **metrics))
        except (psutil.Error, OSError, ValueError) as exc:
            errors.append(dict(pid=process.pid, type=type(exc).__name__))
    return result, errors


def snapshot(root, psutil, opener):
    now = time.time()
    row = dict(time=now, checked_utc=datetime.now(timezone.utc).isoformat(),
        phase=read_json(root / 'modal-control/status.json'),
        controller_phases={str(p.parent.relative_to(root)): read_json(p)
                           for p in (root / 'cycles').glob('*/current-phase.json')},
        rollouts=progress(root), training=training_totals(root))
    memory = psutil.virtual_memory()
    row['cpu_ram'] = dict(total_bytes=memory.total, available_bytes=memory.available,
                          used_bytes=memory.used, cpu_percent=psutil.cpu_percent())
    try:
        row['gpus_csv'] = subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,power.draw',
            '--format=csv,noheader,nounits'], text=True, timeout=5)
        row['gpu_columns'] = ['index', 'uuid', 'memory_used_mib', 'memory_total_mib', 'utilization_percent', 'power_watts']
    except (OSError, subprocess.SubprocessError) as exc:
        row['gpu_observation_error'] = type(exc).__name__
    row['inference'], row['inference_observation_errors'] = inference_metrics(psutil, opener)
    return row


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    control = args.root / 'modal-control'
    control.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    previous = {}
    while True:
        try:
            row = snapshot(args.root, psutil, opener)
            for server in row['inference']:
                identity = (server['pid'], server['port'])
                tokens = server.get('vllm:generation_tokens_total', 0)
                old = previous.get(identity)
                if old and row['time'] > old[0] and tokens >= old[1]:
                    server['generation_tokens_per_second'] = (tokens - old[1]) / (row['time'] - old[0])
                previous[identity] = (row['time'], tokens)
        except Exception as exc:
            row = dict(time=time.time(), observation_error=f'{type(exc).__name__}: {exc}')
        body = json.dumps(row) + '\n'
        temp = control / 'live.tmp'
        temp.write_text(body)
        temp.replace(control / 'live.json')
        with (control / 'telemetry.jsonl').open('a') as stream:
            stream.write(body)
        if args.once or stop.wait(args.interval):
            break


if __name__ == '__main__':
    main()
