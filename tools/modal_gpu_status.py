"""Read live telemetry from an existing Modal run; starts no compute."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def load_snapshot(volume):
    """An updating Volume file may not yet have a coherent external snapshot."""
    try:
        return json.loads(b''.join(volume.read_file('/modal-control/live.json')))
    except (UnicodeError, ValueError):
        # Older history rows are immutable. Never infer zero progress from an
        # incomplete latest row; show the actual age of the last valid sample.
        history = b''.join(volume.read_file('/modal-control/telemetry.jsonl'))
        for line in reversed(history.splitlines()):
            try:
                row = json.loads(line)
            except (UnicodeError, ValueError):
                continue
            if isinstance(row, dict) and 'time' in row:
                row['snapshot_source'] = 'latest complete telemetry-history row'
                return row
        raise RuntimeError('No complete telemetry snapshot is available yet; no rollout count inferred')


def describe(live):
    lines = [f"Observed: {live.get('checked_utc', 'unavailable')}"]
    if 'time' in live:
        lines.append(f"Snapshot age: {max(0, time.time()-live['time']):.0f}s")
    if live.get('snapshot_source'):
        lines.append('Snapshot source: ' + live['snapshot_source'])
    if live.get('observation_error'):
        lines.append('Observation error: ' + live['observation_error'])
        return '\n'.join(lines)
    phase = live.get('phase') or {}
    lines.append(f"Container phase: {phase.get('phase', 'not reported')} ({phase.get('state', 'running')})")
    for row in live.get('controller_phases', {}).values():
        if row:
            elapsed = (row.get('ended_unix') or live['time']) - row['started_unix']
            lines.append(f"Controller: {row['phase']} ({row['status']}, {elapsed:.0f}s)")
    for name, row in live.get('rollouts', {}).items():
        if row.get('available'):
            age = live['time'] - datetime.fromisoformat(row['oldest_worker_update_utc']).timestamp()
            kind='validation/test' if row.get('validation') else 'training'
            lines.append(f"{name} ({kind}): completed {row['completed']}/{row['batch_total']}; "
                f"failed {row['failed']}; cancelled {row['cancelled']}; "
                f"workers {row['workers_reporting']}/{row['expected_workers']}; oldest snapshot {age:.0f}s")
    for name, row in live.get('training', {}).items():
        lines.append(f"{name}: generated {row['generated_rollouts']}; accepted {row['accepted_samples']}; "
                     f"optimizer updates {row['optimizer_updates']}")
    ram = live.get('cpu_ram') or {}
    if ram:
        lines.append(f"CPU: {ram['cpu_percent']:.1f}% (all visible CPUs); RAM used "
                     f"{ram['used_bytes']/2**30:.1f}/{ram['total_bytes']/2**30:.1f} GiB")
    if live.get('gpus_csv'):
        lines.append('GPUs (index, UUID, used/total MiB, utilization %, watts):\n' + live['gpus_csv'].strip())
    for server in live.get('inference', []):
        lines.append(f"Inference PID {server['pid']} port {server['port']}: "
            f"HTTP model responses={server.get('vllm:request_success_total', 0):.0f}; "
            f"running={server.get('vllm:num_requests_running', 0):.0f}; "
            f"waiting={server.get('vllm:num_requests_waiting', 0):.0f}; "
            f"generated tokens={server.get('vllm:generation_tokens_total', 0):.0f}; "
            f"tokens/s={server.get('generation_tokens_per_second', 'not sampled yet')}")
    return '\n'.join(lines)


def main():
    import modal
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True, type=Path)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    volume = modal.Volume.from_name(receipt['plan']['results_volume'])
    live = load_snapshot(volume)
    # Finalization can occur after the last heartbeat; do not present that
    # older heartbeat as evidence the App is still executing.
    live['phase'] = json.loads(b''.join(volume.read_file('/modal-control/status.json')))
    print(json.dumps(live, indent=2) if args.json else describe(live))


if __name__ == '__main__':
    main()
