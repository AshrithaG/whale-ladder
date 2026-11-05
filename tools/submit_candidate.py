"""Submit the explicit-config PSC pipeline; works with login-node Python 3.6."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--account', required=True)
    parser.add_argument('--model-cache', type=Path)
    parser.add_argument('--data-from', type=Path)
    parser.add_argument('--fingerprint')
    parser.add_argument('--setup-only', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(repo), universal_newlines=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=str(repo), universal_newlines=True).strip():
        raise SystemExit('Submit from a clean candidate checkout')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    record = root / 'latest-submission.json'
    if record.exists():
        raise SystemExit('A submission record already exists: inspect/archive it before submitting again: ' + str(record))
    config = dict(LADDER_ROOT=str(root), LADDER_CODE_DIR=str(repo), EXPECTED_REVISION=revision)
    for key, value in [('LADDER_MODEL_CACHE', args.model_cache), ('CHESS_DATA_FROM', args.data_from)]:
        if value is not None:
            config[key] = str(value.resolve())
    if args.fingerprint:
        config['CHESS_DATA_FINGERPRINT'] = args.fingerprint
    path = root / 'job-config.json'
    path.write_text(json.dumps(config, indent=2) + '\n')
    setup = subprocess.check_output(['sbatch', '--parsable', '--export=ALL', '-A', args.account,
        'slurm/candidate_setup.sbatch', str(path)], cwd=str(repo), universal_newlines=True).strip().split(';')[0]
    result = dict(revision=revision, config=str(path), setup_job_id=setup, gpu_job_id=None)
    record.write_text(json.dumps(result, indent=2) + '\n')
    print('CPU setup job: ' + setup, flush=True)
    if not args.setup_only:
        gpu = subprocess.check_output(['sbatch', '--parsable', '--export=ALL', '-A', args.account,
            '--dependency=afterok:' + setup, 'slurm/candidate_validate.sbatch', str(path)],
            cwd=str(repo), universal_newlines=True).strip().split(';')[0]
        result['gpu_job_id'] = gpu
        record.write_text(json.dumps(result, indent=2) + '\n')
        print('GPU validation job: ' + gpu + ' (after setup succeeds)', flush=True)
    print('Submission record: ' + str(record), flush=True)


if __name__ == '__main__':
    main()
