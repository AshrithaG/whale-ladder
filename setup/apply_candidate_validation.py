"""Apply current validation patch, migrating only its known original version."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

TARGET = Path('domains/chess_puzzles/meta_harness/meta_harness_chess_puzzle.py')


def apply(root, home):
    root, home = Path(root), Path(home)
    current = (home / 'patches/candidate-validation.patch').resolve()
    legacy = (home / 'patches/compatibility/candidate-validation-d5af0e6.patch').resolve()
    def check(patch, reverse=False):
        return subprocess.run(['git', 'apply', '--check', *(['--reverse'] if reverse else []), str(patch)],
                              cwd=root, capture_output=True).returncode == 0
    if check(current, reverse=True):
        return 'current'
    if check(current):
        subprocess.run(['git', 'apply', str(current)], cwd=root, check=True)
        return 'applied'
    if not check(legacy, reverse=True):
        raise RuntimeError('Unknown candidate-validation state; preserved checkout without changes')
    # Prove the whole known migration on a copy before publishing one file.
    with tempfile.TemporaryDirectory(prefix='candidate-validation-migration-') as tmp:
        staging = Path(tmp)
        target = staging / TARGET
        target.parent.mkdir(parents=True)
        shutil.copyfile(root / TARGET, target)
        subprocess.run(['git', 'apply', '--reverse', str(legacy)], cwd=staging, check=True)
        subprocess.run(['git', 'apply', str(current)], cwd=staging, check=True)
        destination = root / TARGET
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as out:
            temporary = Path(out.name)
            out.write(target.read_bytes())
        try:
            shutil.copymode(destination, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return 'migrated-original'


if __name__ == '__main__':
    print(apply(*sys.argv[1:]))
