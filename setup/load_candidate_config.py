"""Safely quoted exports from an explicit Slurm config; Python 3.6 compatible."""
import json
from pathlib import Path
import shlex
import sys

required = ('LADDER_ROOT', 'LADDER_CODE_DIR', 'EXPECTED_REVISION')
allowed = required + ('LADDER_MODEL_CACHE', 'CHESS_DATA_FROM', 'CHESS_DATA_FINGERPRINT',
                      'LADDER_REUSE_INSTALL_BASE', 'LADDER_VALIDATION_PROFILE')
config = json.loads(Path(sys.argv[1]).read_text())
for key in required:
    if not isinstance(config.get(key), str) or not config[key]:
        raise SystemExit('Missing configuration field: ' + key)
for key in allowed:
    if key in config:
        if not isinstance(config[key], str):
            raise SystemExit('Configuration field must be a string: ' + key)
        print('export ' + key + '=' + shlex.quote(config[key]))
