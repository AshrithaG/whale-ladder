#!/usr/bin/env bash
# Stands in for the prepared verl launcher (LADDER_TRAIN_LAUNCHER) in the mock run.
exec python "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fake_train.py" "$@"
