#!/usr/bin/env bash
set -euo pipefail
mkdir -p /results
bash setup/install_mock.sh > /results/install.log 2>&1
bash ladder/mock_run.sh > /results/mock.log 2>&1
bash ladder/mock_persistent.sh > /results/persistent.log 2>&1
printf 'CPU mocks passed\n' > /results/passed
