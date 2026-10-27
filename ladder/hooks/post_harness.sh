#!/usr/bin/env bash
# Runs after every harness-search phase, and from the Slurm trap on exit.
# Stops the solver, proposer and proxy so the next weight phase gets all GPUs.
set -uo pipefail
source "$LADDER_HOME/ladder/lib.sh"

timing_event post_harness_start "\"round\": ${2:-0}"
stop_servers
wait_gpus_free 300
timing_event post_harness_end "\"round\": ${2:-0}"
