#!/bin/bash
# Few-minute check of one proposer rung (no training).
#
#   sbatch -A <psc-account> slurm/proposer_check.sbatch whale_qwen27b
#
# 2 GPUs covers every rung (the 27B needs 2; the others use 1 or none).
#SBATCH --job-name=ladder-propcheck
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:2
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -uo pipefail
ARM="${1:?usage: sbatch slurm/proposer_check.sbatch ARM}"
cd "$SLURM_SUBMIT_DIR"
bash ladder/proposer_check.sh "$ARM"
