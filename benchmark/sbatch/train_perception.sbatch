#!/usr/bin/env bash
#SBATCH --job-name=bm_perception
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out
# Train the perception baseline (UNet semantic segmentation) on the generated
# dataset. Host conda env (jwst-rl, torch cu128) - no container needed.
#   sbatch ~/team/jwst_inspect/benchmark/sbatch/train_perception.sbatch
set -euo pipefail
ENV_DIR="${ENV_DIR:-/data/shared/env}"
PY="${JWST_RL_PY:-${ENV_DIR}/miniforge3/envs/jwst-rl/bin/python}"
SRC="$(cd "$(dirname "$0")/../src" && pwd)"
export DATASET="${DATASET:-/data/shared/datasets/v1}"
export OUT="${OUT:-/data/shared/checkpoints/perception}"
export EPOCHS="${EPOCHS:-30}"
"${PY}" "${SRC}/perception_baseline.py"
