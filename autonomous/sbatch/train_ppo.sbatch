#!/usr/bin/env bash
#SBATCH --job-name=au_ppo
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=14-00:00:00
#SBATCH --output=slurm-%x-%j.out
# Train the PPO inspection policy on the 6-DoF free-flyer env (jwst-rl, torch cu128).
# Long partition: scale TOTAL_STEPS up for the headline run (do not shrink to finish
# faster). The scripted gate must pass first (scripted_policy.py).
#   sbatch ~/team/jwst_inspect/autonomous/sbatch/train_ppo.sbatch
set -euo pipefail
ENV_DIR="${ENV_DIR:-/data/shared/env}"
PY="${JWST_RL_PY:-${ENV_DIR}/miniforge3/envs/jwst-rl/bin/python}"
SRC="$(cd "$(dirname "$0")/../src" && pwd)"
export OUT="${OUT:-/data/shared/checkpoints/ppo}"
export TOTAL_STEPS="${TOTAL_STEPS:-50000000}"
export NUM_ENVS="${NUM_ENVS:-2048}"

echo "scripted credibility gate (must pass before learned-policy claims)"
"${PY}" "${SRC}/scripted_policy.py"
echo "PPO training -> ${OUT} (TOTAL_STEPS=${TOTAL_STEPS}, NUM_ENVS=${NUM_ENVS})"
"${PY}" "${SRC}/train_ppo.py"
