#!/usr/bin/env bash
#SBATCH --job-name=au_r2p
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.out
# Paired rasterized-vs-path-traced evaluation of scripted + learned policies. Exports
# raster_vs_path_traced_eval.csv, failure_taxonomy.md, policy_eval_report.md.
# This (jwst-rl) run uses the sensor-fidelity R2P model. The image-in-the-loop
# version (Replicator-rendered observations under RaytracedLighting vs PathTracing
# feeding a visual policy) reuses the Benchmark generate_dataset.sbatch job per
# condition and is the scale-up; both share identical seeds + metric extraction.
#   sbatch ~/team/jwst_inspect/autonomous/sbatch/eval_r2p.sbatch
set -euo pipefail
ENV_DIR="${ENV_DIR:-/data/shared/env}"
PY="${JWST_RL_PY:-${ENV_DIR}/miniforge3/envs/jwst-rl/bin/python}"
SRC="$(cd "$(dirname "$0")/../src" && pwd)"
export OUT="${OUT:-/data/shared/checkpoints}"
"${PY}" "${SRC}/eval_r2p.py"
echo "R2P evaluation complete; see ${OUT}/raster_vs_path_traced_eval.csv"
