#!/usr/bin/env bash
#SBATCH --job-name=bm_gen
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=14-00:00:00
#SBATCH --output=slurm-%x-%j.out
# Generate the JWST-Inspect synthetic benchmark dataset with Replicator under RTX,
# at full scale. Scale = NUM_EPISODES x FRAMES_PER_EPISODE (defaults below target
# the 200-600 GB range; do NOT shrink to "make it run" - raise time/mem instead).
# Resumable: completed episodes are skipped. Run as a jwst-benchmark member.
#   sbatch ~/team/jwst_inspect/benchmark/sbatch/generate_dataset.sbatch
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-/data}"
IMG="${ISAAC_SIM_IMAGE:-${DATA_ROOT}/shared/containers/isaac-sim.sqsh}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/isaac-sim/python.sh}"
SRC="$(cd "$(dirname "$0")/../src" && pwd)"
OVC="${DATA_ROOT}/shared/ov-cache"; mkdir -p "${OVC}/kit"

export SCENE="${SCENE:-${DATA_ROOT}/shared/assets/jwst_inspect_scene_v1.usd}"
export OUT_DIR="${OUT_DIR:-${DATA_ROOT}/shared/datasets/v1}"
export RENDERER="${RENDERER:-PathTracing}"
export DATASET_SEED="${DATASET_SEED:-20260627}"
export NUM_EPISODES="${NUM_EPISODES:-2000}"
export FRAMES_PER_EPISODE="${FRAMES_PER_EPISODE:-100}"
export RENDER_W="${RENDER_W:-1024}" RENDER_H="${RENDER_H:-1024}" RENDER_SPP="${RENDER_SPP:-64}"
mkdir -p "${OUT_DIR}"

echo "generating ${NUM_EPISODES}x${FRAMES_PER_EPISODE} frames @ ${RENDER_W}x${RENDER_H} (${RENDERER}) -> ${OUT_DIR}"
cp "${SRC}/replicator_generate.py" "${OUT_DIR}/replicator_generate.py"   # provenance
srun --container-image="${IMG}" --container-writable \
  --container-mounts="${DATA_ROOT}:${DATA_ROOT},${OVC}/kit:/isaac-sim/kit/cache" \
  --container-env=SCENE,OUT_DIR,RENDERER,DATASET_SEED,NUM_EPISODES,FRAMES_PER_EPISODE,RENDER_W,RENDER_H,RENDER_SPP,DR_STANDOFF_MIN_M,DR_STANDOFF_MAX_M,DR_ELEV_MAX_DEG,DR_ANOMALY_PROB,DR_GOLD_ROUGH_MIN,DR_GOLD_ROUGH_MAX,DR_TASKS \
  bash -lc "${ISAAC_PYTHON} ${OUT_DIR}/replicator_generate.py"
echo "generation step complete; episodes under ${OUT_DIR}"
