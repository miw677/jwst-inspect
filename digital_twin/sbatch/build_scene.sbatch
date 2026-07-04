#!/usr/bin/env bash
#SBATCH --job-name=dt_build_scene
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
# Build the JWST-Inspect OpenUSD scene + inspector asset from the public NASA
# geometry, then run the acceptance load test. CPU-only (usd-core). Publishes to
# /data/shared/assets (run as a jwst-digital_twin member who can write there).
#   sbatch ~/team/jwst_inspect/digital_twin/sbatch/build_scene.sbatch
set -euo pipefail
ENV_DIR="${ENV_DIR:-/data/shared/env}"
PY="${JWST_USD_PY:-${ENV_DIR}/miniforge3/envs/jwst-usd/bin/python}"
SRC="$(cd "$(dirname "$0")/../src" && pwd)"
export OUT_DIR="${OUT_DIR:-/data/shared/assets}"
export JWST_GLB="${JWST_GLB:-/data/shared/raw/jwst_geometry/James_Webb_Space_Telescope_B.glb}"

echo "building inspector + scene -> ${OUT_DIR}"
"${PY}" "${SRC}/build_inspector.py"
"${PY}" "${SRC}/jwst_scene.py"
echo "running acceptance load test (gate)"
SCENE="${OUT_DIR}/jwst_inspect_scene_v1.usd" "${PY}" "${SRC}/scene_load_test.py"
echo "done"
