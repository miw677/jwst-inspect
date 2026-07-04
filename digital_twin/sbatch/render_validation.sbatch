#!/usr/bin/env bash
#SBATCH --job-name=dt_render
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out
# Path-traced validation renders of the published JWST scene under RTX (definition of done:
# "renders under RTX path tracing"). Loads the scene in the Isaac Sim container,
# enables PathTracing, and captures RGB from the scene camera via Replicator. Writes
# into <assets>/validation_renders/. Resolution + SPP are env knobs.
#   sbatch ~/team/jwst_inspect/digital_twin/sbatch/render_validation.sbatch
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-/data}"
IMG="${ISAAC_SIM_IMAGE:-${DATA_ROOT}/shared/containers/isaac-sim.sqsh}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/isaac-sim/python.sh}"
SCENE="${SCENE:-${DATA_ROOT}/shared/assets/jwst_inspect_scene_v1.usd}"
OUT="${OUT:-${DATA_ROOT}/shared/assets/validation_renders}"
OVC="${DATA_ROOT}/shared/ov-cache"; mkdir -p "${OUT}" "${OVC}/kit"
export SCENE OUT
export RENDER_W="${RENDER_W:-1280}" RENDER_H="${RENDER_H:-720}" RENDER_SPP="${RENDER_SPP:-128}"
export RENDER_FRAMES="${RENDER_FRAMES:-4}"

WORK="${DATA_ROOT}/scratch/${USER:-apmc5}/dt_render_${SLURM_JOB_ID:-local}"; mkdir -p "${WORK}"
cat > "${WORK}/render.py" <<'PY'
import os
from isaacsim import SimulationApp
sim = SimulationApp({"headless": True})
import carb, omni.usd
import omni.replicator.core as rep
s = carb.settings.get_settings()
s.set("/rtx/rendermode", "PathTracing")
s.set("/rtx/pathtracing/spp", int(os.environ.get("RENDER_SPP", "128")))
s.set("/rtx/pathtracing/totalSpp", int(os.environ.get("RENDER_SPP", "128")))
ctx = omni.usd.get_context()
ok = ctx.open_stage(os.environ["SCENE"])
print("opened stage:", os.environ["SCENE"], "ok=", ok, flush=True)
W = int(os.environ.get("RENDER_W", "1280")); H = int(os.environ.get("RENDER_H", "720"))
cam = "/World/InspectorCam"
rp = rep.create.render_product(cam, (W, H))
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=os.environ["OUT"], rgb=True)
writer.attach([rp])
with rep.trigger.on_frame(num_frames=int(os.environ.get("RENDER_FRAMES", "4"))):
    pass
rep.orchestrator.run()
rep.orchestrator.wait_until_complete()
print("path-traced validation render written to", os.environ["OUT"], flush=True)
sim.close()
PY

echo "rendering ${SCENE} under RTX path tracing (${RENDER_W}x${RENDER_H} @ ${RENDER_SPP} spp)"
srun --container-image="${IMG}" --container-writable \
  --container-mounts="${DATA_ROOT}:${DATA_ROOT},${OVC}/kit:/isaac-sim/kit/cache" \
  --container-env=SCENE,OUT,RENDER_W,RENDER_H,RENDER_SPP,RENDER_FRAMES \
  bash -lc "${ISAAC_PYTHON} ${WORK}/render.py"

# Sidecar describing exactly what produced these frames.
cat > "${OUT}/validation_render.sidecar.json" <<JSON
{"scene": "${SCENE}", "camera": "/World/InspectorCam", "renderer": "PathTracing",
 "spp": ${RENDER_SPP}, "resolution": "${RENDER_W}x${RENDER_H}", "frames": ${RENDER_FRAMES},
 "image_tag": "isaac-sim 5.1.0", "job": "${SLURM_JOB_ID:-local}"}
JSON
echo "renders:"; ls -la "${OUT}" || true
