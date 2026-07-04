# Group 1 - Digital Twin and Asset Benchmark

We own the reusable JWST OpenUSD benchmark scene and the inspector microsat
asset. We depend on no other team; we publish a stable interface the others
consume. Our GitLab fork is `nvidia-harvard/jwst_inspect-digital_twin`; the
project-wide git rules are in [docs/gitlab_workflow.md](../docs/gitlab_workflow.md).

## How our folders are organized

Our tree inside the repo (`digital_twin/` in every clone of the fork):

```
digital_twin/
  README.md          this file
  src/               our code: scene build, inspector build, acceptance tests, converters
  sbatch/            our Slurm job scripts (one .sbatch per job type)
  interface/         the STABLE contract files other teams consume (versioned here)
    semantic_labels_v0.json     class taxonomy + node mapping
    coordinate_frames.md        frame + unit conventions
    validation_renders/         small reference frames checked in for review
  usd_files/         working USD/STL conversions we keep versioned (small files only)
```

On the workstation:

```
~/team = /data/groups/digital_twin     our team space (writable by us, readable by all)
  jwst_inspect/                        the shared clone of our fork; rep-managed, always on current main;
                                       run sbatch jobs from here (stable paths)
  runs/, caches, backups               runtime output, never in git
~/jwst_inspect                         your personal clone of the fork; develop + push branches from here
/data/shared/assets                    WE PUBLISH HERE: scenes, materials, the asset manifest (read-only to others)
/data/shared/raw                       public datasets, read-only (geometry, textures, star maps, ephemeris)
/data/scratch/<you>                    big temp output (WIP renders, conversions); auto-cleaned after 14 days
```

## Where each thing goes

- Code and notebooks: `digital_twin/src/` in your personal clone, shipped via
  branch + review. Private experiments can start in `~`, but anything a
  teammate might need moves to `src/`.
- Slurm scripts: `digital_twin/sbatch/`; parameterize with env vars, never
  hardcode paths or sizes.
- Stable contracts (label schema, frame conventions, small reference renders):
  `digital_twin/interface/`. Changing these is an interface release: the
  representative ships it upstream the same week so other teams see it in git.
- Built artifacts (USD scenes, the inspector asset, MDL materials, validation
  renders at full resolution, `asset_manifest.csv` + sidecars): publish to
  `/data/shared/assets`. These are too big for git; the manifest carries
  bytes + sha256 so every published file is verifiable.
- Work in progress (partial conversions, test renders): `/data/scratch/<you>`.
- The NASA source geometry stays in `/data/shared/raw/jwst_geometry` (read-only)
  and, mirrored, in `data/jwst_geometry` in the repo; never copy it into our tree.

## What we build

- `src/jwst_scene.py` - converts the NASA 3D Resources GLB into a USD stage:
  the JWST component hierarchy with `semanticClass` tags (18 primary-mirror
  segment frames, secondary + struts, 5 sunshield layers, bus, solar array,
  antenna, star trackers, trusses, backplane), parameterized safety regions
  (keep-out, standoff shell, approach corridor), `lighting` variant set, and a
  camera.
- `src/build_inspector.py` - the free-flyer inspector USD (bus, solar arrays,
  12 microthruster sites for 6-DoF, reaction wheels, stereo RGB, depth/LiDAR,
  IR, star tracker, IMU, antenna) as labeled frames the Autonomous team mounts
  dynamics onto.
- `src/scene_load_test.py` - the acceptance gate (load, hierarchy, labels,
  safety regions, lighting, camera). Exits non-zero on any failure.

Every dimension is an env knob with an inline default (e.g. `JWST_STANDOFF_M`,
`JWST_KEEPOUT_M`, `JWST_CORRIDOR_HALF_DEG`); nothing dimensional is hardcoded.

## Run

```bash
# Build + acceptance test (CPU, usd-core env):
sbatch ~/team/jwst_inspect/digital_twin/sbatch/build_scene.sbatch
# Path-traced validation renders (1 GPU, Isaac Sim container, RTX PathTracing):
sbatch ~/team/jwst_inspect/digital_twin/sbatch/render_validation.sbatch
```

Local (outside Slurm), in the `jwst-usd` env:

```bash
PY=/data/shared/env/miniforge3/envs/jwst-usd/bin/python
SRC=~/team/jwst_inspect/digital_twin/src
OUT_DIR=/data/shared/assets $PY $SRC/build_inspector.py
OUT_DIR=/data/shared/assets $PY $SRC/jwst_scene.py
SCENE=/data/shared/assets/jwst_inspect_scene_v1.usd $PY $SRC/scene_load_test.py
```

## Published interface (-> /data/shared/assets, read-only to other teams)

- `jwst_digital_twin_stub.usd` - fast structural stub (labels + safety regions).
- `jwst_inspect_scene_v1.usd` - full scene with converted NASA geometry.
- `inspector_microsat.usd` - the inspector asset.
- `semantic_labels_v0.json` - class taxonomy + GLB-node mapping (in `interface/`).
- `coordinate_frames.md` - frame + unit conventions (in `interface/`).
- `asset_manifest.csv` + `*.sidecar.json` - provenance, sizes, sha256.
- `validation_renders/` - path-traced frames + their sidecar.

## Definition of done

The scene loads in USD Composer / Kit and renders under RTX path tracing; the
full hierarchy, semantic labels, and safety regions are present; MDL material
and lighting variants exist with validation renders to prove them; the asset
manifest is complete and the load/reuse acceptance test passes; the interface
is published so the other teams consume it without waiting on us.
