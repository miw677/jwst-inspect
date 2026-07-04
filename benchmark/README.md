# Group 2 - Synthetic Data and Perception Benchmark

We own the Omniverse Replicator pipeline, the labeled synthetic dataset, and
the perception baseline. We start on the Digital Twin scene but never block on
it (a stub scene works); we publish a stable interface the others consume. Our
GitLab fork is `nvidia-harvard/jwst_inspect-benchmark`; the project-wide git
rules are in [docs/gitlab_workflow.md](../docs/gitlab_workflow.md).

## How our folders are organized

Our tree inside the repo (`benchmark/` in every clone of the fork):

```
benchmark/
  README.md          this file
  src/               our code: Replicator generation, perception baseline, split logic
  sbatch/            our Slurm job scripts (one .sbatch per job type)
  interface/         the STABLE contract files other teams consume (versioned here)
    synthetic_schema_v0.json    the dataset schema (fields, dtypes, semantics)
```

On the workstation:

```
~/team = /data/groups/benchmark        our team space (writable by us, readable by all)
  jwst_inspect/                        the shared clone of our fork; rep-managed, always on current main;
                                       run sbatch jobs from here (stable paths)
  runs/, caches, backups               runtime output, never in git
~/jwst_inspect                         your personal clone of the fork; develop + push branches from here
/data/shared/datasets                  WE PUBLISH HERE: the synthetic dataset, data card, splits (read-only to others)
/data/shared/assets                    Group 1's published scenes (we consume, never write)
/data/shared/raw                       public datasets, read-only (FITS backgrounds, SPEED+ on the box)
/data/scratch/<you>                    big temp output (shards mid-generation); auto-cleaned after 14 days
```

## Where each thing goes

- Code and notebooks: `benchmark/src/` in your personal clone, shipped via
  branch + review.
- Slurm scripts: `benchmark/sbatch/`; scale knobs (`NUM_EPISODES`,
  `FRAMES_PER_EPISODE`, renderer, output dir) are env vars, never edits.
- The schema and data card templates: `benchmark/interface/`. A schema change
  is an interface release: the representative ships it upstream the same week.
- Generated data (episodes, shards, the fixed-seed sample, `data_card.md`,
  `dataset_manifest.json`, per-episode metadata): publish to
  `/data/shared/datasets/<version>/`. Never in git - the target scale is
  hundreds of GB; git carries the generator code and the schema instead.
- Perception baseline outputs (`perception_baseline_report.md`,
  `perception_metrics.json`, model weights): `/data/shared/checkpoints/perception`.
- Dataset shards mid-generation: `/data/scratch/<you>`, then move complete
  versions into `/data/shared/datasets`.
- SPEED+ for the transfer track: read from `/data/shared/raw/speed_plus` (33 GiB
  extracted, on the box only; `data/README.md` explains why and how to fetch it
  elsewhere).

## What we build

- `src/replicator_generate.py` - reproducible, fixed-seed Replicator
  generation: opens the scene, applies semantics from the label schema, renders
  episodes with domain randomization (camera pose on a standoff shell, sun,
  exposure, gold roughness) and parameterized anomaly injection, and writes
  RGB / depth / semantic + instance segmentation / camera params / per-frame
  metadata per `interface/synthetic_schema_v0.json`. Resumable (completed
  episodes are skipped).
- `src/perception_baseline.py` - a real UNet semantic-segmentation baseline
  with deterministic episode-level splits; reports per-class IoU, mIoU, pixel
  accuracy.

Every knob (scale, DR ranges, splits, hyperparameters) is an env var with an
inline default; nothing is hardcoded.

## Run

```bash
# Generate (1 GPU, Isaac Sim container; long partition for full scale):
sbatch ~/team/jwst_inspect/benchmark/sbatch/generate_dataset.sbatch
# Fast raster-mode pass for training data:
RENDERER=RaytracedLighting OUT_DIR=/data/shared/datasets/v1_raster \
  sbatch ~/team/jwst_inspect/benchmark/sbatch/generate_dataset.sbatch
# Train + score the perception baseline (1 GPU, jwst-rl env):
DATASET=/data/shared/datasets/v1 \
  sbatch ~/team/jwst_inspect/benchmark/sbatch/train_perception.sbatch
```

## Published interface (-> /data/shared/datasets, read-only to other teams)

- `synthetic_schema_v0.json`, `data_card.md` (in `interface/`).
- `replicator_generate.py` (copied alongside the data for provenance).
- `sample_dataset_fixed_seed/` - the small fixed-seed sample.
- `dataset_manifest.json` + per-episode `episode_metadata.json`.
- `perception_baseline_report.md` + `perception_metrics.json`
  (-> /data/shared/checkpoints/perception).

## Definition of done

The fixed-seed sample regenerates to the schema (RGB, depth, semantic +
instance segmentation, camera, inspector state, lighting, material, anomaly,
task, seed); domain randomization is parameterized code, never a hand-picked
list; a perception baseline reports metrics on documented splits; the data
card and data-quality report exist; regeneration under the same seed
reproduces the sample.
