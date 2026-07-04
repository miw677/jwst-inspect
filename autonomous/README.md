# Group 3 - Autonomous Inspection Policy and R2P Evaluation

We own the inspection environment, the scripted + learned baselines, and the
rasterized-vs-path-traced (R2P) evaluation. We start on bounding-geometry
dynamics and never block on the other teams; we publish checkpoints and the
eval to `/data/shared/checkpoints`. Our GitLab fork is
`nvidia-harvard/jwst_inspect-autonomous`; the project-wide git rules are in
[docs/gitlab_workflow.md](../docs/gitlab_workflow.md).

## How our folders are organized

Our tree inside the repo (`autonomous/` in every clone of the fork):

```
autonomous/
  README.md          this file
  src/               our code: env, scripted policy + gate, PPO, BC, R2P runner
  sbatch/            our Slurm job scripts (one .sbatch per job type)
  interface/         the STABLE contract files other teams consume (versioned here)
```

On the workstation:

```
~/team = /data/groups/autonomous       our team space (writable by us, readable by all)
  jwst_inspect/                        the shared clone of our fork; rep-managed, always on current main;
                                       run sbatch jobs from here (stable paths)
  runs/                                training/eval run outputs (TensorBoard, rollouts); never in git
  oci-*/ , isaac-cache/, ov-cache/     container + runtime caches; never in git
~/jwst_inspect                         your personal clone of the fork; develop + push branches from here
/data/shared/checkpoints               WE PUBLISH HERE: policy checkpoints + R2P eval (read-only to others)
/data/shared/assets                    Group 1's published scenes (we consume, never write)
/data/shared/datasets                  Group 2's published dataset (we consume, never write)
/data/scratch/<you>                    big temp output (raw rollouts); auto-cleaned after 14 days
```

## Where each thing goes

- Code and notebooks: `autonomous/src/` in your personal clone, shipped via
  branch + review.
- Slurm scripts: `autonomous/sbatch/`; training scale (`TOTAL_STEPS`,
  `NUM_ENVS`, output dirs) are env vars, never edits.
- Published checkpoints (`ppo_policy_checkpoint.pt`, `bc_policy_checkpoint.pt`)
  and the eval outputs (`raster_vs_path_traced_eval.csv`, `failure_taxonomy.md`,
  `policy_eval_report.md`): `/data/shared/checkpoints`. Checkpoints never go in
  git; git carries the code that produced them.
- In-progress training state, TensorBoard logs, rollouts: `~/team/runs/` (kept
  out of git by `.gitignore`); track every run in MLflow (`http://jwst-ws:5000`).
- Raw rollout dumps and other large temporaries: `/data/scratch/<you>`.
- Contract files other teams read (metric column definitions, eval seeds
  policy): `autonomous/interface/`; changes are an interface release the
  representative ships upstream the same week.

## What we build

- `src/free_flyer_env.py` - real 6-DoF zero-g rigid-body env (Newton-Euler)
  with the published safety geometry (approach corridor, keep-out, standoff,
  abort, relative-velocity limit), viewpoint-coverage reward, comms latency,
  and sensor noise. Vectorized for fast PPO; every constant is an env knob.
- `src/scripted_policy.py` - the deterministic known-safe baseline AND the
  credibility hard gate (`gate()`): zero keep-out breaches, bounded velocity,
  real coverage. This gate must pass before any learned-policy claim.
- `src/train_ppo.py` - real clipped-PPO + GAE actor-critic; checkpoints
  `ppo_policy_checkpoint.pt`.
- `src/behavior_cloning.py` - BC from scripted demonstrations ->
  `bc_policy_checkpoint.pt`.
- `src/eval_r2p.py` - paired R2P runner: same policy + seeds under rasterized
  vs path-traced sensor fidelity; exports coverage, mean standoff error, safety
  violations, abort rate, relative velocity, and the R2P gap.

## Run

```bash
# Hard gate (must pass first):
/data/shared/env/miniforge3/envs/jwst-rl/bin/python \
  ~/team/jwst_inspect/autonomous/src/scripted_policy.py
# PPO (1 GPU, long partition for the headline run):
sbatch ~/team/jwst_inspect/autonomous/sbatch/train_ppo.sbatch
# Behavior cloning:
OUT=/data/shared/checkpoints/bc \
  sbatch ~/team/jwst_inspect/autonomous/sbatch/train_ppo.sbatch   # or run behavior_cloning.py
# R2P evaluation (scripted + learned):
sbatch ~/team/jwst_inspect/autonomous/sbatch/eval_r2p.sbatch
```

## R2P: two fidelity conditions

The env dynamics are identical across conditions; the appearance/sensor
fidelity differs. `rasterized` = clean low-noise observations a fast renderer
yields; `path_traced` = specular-glare sensor noise + intermittent depth
dropout that physically-accurate path tracing exposes on the gold optics. The
gap is the rasterized-minus-path-traced metric delta. The image-in-the-loop
scale-up feeds Replicator-rendered frames (RaytracedLighting vs PathTracing,
the Benchmark pipeline) to a visual policy with the same seeds + metric
extraction.

## Published interface (-> /data/shared/checkpoints, read-only to other teams)

`scripted_policy.py`, `ppo_policy_checkpoint.pt`, `bc_policy_checkpoint.pt`,
`raster_vs_path_traced_eval.csv`, `failure_taxonomy.md`, `policy_eval_report.md`.

## Definition of done

The env runs headless under Slurm; the scripted baseline and at least one
learned policy (BC + PPO) are trained and checkpointed; the paired R2P runner
exports the metrics and the R2P gap under identical seeds; the report and
failure taxonomy are published with every number traced to a stored artifact;
the scripted credibility gate passes before any headline claim.
