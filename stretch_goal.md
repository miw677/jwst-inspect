# JWST-Inspect: Stretch Goals

This is the ambitious version of the project - how far JWST-Inspect can go beyond
the baseline deliverables. Your team's baseline "done" is enough on its own; this
document is the reach. JWST-Inspect is not a class exercise: it is an
NVIDIA-sponsored research platform meant to produce a top-tier publication, an
NVIDIA GTC 2027 showcase, an open-source release, and a simulation capability
credible enough that a NASA or commercial operator could evaluate autonomous
inspection procedures in it. The workstation is dedicated to this project;
multi-week runs are expected, not exceptional. We do not cap scope, data, or
compute to make something easier - if a real limit is hit, we solve it.

The three independent teams, the published interfaces, the hard safety gates, and
the reproducibility discipline from the baseline all still hold. What changes here
is the target altitude. The base infrastructure and datasets are already validated
end to end on the workstation (Isaac Sim 5.1.0 GUI + WebRTC streaming, Replicator
synthetic data with the full annotator set, Isaac Lab 2.3.2 training at ~155k
env-steps/s on one GPU, USD Composer streaming, both RTX PRO 6000 GPUs running
PyTorch with Blackwell kernels). Everything below is buildable on what is already
on the box.

## 1. What "stretch" means

The v1 targets are the floor, not the finish line. The stretch targets, per
dimension:

| Dimension | v1 floor | Stretch target |
|---|---|---|
| Orbital context | static scene, sun as a fixed light | ephemeris-driven Sun-Earth L2 halo orbit, time-varying sun vector, mission-epoch lighting |
| Geometry | 54-mesh GLB-derived scene | NASA model A detailed CAD with real texture maps, articulated 18-segment mirror, layered sunshield |
| Materials | 4 MDL variants, plausible values | MDL grounded in measured n/k data (Au, Al, Si), wrinkle/seam maps from the NASA source textures |
| Environment | black void + dome fill | 16k HDR all-sky star maps, TSIS-1 measured solar spectrum, Earth/Moon phase from ephemeris |
| Sensors | RGB + depth + segmentation | + RTX LiDAR, IMU, star-tracker view, thermal-IR proxy, exposure/noise/latency models |
| Dataset | fixed-seed sample, 200-600 GB | multi-task benchmark at 1M+ frames / 1-2 TB with anomaly suite and SPEED+ transfer track |
| Autonomy | scripted + BC + PPO on state obs | + vision-in-the-loop policies, MPC baseline, fuel/delta-v budgets, comms-latency robustness, multi-seed at scale |
| Evaluation | R2P gap on held-out seeds | + statistical confidence intervals, failure taxonomy grounded in the real C3 MMOD strike class, coverage-vs-fuel Pareto |
| Showcase | validation renders + short video | 4K path-traced mission film, live streamed demo, full mission rehearsal end-to-end |

## 2. Limits worth lifting

The Digital Twin team's first chapter scoped the work to inspection-scale
simulation and set aside full orbital mechanics, thermal dynamics, radiation
effects, and flight-readiness claims. Those exclusions were right for a first
chapter; they are no longer binding, because the data and infrastructure to lift
them are now on the workstation:

1. Full orbital mechanics is in reach. `/data/shared/raw/jwst_ephemeris/` holds
   14,602 real JPL Horizons state vectors for JWST (sun-centered and
   earth-centered, 2022-2026 at 6-hour steps). Teams can drive the L2 halo orbit,
   the sun vector, Earth/Moon geometry, and relative-motion dynamics (CR3BP or
   Clohessy-Wiltshire local frames) from measured data. The upgrade: inspection
   episodes are embedded in the real mission geometry, not a floating void.
2. Sun and sky are physically grounded. TSIS-1 measured solar spectral irradiance
   plus NASA SVS Deep Star Maps (8k/16k EXR) are in `/data/shared/raw/env_lighting/`.
   Lighting variation across an episode can follow the ephemeris, not an arbitrary
   randomization range.
3. Materials are measured, not eyeballed. refractiveindex.info n/k datasets for
   gold (mirror coating), aluminum and silicon (the actual JWST sunshield layer
   coatings) ground the MDL materials; the NASA model A source archive provides
   the real texture maps (gold reflectance, wrinkled sunshield).
4. Radiation and thermal enter as effects, not omissions. You do not need a
   thermal solver; you model what a camera and spacecraft experience: hot-side vs
   cold-side illumination asymmetry, sensor noise vs a temperature proxy,
   micrometeoroid (MMOD) surface damage. The real C3 segment micrometeoroid strike
   (documented in the commissioning report at `/data/shared/raw/inspector_refs/`)
   becomes a grounded anomaly class.
5. Flight relevance becomes a design goal, stated carefully. The defensible claim
   stays simulation-based. But the simulation can carry the constraints a flight
   procedure would meet: approach corridors, keep-out, abort, delta-v budgets,
   comms latency at L2 scale, camera-only navigation. The deliverable is a
   rehearsal and benchmarking environment an operator could load their own
   procedures into.

## 3. Physics and realism modules (cross-team foundations)

Each module is buildable now, has a data source on disk, and is published through
a stable interface so any team can consume it without waiting on another.

- Orbital mechanics module (Digital Twin + Autonomous). Ephemeris loader (parquet
  from the Horizons CSVs), local-frame transforms (LVLH/Hill frame around JWST),
  sun-vector-at-epoch function, optional CR3BP propagation for long arcs,
  Clohessy-Wiltshire or full relative dynamics for the inspector. Publish as
  `orbital_frames_v1` (code + parquet + doc).
- Environment module (Digital Twin). Dome light from the 16k star map oriented by
  epoch; sun DistantLight with TSIS-1-derived intensity and correct angular size
  (0.53 deg); Earth/Moon as epoch-positioned distant bodies. Publish as a scene
  layer `environment_epoch.usd` + generator script.
- Material module (Digital Twin). MDL library authored from measured n/k plus the
  model A texture maps: gold segments, aluminized kapton layers, doped-silicon
  hot-side layers, MLI foil, truss composites. Publish as `mdl_library_v1/` with
  per-material provenance sidecars.
- Sensor module (Benchmark + Autonomous). Camera intrinsics/exposure/noise models,
  depth noise, RTX LiDAR config, IMU noise, star-tracker camera preset, thermal-IR
  proxy render pass (emissive hot side). Publish as a sensor config + Replicator
  annotator set.
- Operations module (Autonomous). Delta-v accounting per episode, thruster
  saturation and minimum-impulse limits, comms-latency injection (one-way ~5 s at
  L2 scale; exact value from the ephemeris distances), abort logic. Publish as an
  env config + metrics columns.

## 4. Group 1 stretch roadmap - Digital Twin and Asset Benchmark

Phase A (floor, largely met): stub + v1 scene, labels, safety regions.

Phase B (fidelity):
- Rebuild the target from NASA model A (detailed CAD + real textures) with the 18
  primary segments as articulated, individually posed prims; five separate
  sunshield layer meshes; bus/ISIM/antenna/star trackers as labeled subtrees.
- Author the MDL library from measured optics (module above); validation renders
  should show glare behavior consistent with the reference imagery.
- Scene lighting at physical scale: the v1 scene's lights are far too dim and
  render near-black under path tracing; replace them with the environment module.
  Every validation render carries a sidecar with epoch, sun vector, and settings.
- Inspector microsat v2: sensor-suite placements, thruster cluster geometry,
  articulating solar array, MLI materials from the same library.

Phase C (hero):
- Epoch-driven scene: load any mission time, get correct sun/star/Earth geometry
  from the ephemeris.
- Anomaly authoring: MMOD strike decals/geometry on segment C3 (grounded in the
  commissioning report), sunshield tear/deformation variants, missing or degraded
  MLI patches - each a switchable USD variant with labels.
- 4K/8K path-traced hero renders and camera-path cinematics for GTC; every frame
  reproducible from a sidecar.
- Acceptance: the Chapter 2 test matrix (load, hierarchy, labels, frames, safety
  zones, materials resolve, path-traced render completes, manifest complete)
  automated as a validation script the team publishes.

## 5. Group 2 stretch roadmap - Synthetic Data and Perception Benchmark

Phase A (floor): schema freeze, stub-scene synthetic data, fixed-seed sample.

Phase B (scale + realism):
- Replicator pipeline over the fidelity scene with the sensor module: RGB, depth,
  normals, motion vectors, semantic + instance segmentation, 2D/3D boxes, camera
  params, inspector state, lighting/epoch metadata, anomaly masks, episode seeds.
  The annotator set already runs on the box.
- Domain randomization driven by physical parameters (epoch, sun angle, exposure,
  sensor noise, material variant, anomaly variant), all expressed as code.
- Scale to the full benchmark: episode-structured, multi-task (approach, survey,
  mirror inspection, anomaly detection), targeting 1M+ frames and 1-2 TB on the
  `long` partition with checkpointed, resumable generation. Storage is budgeted
  (section 9); do not shrink resolution or frame count to fit - raise the budget.
- Data card, splits, and a data-quality report with per-dimension coverage
  statistics (sun angles, ranges, anomaly rates).

Phase C (benchmark science):
- Perception baselines at two tiers: a from-scratch segmentation/detection
  baseline and a modern pretrained backbone fine-tune; report both on fixed splits
  with confidence intervals.
- SPEED+ transfer track: evaluate JWST-trained pose/detection models on SPEED+
  (staged at `/data/shared/raw/speed_plus/`, CC BY 4.0) and vice versa; report the
  cross-domain gap next to the R2P gap. This positions the dataset against the
  public state of the art.
- Anomaly detection suite over the grounded anomaly classes with
  precision/recall/F1 and false-alarm analysis.
- Publish: dataset, generator, card, baselines, and a leaderboard-style results
  table in `/data/shared/datasets/`.

## 6. Group 3 stretch roadmap - Autonomous Inspection and R2P Evaluation

Phase A (floor): 6-DoF zero-g env + scripted baseline + behavior cloning + PPO on
state observations. The env and PPO loop are validated on the box (256 envs at
~150k steps/s on one GPU). Plus: migrate the same task into an Isaac Lab task
config so GPU-parallel physics and the RL frameworks (rl_games/rsl_rl/skrl) are
available at scale.

Phase B (realism):
- Dynamics upgrades from the operations module: thruster minimum impulse,
  saturation, actuation latency, comms delay on observations/commands, fuel/delta-v
  accounting surfaced in the reward and metrics.
- Relative-motion dynamics in the ephemeris frame (Hill/CW terms) so approach and
  station-keeping behave like L2 proximity ops, not empty-space physics.
- Scripted baseline extended to a full mission profile: approach corridor,
  standoff hold, survey pattern over sunshield and mirror, anomaly close-up,
  retreat. This remains the credibility gate: no learned-policy claim until it
  passes on the fidelity scene.
- MPC or visual-servoing mid-tier baseline (design choice owned by the team).

Phase C (learning + evaluation at scale):
- Vision-in-the-loop policies: RGB/depth (and optionally LiDAR) observations
  rendered rasterized during training; multi-seed PPO/BC sweeps as multi-day `long`
  jobs across both GPUs with checkpoint/resume and experiment tracking.
- The R2P benchmark at scale: paired rasterized vs path-traced evaluation on
  identical seeds, swept over epoch/sun-angle and anomaly variants; coverage,
  standoff error, safety violations, abort rate, relative velocity, fuel used, and
  the R2P gap with confidence intervals.
- Failure taxonomy tied to physical causes (glare-driven perception loss,
  latency-induced overshoot, fuel-limited coverage) with example rollouts rendered
  for the paper and the film.

## 7. Cross-team showcase (opportunistic, never blocking)

- Mission rehearsal demo: fidelity scene + epoch lighting + learned policy + live
  streamed viewport; one continuous inspection mission from approach to anomaly
  report. This is the GTC centerpiece.
- The 60-90 s research video grows into a 4K path-traced mission film cut from
  Group 1 cinematics, Group 2 dataset visuals, and Group 3 rollouts.
- Integration happens only through each team's published interfaces; a team that
  is behind never blocks a team that is ahead.

## 8. Data foundation (staged and verified on the box)

Everything below is in `/data/shared/raw/` with size, source, checksum, and
retrieved-at recorded in `/data/shared/raw/manifest.csv` (see the README data
section for how to load each type):

- NASA model A detailed CAD (Maya source + real texture maps) + 3D-print STL kit +
  USDC/USDZ conversions (`jwst_geometry/`).
- JPL Horizons JWST ephemeris, sun- and earth-centered, 2022-2026 @ 6 h
  (`jwst_ephemeris/`).
- Deep Star Maps 2020 8k/16k EXR + galactic variant; TSIS-1 solar spectrum;
  Au/Al/Si measured optical constants (`env_lighting/`).
- JWST commissioning science-performance report incl. the C3 MMOD strike
  (`inspector_refs/`).
- 137 JWST FITS products: MIRI + WR 140 + NIRCam I2D frames (NGC 3132, SMACS J0723,
  Stephan's Quintet) (`jwst_imagery_fits/`); 23 MIRI MRS IFU cubes (`ifu_cubes/`).
- SPEED+ v2 spacecraft pose benchmark, extracted (~33 GB, CC BY 4.0)
  (`speed_plus/`).

Gaps to flag early: converted PBR textures from the Maya IFF maps (conversion is
part of Group 1's pipeline), any additional inspector-heritage references, and more
IFU targets if the volumetrics need variety.

## 9. Compute, storage, and long-run discipline

- Two RTX PRO 6000 Blackwell GPUs (96 GB each) under Slurm; partitions
  `interactive` (8 h, 1 GPU), `batch` (3 d), `long` (28 d). Multi-week jobs are
  normal; use `long` and checkpoint on a cadence so a restart never loses more
  than one interval.
- Storage: `/data` is a 3.5 TB redundant volume plus a 3.6 TB fast scratch, with
  NAS overflow. The 1-2 TB dataset target fits; coordinate with the admin before
  exceeding ~2.5 TB on `/data`.
- Experiment tracking: the shared MLflow at `http://jwst-ws:5000` for every
  training/data-generation run; TensorBoard per team for step curves. A multi-week
  run without tracking is a mistake.
- GUI: `jwst-gui isaac` (full Isaac Sim) and `jwst-gui composer` (USD Composer)
  stream over WebRTC; see the README for the connect steps.
- Burst path: DGX Spark / DGX Cloud can be added later without changing the
  workflow.

## 10. Publication and showcase timeline

Dates below were current when this was written; re-verify at each milestone.

| When | What |
|---|---|
| Jul-Aug 2026 | Phase B builds in all three teams; first fidelity-scene synthetic data; first Isaac Lab-scale training runs |
| Sep 2026 | ICRA 2027 submission window (deadline historically mid-September). Decision point: submit the R2P benchmark early, or hold for a stronger venue fit |
| Oct-Dec 2026 | Full-scale dataset generation + multi-seed training on `long`; hero renders; draft paper |
| Jan-Feb 2027 | Freeze benchmark v1; ablations + confidence intervals; film cut; GTC materials |
| Mar 1, 2027 | IROS 2027 paper deadline - robotics-track option |
| Mar 14-18, 2027 | NVIDIA GTC, San Jose - sponsor showcase, live demo + film |
| ~May 2027 | NeurIPS 2027 Datasets & Benchmarks (deadline historically May) - primary target if the dataset/benchmark is the strongest asset |
| Mid 2027 | Open-source release: scene, generator, baselines, eval harness, data card |

## 11. Where to start

Group 1: pull `jwst_geometry/model_a/`, convert the Maya IFF textures to PNG/EXR,
rebuild the target hierarchy from model A, adopt the environment-module design, and
fix scene lighting to physical scale (the v1-style lighting renders near-black
under path tracing; see section 4 Phase B).

Group 2: freeze `synthetic_schema_v1` including epoch/sun metadata and anomaly
fields; stand up the Replicator pipeline on the current scene now (validated
annotators: RGB, depth, semantic/instance segmentation, 2D boxes) and swap the
fidelity scene in when Group 1 publishes it; start the SPEED+ transfer design.

Group 3: port the free-flyer task into an Isaac Lab task config; add the
operations-module constraints one at a time with metrics; keep the scripted gate
green on every scene upgrade; wire experiment tracking into the training loop from
the first run.

None of this is required to pass the baseline. It is where the project goes if you
want to push it to a real publication, the GTC showcase, and something the space
industry could actually use.
