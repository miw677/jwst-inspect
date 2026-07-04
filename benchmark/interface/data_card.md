# JWST-Inspect synthetic dataset - data card

## Summary

Labeled synthetic frames of the JWST-Inspect scene rendered with NVIDIA Omniverse
Replicator under RTX, for perception and the rasterized-to-path-traced (R2P) study.
Generated reproducibly from a fixed seed by `src/replicator_generate.py` against the
Digital Twin scene (`/data/shared/assets/jwst_inspect_scene_v1.usd`). Schema:
`interface/synthetic_schema_v0.json`. Classes: `groups/digital_twin/interface/semantic_labels_v0.json`.

## Per frame

RGB (uint8), depth (float32 m, `distance_to_camera`), semantic + instance
segmentation (+ label sidecars), camera intrinsics/extrinsics, inspector state
(position, standoff, keep-out flag), lighting params (variant, sun intensity/angle,
exposure), material params (gold roughness), anomaly fields, and task label.

## Domain randomization (parameterized; env knobs)

- Camera pose on a standoff shell: radius `DR_STANDOFF_MIN_M`..`DR_STANDOFF_MAX_M`,
  elevation +/- `DR_ELEV_MAX_DEG`, full azimuth, always looking at the target.
- Sun intensity + azimuth/elevation; exposure.
- Gold-mirror roughness `DR_GOLD_ROUGH_MIN`..`DR_GOLD_ROUGH_MAX` (specular DR - the
  core R2P stressor).
- Anomalies with probability `DR_ANOMALY_PROB`: missing/obscured component,
  unexpected glare, sunshield-deformation proxy, mirror-region anomaly, sensor
  confidence failure.

## Splits

Deterministic by `sha256(SEED:episode_id)`: train/val/test fractions
`DATASET_TRAIN_FRAC` (0.7) / `DATASET_VAL_FRAC` (0.15) / remainder. Because splits
are seed-derived, regeneration reproduces them.

## Reproducibility

Fixed `DATASET_SEED`; the renderer (`PathTracing` for evaluation, a raster-like
mode for fast training) is an env knob so the SAME pipeline produces both sides of
the R2P comparison. Each episode writes `episode_metadata.json`; the run writes
`dataset_manifest.json` (seed, scale, scene, schema). Regenerating with the same
knobs reproduces the dataset.

## Scale

`NUM_EPISODES x FRAMES_PER_EPISODE`. The `generate_dataset.sbatch` defaults target
the 200-600 GB range; raise episode/frame counts (and Slurm time/mem) to grow it -
do not shrink resolution or counts to "make it run".

## Provenance / licenses

The scene derives from public NASA geometry (data_plan.md Dataset 1); FITS/IFU
context from STScI/NASA (Datasets 3-5). Synthetic frames are generated, not
downloaded. See the raw `manifest.csv` for source URLs + sha256.
