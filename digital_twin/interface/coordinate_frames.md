# JWST-Inspect coordinate frames (Digital Twin interface)

Stable frame + unit conventions for the published scene. Consumed by Benchmark
(camera poses) and Autonomous (dynamics + observation frames). Dimensions are env
knobs in `src/jwst_scene.py`; this document fixes the conventions, not the values.

- Units: meters (`metersPerUnit = 1.0`). Up axis: `Z` (`UsdGeom` stage up `Z`).
- World frame `/World`: scene root, default prim. Origin at the JWST geometric center.
- JWST body frame `/World/JWST`: identity to world at author time; the target is
  centered at the origin so safety regions are defined about it.
- Component frames `/World/JWST/<component>` and `/World/JWST/primary_mirror/inst_NN`
  (18 mirror-segment frames): each carries a `semanticClass` token from
  `semantic_labels_v0.json`.
- Inspector frame `/World/Inspector` (referenced `inspector_microsat.usd`): the
  free-flyer body frame; the Autonomous env attaches 6-DoF rigid-body dynamics here.
  Sensor/thruster sites are child Xforms tagged with a `siteKind` token.
- Safety regions `/World/SafetyRegions` (all `purpose=guide`, not rendered solid):
  - `keep_out_zone`: sphere, radius `JWST_KEEPOUT_M` (default 8 m) about the origin.
  - `standoff_shell`: sphere marker at radius `JWST_STANDOFF_M` (default 15 m).
  - `approach_corridor`: cone, half-angle `JWST_CORRIDOR_HALF_DEG` (default 20 deg),
    length `JWST_CORRIDOR_LEN_M` (default 30 m), axis `+Z` from the standoff shell.
- Camera `/World/InspectorCam`: focal `JWST_CAM_FOCAL_MM` (default 24 mm), placed at
  standoff distance looking toward the target.
- Lighting `/World/Lighting`: `Sun` (DistantLight) + `Fill` (DomeLight) with a
  `lighting` variant set: `sun_high` / `sun_low` / `eclipse`.
