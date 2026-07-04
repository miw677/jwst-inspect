#!/usr/bin/env python3
"""Reproducible synthetic dataset generation for JWST-Inspect (Group 2).

Runs inside the Isaac Sim container. Opens the Digital Twin scene, applies semantics
from the published label schema, and renders fixed-seed episodes with domain
randomization (camera pose on a standoff shell, sun intensity/angle, exposure, gold
roughness) and parameterized anomaly injection. Per frame it writes RGB, depth,
semantic + instance segmentation, camera params, and a metadata JSON matching
synthetic_schema_v0.json. Re-runnable: completed episodes are skipped (resume).

Everything dimensional is an env knob with an inline default; scale is set by
NUM_EPISODES x FRAMES_PER_EPISODE (the generate_dataset.sbatch drives full scale).

  SCENE=/data/shared/assets/jwst_inspect_scene_v1.usd OUT_DIR=/data/shared/datasets/v0 \
  NUM_EPISODES=4 FRAMES_PER_EPISODE=8 RENDERER=PathTracing /isaac-sim/python.sh replicator_generate.py
"""
import json
import math
import os
import random
import datetime as dt


def knob(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v is not None and v != "" else default


def log(m):
    print(f"[{dt.datetime.utcnow():%H:%M:%S}Z] {m}", flush=True)


SCENE = knob("SCENE", "/data/shared/assets/jwst_inspect_scene_v1.usd")
OUT_DIR = knob("OUT_DIR", "/data/shared/datasets/v0")
SEED = knob("DATASET_SEED", 20260627, int)
NUM_EPISODES = knob("NUM_EPISODES", 4, int)
FRAMES_PER_EP = knob("FRAMES_PER_EPISODE", 8, int)
W = knob("RENDER_W", 1024, int)
H = knob("RENDER_H", 1024, int)
SPP = knob("RENDER_SPP", 64, int)
RENDERER = knob("RENDERER", "PathTracing")          # PathTracing | RaytracedLighting (raster-like)
STANDOFF_MIN = knob("DR_STANDOFF_MIN_M", 10.0, float)
STANDOFF_MAX = knob("DR_STANDOFF_MAX_M", 22.0, float)
ELEV_MAX_DEG = knob("DR_ELEV_MAX_DEG", 60.0, float)
ANOMALY_PROB = knob("DR_ANOMALY_PROB", 0.3, float)
GOLD_ROUGH_MIN = knob("DR_GOLD_ROUGH_MIN", 0.02, float)
GOLD_ROUGH_MAX = knob("DR_GOLD_ROUGH_MAX", 0.18, float)
TASKS = knob("DR_TASKS", "approach_hold,sunshield_survey,mirror_inspection,anomaly_detection").split(",")
ANOMALY_CLASSES = ["missing_or_obscured_component", "unexpected_glare",
                   "sunshield_deformation_proxy", "mirror_region_visual_anomaly",
                   "sensor_confidence_failure"]

from isaacsim import SimulationApp
sim = SimulationApp({"headless": True})

import carb
import omni.usd
import omni.replicator.core as rep
from pxr import UsdGeom, Gf, Usd

settings = carb.settings.get_settings()
settings.set("/rtx/rendermode", RENDERER)
settings.set("/rtx/pathtracing/spp", SPP)
settings.set("/rtx/pathtracing/totalSpp", SPP)


def apply_semantics(stage):
    """Tag prims with Replicator semantics from their semanticClass attribute."""
    fn = None
    try:
        from isaacsim.core.utils.semantics import add_update_semantics
        fn = lambda prim, c: add_update_semantics(prim, c, type_label="class")
    except Exception:
        from pxr import Semantics
        def fn(prim, c):
            s = Semantics.SemanticsAPI.Apply(prim, "Semantics")
            s.CreateSemanticTypeAttr().Set("class")
            s.CreateSemanticDataAttr().Set(c)
    n = 0
    for prim in stage.Traverse():
        a = prim.GetAttribute("semanticClass")
        if a.IsValid() and a.Get():
            try:
                fn(prim, a.Get()); n += 1
            except Exception:
                pass
    return n


def look_at_matrix(pos, target, up=(0.0, 0.0, 1.0)):
    p = Gf.Vec3d(*pos); t = Gf.Vec3d(*target); u = Gf.Vec3d(*up)
    fwd = (t - p).GetNormalized()
    if abs(Gf.Dot(fwd, u)) > 0.999:
        u = Gf.Vec3d(0.0, 1.0, 0.0)
    right = Gf.Cross(fwd, u).GetNormalized()
    up2 = Gf.Cross(right, fwd)
    m = Gf.Matrix4d(1.0)
    m.SetColumn(0, Gf.Vec4d(right[0], right[1], right[2], 0.0))
    m.SetColumn(1, Gf.Vec4d(up2[0], up2[1], up2[2], 0.0))
    m.SetColumn(2, Gf.Vec4d(-fwd[0], -fwd[1], -fwd[2], 0.0))
    m.SetColumn(3, Gf.Vec4d(p[0], p[1], p[2], 1.0))
    return m


def set_camera(stage, cam_path, pos, target):
    cam = UsdGeom.Camera.Get(stage, cam_path)
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(look_at_matrix(pos, target))
    return cam


def sample_pose(rng):
    r = rng.uniform(STANDOFF_MIN, STANDOFF_MAX)
    az = math.radians(rng.uniform(0.0, 360.0))
    el = math.radians(rng.uniform(-ELEV_MAX_DEG, ELEV_MAX_DEG))
    return (r * math.cos(el) * math.cos(az), r * math.cos(el) * math.sin(az), r * math.sin(el)), r


def set_sun(stage, intensity, az_deg, el_deg):
    sun = stage.GetPrimAtPath("/World/Lighting/Sun")
    if sun and sun.IsValid():
        a = sun.GetAttribute("inputs:intensity") or sun.GetAttribute("intensity")
        if a and a.IsValid():
            a.Set(float(intensity))
        xf = UsdGeom.Xformable(sun)
        xf.ClearXformOpOrder()
        xf.AddRotateXYZOp().Set(Gf.Vec3f(float(el_deg), float(az_deg), 0.0))


def set_gold_roughness(stage, rough):
    a = None
    p = stage.GetPrimAtPath("/World/Looks/gold_mirror/Shader")
    if p and p.IsValid():
        a = p.GetAttribute("inputs:reflection_roughness_constant")
    if a and a.IsValid():
        a.Set(float(rough))


def set_visibility(stage, path, visible):
    p = stage.GetPrimAtPath(path)
    if p and p.IsValid():
        img = UsdGeom.Imageable(p)
        (img.MakeVisible if visible else img.MakeInvisible)()


def intrinsics(cam_path, stage):
    cam = UsdGeom.Camera.Get(stage, cam_path)
    focal = cam.GetFocalLengthAttr().Get() or 24.0
    h_ap = cam.GetHorizontalApertureAttr().Get() or 20.955
    fx = W * focal / h_ap
    fy = fx
    return dict(fx=fx, fy=fy, cx=W / 2.0, cy=H / 2.0, width=W, height=H)


def main():
    log(f"opening scene {SCENE} (renderer={RENDERER}, spp={SPP}, {W}x{H})")
    ctx = omni.usd.get_context()
    if not ctx.open_stage(SCENE):
        log(f"ERROR: could not open scene {SCENE}")
        return 1
    stage = ctx.get_stage()
    nsem = apply_semantics(stage)
    log(f"applied semantics to {nsem} prims")

    cam_path = "/World/InspectorCam"
    if not stage.GetPrimAtPath(cam_path).IsValid():
        UsdGeom.Camera.Define(stage, cam_path)
    rp = rep.create.render_product(cam_path, (W, H))
    writer = rep.WriterRegistry.get("BasicWriter")
    os.makedirs(OUT_DIR, exist_ok=True)

    # component paths eligible for "missing component" anomaly
    comp_paths = [p.GetPath().pathString for p in stage.Traverse()
                  if p.GetPath().pathString.count("/") == 3
                  and p.GetPath().pathString.startswith("/World/JWST/")]
    rng_master = random.Random(SEED)
    manifest = []

    for ep in range(NUM_EPISODES):
        ep_seed = SEED + ep
        rng = random.Random(ep_seed)
        ep_dir = os.path.join(OUT_DIR, f"episode_{ep:05d}")
        meta_path = os.path.join(ep_dir, "episode_metadata.json")
        if os.path.exists(meta_path):
            try:
                if json.load(open(meta_path)).get("num_frames") == FRAMES_PER_EP:
                    log(f"episode {ep}: complete, skipping (resume)")
                    continue
            except Exception:
                pass
        os.makedirs(ep_dir, exist_ok=True)
        task = TASKS[ep % len(TASKS)]
        has_anom = rng.random() < ANOMALY_PROB
        anom_class = rng.choice(ANOMALY_CLASSES) if has_anom else None
        anom_comp = None

        if has_anom and anom_class == "missing_or_obscured_component" and comp_paths:
            anom_comp = rng.choice(comp_paths)
            set_visibility(stage, anom_comp, False)

        writer.initialize(output_dir=ep_dir, rgb=True, distance_to_camera=True,
                          semantic_segmentation=True, instance_segmentation=True,
                          camera_params=True, bounding_box_2d_tight=True)
        writer.attach([rp])

        intr = intrinsics(cam_path, stage)
        frames_meta = []
        for fr in range(FRAMES_PER_EP):
            pos, standoff = sample_pose(rng)
            sun_i = rng.uniform(2.0, 12.0)
            if has_anom and anom_class == "unexpected_glare":
                sun_i *= rng.uniform(3.0, 6.0)
            az = rng.uniform(0, 360); el = rng.uniform(-30, 30)
            rough = rng.uniform(GOLD_ROUGH_MIN, GOLD_ROUGH_MAX)
            set_camera(stage, cam_path, pos, (0, 0, 0))
            set_sun(stage, sun_i, az, el)
            set_gold_roughness(stage, rough)
            rep.orchestrator.step(rt_subframes=max(1, SPP // 16))
            frames_meta.append(dict(
                frame=fr, camera_extrinsics=dict(position_xyz_m=list(pos), look_at_xyz_m=[0, 0, 0]),
                inspector_state=dict(position_xyz_m=list(pos), standoff_m=standoff,
                                     in_keep_out=standoff < knob("JWST_KEEPOUT_M", 8.0, float)),
                lighting_params=dict(sun_intensity=sun_i, sun_azimuth_deg=az, sun_elevation_deg=el),
                material_params=dict(gold_roughness=rough),
                anomaly=dict(has_anomaly=has_anom, anomaly_class=anom_class, anomaly_component=anom_comp),
                task_label=task))
        rep.orchestrator.wait_until_complete()
        writer.detach()

        if anom_comp:
            set_visibility(stage, anom_comp, True)

        ep_meta = dict(episode_id=ep, seed=ep_seed, scene_usd=SCENE, num_frames=FRAMES_PER_EP,
                       renderer=RENDERER, spp=SPP, resolution=f"{W}x{H}", task=task,
                       camera_intrinsics=intr, has_anomaly=has_anom, anomaly_class=anom_class,
                       anomaly_component=anom_comp, created_utc=dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                       image_tag=knob("ISAAC_SIM_TAG", "isaac-sim 5.1.0"), frames=frames_meta)
        with open(meta_path, "w") as f:
            json.dump(ep_meta, f, indent=2)
        manifest.append(dict(episode=ep, task=task, frames=FRAMES_PER_EP, anomaly=anom_class))
        log(f"episode {ep}: task={task} anomaly={anom_class} frames={FRAMES_PER_EP} -> {ep_dir}")

    with open(os.path.join(OUT_DIR, "dataset_manifest.json"), "w") as f:
        json.dump(dict(seed=SEED, episodes=NUM_EPISODES, frames_per_episode=FRAMES_PER_EP,
                       renderer=RENDERER, resolution=f"{W}x{H}", scene=SCENE,
                       schema="synthetic_schema_v0.json", entries=manifest), f, indent=2)
    log(f"dataset manifest written; {len(manifest)} episodes generated this run -> {OUT_DIR}")
    sim.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
