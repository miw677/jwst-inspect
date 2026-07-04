#!/usr/bin/env python3
"""Author the free-flying inspector microsat USD asset (Group 1).

A NASA-inspired non-contact inspector (Mini AERCam / Seeker / LICIACube heritage,
data_plan.md Dataset 2): MLI-wrapped bus, body-mounted solar arrays, cold-gas
microthruster locations, reaction-wheel block, star tracker, stereo RGB cameras,
depth/LiDAR, IR camera, and antenna. Sensor/thruster sites are labeled empty Xforms
(real frames the Autonomous team mounts dynamics + sensors onto). Dimensions are
env knobs. Run in the jwst-usd env:
    OUT_DIR=/data/shared/assets python build_inspector.py
"""
import datetime as dt
import os
import sys

from pxr import Usd, UsdGeom, Gf, Sdf


def knob(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v is not None and v != "" else default


def log(msg):
    print(f"[{dt.datetime.utcnow():%H:%M:%S}Z] {msg}", flush=True)


def box(stage, path, sx, sy, sz, cls, tx=0.0, ty=0.0, tz=0.0):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(c)
    xf.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
    xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
    c.GetPrim().CreateAttribute("semanticClass", Sdf.ValueTypeNames.Token).Set(cls)
    return c


def site(stage, path, tx, ty, tz, kind):
    x = UsdGeom.Xform.Define(stage, path)
    UsdGeom.Xformable(x).AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
    p = x.GetPrim()
    p.CreateAttribute("semanticClass", Sdf.ValueTypeNames.Token).Set("inspector")
    p.CreateAttribute("siteKind", Sdf.ValueTypeNames.Token).Set(kind)
    return x


def main():
    out_dir = knob("OUT_DIR", "/data/shared/assets")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "inspector_microsat.usd")
    if os.path.exists(path):
        os.remove(path)
    # Bus dimensions (m) - small free-flyer; env-overridable.
    bx = knob("INSP_BUS_X", 0.5, float)
    by = knob("INSP_BUS_Y", 0.5, float)
    bz = knob("INSP_BUS_Z", 0.6, float)

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Inspector")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().CreateAttribute("semanticClass", Sdf.ValueTypeNames.Token).Set("inspector")

    box(stage, "/Inspector/bus", bx, by, bz, "inspector")
    box(stage, "/Inspector/solar_array_pX", bx * 1.4, 0.02, bz * 0.8, "inspector", tx=bx * 0.9)
    box(stage, "/Inspector/solar_array_nX", bx * 1.4, 0.02, bz * 0.8, "inspector", tx=-bx * 0.9)
    box(stage, "/Inspector/reaction_wheels", bx * 0.4, by * 0.4, bz * 0.3, "inspector", tz=-bz * 0.2)

    # Cold-gas microthruster sites: +/- on each body axis (12 sites for 6-DoF).
    half = [(bx / 2, "x"), (by / 2, "y"), (bz / 2, "z")]
    for d, ax in half:
        for s in (1, -1):
            t = {"x": (s * d, 0, 0), "y": (0, s * d, 0), "z": (0, 0, s * d)}[ax]
            site(stage, f"/Inspector/thruster_{ax}{'p' if s > 0 else 'n'}", *t, kind="thruster")

    # Sensors: stereo RGB (baseline), depth/LiDAR, IR, star tracker, IMU, antenna.
    base = knob("INSP_STEREO_BASELINE_M", 0.15, float)
    site(stage, "/Inspector/cam_left", base / 2, by / 2, 0, kind="rgb_camera")
    site(stage, "/Inspector/cam_right", -base / 2, by / 2, 0, kind="rgb_camera")
    site(stage, "/Inspector/depth_lidar", 0, by / 2, 0.05, kind="depth_lidar")
    site(stage, "/Inspector/ir_camera", 0, by / 2, -0.05, kind="ir_camera")
    site(stage, "/Inspector/star_tracker", 0, 0, bz / 2, kind="star_tracker")
    site(stage, "/Inspector/imu", 0, 0, 0, kind="imu")
    site(stage, "/Inspector/antenna", 0, -by / 2, 0, kind="antenna")

    stage.GetRootLayer().Save()
    log(f"inspector asset published: {path} ({os.path.getsize(path)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
