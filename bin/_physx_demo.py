# PhysX space-physics demo, streamed over WebRTC (launched by `jwst-gui physx`).
# Self-contained: loads the public NASA JWST model, builds a small procedural
# inspector microsat, star dome + sun, then flies the inspector on Clohessy-
# Wiltshire relative-orbit dynamics applied as real per-step PhysX forces with
# thruster burns and a debris field. The PhysX trajectory is compared against an
# independent RK4 integration and the drift is printed. Watch it live in the
# Isaac Sim WebRTC Streaming Client. All knobs from env.
import math
import os
import time

from isaacsim import SimulationApp

sim = SimulationApp({
    "width": 1280, "height": 720,
    "window_width": 1920, "window_height": 1080,
    "headless": True, "hide_ui": False,
    "renderer": "RaytracedLighting",
})

import carb
import numpy as np
import omni.timeline
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics

sim.set_setting("/app/window/drawMouse", True)
enable_extension("omni.services.livestream.nvcf")
print("LIVESTREAM enabled: connect the Isaac Sim WebRTC Streaming Client to jwst-ws", flush=True)

JWST_SRC = os.environ.get("DEMO_JWST", "/data/shared/raw/jwst_geometry/model_a/nasa-jwst-model-a.usdc")
STARMAP = os.environ.get("DEMO_STARMAP", "/data/shared/raw/env_lighting/starmaps/starmap_2020_8k.exr")
sim_seconds = float(os.environ.get("DEMO_SIM_S", "180"))
hold_seconds = float(os.environ.get("DEMO_HOLD_S", "600"))
fps_physics = float(os.environ.get("DEMO_PHYS_HZ", "60"))
time_warp = float(os.environ.get("DEMO_TIME_WARP", "1e5"))
mass = float(os.environ.get("FF_MASS_KG", "12.0"))
keepout_m = float(os.environ.get("JWST_KEEPOUT_M", "8.0"))
n_debris = int(os.environ.get("DEMO_DEBRIS_N", "6"))
seed = int(os.environ.get("SEED", "20260627"))
burn_spec = os.environ.get("DEMO_BURNS", "20:3:0.28,0.06,0.08;80:3:-0.22,0.16,-0.05")

n_real = 2.0 * math.pi / (365.25 * 86400.0)
n = n_real * time_warp
dt = 1.0 / fps_physics
print(f"PHYSX DEMO: mean motion {n_real:.3e} rad/s x warp {time_warp:.0e} -> {n:.4e} rad/s "
      f"(orbit period {2 * math.pi / n:.0f} s at warp), dt={dt:.4f} s", flush=True)

burns = []
for spec in burn_spec.split(";"):
    t0, dur, f = spec.split(":")
    fx, fy, fz = (float(v) for v in f.split(","))
    burns.append((float(t0), float(t0) + float(dur), np.array([fx, fy, fz])))
    print(f"thruster burn t=[{t0},{float(t0) + float(dur):.0f}]s force=({fx},{fy},{fz}) N", flush=True)

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

# JWST target, centered; stray reference children off.
jwst = UsdGeom.Xform.Define(stage, "/World/JWST")
jwst.GetPrim().GetReferences().AddReference(JWST_SRC)
src = Usd.Stage.Open(JWST_SRC)
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
center = Gf.Vec3d(cache.ComputeWorldBound(src.GetPseudoRoot()).ComputeAlignedRange().GetMidpoint())
jwst.AddTranslateOp().Set(-center)
for stray in ("Cube", "Camera", "env_light", "Light"):
    p = stage.GetPrimAtPath(f"/World/JWST/{stray}")
    if p.IsValid():
        p.SetActive(False)
print(f"JWST loaded from {JWST_SRC}", flush=True)

# Environment: star dome + sun.
dome = UsdLux.DomeLight.Define(stage, "/World/Stars")
dome.CreateIntensityAttr(float(os.environ.get("DEMO_DOME", "3000")))
if os.path.exists(STARMAP):
    dome.CreateTextureFileAttr(STARMAP)
    dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(float(os.environ.get("DEMO_SUN", "4000")))
sun.CreateAngleAttr(0.53)
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-40, 30, 0))

# Procedural inspector microsat: gold bus + two blue solar panels. The visual
# scale knob keeps it legible at demo camera distances without changing mass.
insp_scale = float(os.environ.get("DEMO_INSP_SCALE", "2.5"))
insp = UsdGeom.Xform.Define(stage, "/World/Inspector")
bus = UsdGeom.Cube.Define(stage, "/World/Inspector/bus")
bus.CreateSizeAttr(1.0)
UsdGeom.Xformable(bus.GetPrim()).AddScaleOp().Set(Gf.Vec3f(0.8, 0.5, 0.5))
bus.CreateDisplayColorAttr([Gf.Vec3f(0.75, 0.62, 0.22)])
for name, x in (("panel_p", 1.1), ("panel_n", -1.1)):
    pan = UsdGeom.Cube.Define(stage, f"/World/Inspector/{name}")
    pan.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(pan.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(x, 0, 0))
    xf.AddScaleOp().Set(Gf.Vec3f(0.9, 0.02, 0.45))
    pan.CreateDisplayColorAttr([Gf.Vec3f(0.06, 0.10, 0.32)])
p0 = np.array([float(v) for v in os.environ.get("DEMO_P0", "12,-12,6").split(",")])
insp_xf = UsdGeom.Xformable(insp.GetPrim())
insp_xf.AddTranslateOp().Set(Gf.Vec3d(*p0))
insp_xf.AddScaleOp().Set(Gf.Vec3f(insp_scale))

# Physics: zero-g scene, keep-out collider, rigid bodies.
phys = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
phys.CreateGravityMagnitudeAttr(0.0)
PhysxSchema.PhysxSceneAPI.Apply(phys.GetPrim()).CreateTimeStepsPerSecondAttr(fps_physics)
ko = UsdGeom.Sphere.Define(stage, "/World/JWST_Keepout")
ko.CreateRadiusAttr(keepout_m)
UsdPhysics.CollisionAPI.Apply(ko.GetPrim())
UsdGeom.Imageable(ko.GetPrim()).MakeInvisible()

rb = UsdPhysics.RigidBodyAPI.Apply(insp.GetPrim())
UsdPhysics.MassAPI.Apply(insp.GetPrim()).CreateMassAttr(mass)
for child in insp.GetPrim().GetChildren():
    if child.IsA(UsdGeom.Cube):
        UsdPhysics.CollisionAPI.Apply(child)
force_api = PhysxSchema.PhysxForceAPI.Apply(insp.GetPrim())
force_api.CreateModeAttr("force")
force_api.CreateWorldFrameEnabledAttr(True)
v0 = np.array([0.0, -2.0 * n * p0[0], 0.0])
rb.CreateVelocityAttr(Gf.Vec3f(*v0))
print(f"inspector rigid body: mass {mass} kg, p0 {p0.tolist()}, v0 {v0.round(3).tolist()} m/s (drift-free CW start)", flush=True)

rng = np.random.default_rng(seed)
for i in range(n_debris):
    d = UsdGeom.Sphere.Define(stage, f"/World/Debris_{i:02d}")
    d.CreateRadiusAttr(float(rng.uniform(0.12, 0.3)))
    d.CreateDisplayColorAttr([Gf.Vec3f(0.6, 0.6, 0.65)])
    pos = rng.uniform([-30, -30, -8], [30, 30, 12])
    while np.linalg.norm(pos) < keepout_m + 6:
        pos = rng.uniform([-30, -30, -8], [30, 30, 12])
    UsdGeom.Xformable(d.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdPhysics.RigidBodyAPI.Apply(d.GetPrim()).CreateVelocityAttr(
        Gf.Vec3f(0.0, float(-2.0 * n * pos[0]), float(n * pos[2] * rng.uniform(-0.5, 0.5))))
    UsdPhysics.CollisionAPI.Apply(d.GetPrim())
    UsdPhysics.MassAPI.Apply(d.GetPrim()).CreateMassAttr(2.0)
print(f"{n_debris} debris rigid bodies seeded (seed {seed})", flush=True)

# Chase camera: tracks the JWST-inspector midpoint and zooms with separation
# so both craft stay framed for the whole demo.
cam = UsdGeom.Camera.Define(stage, "/World/DemoCam")
cam.CreateFocalLengthAttr(24.0)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 20000.0))
cam_dir = Gf.Vec3d(25, -33, 14).GetNormalized()
cam_op = UsdGeom.Xformable(cam.GetPrim()).AddTransformOp()


def track_camera(insp_pos):
    # 24 mm lens, ~47 deg horizontal FOV: distance grows with separation so the
    # telescope (~13 m half-extent) and the microsat both stay in frame.
    look = Gf.Vec3d(*(0.5 * insp_pos))
    dist = max(38.0, 0.65 * float(np.linalg.norm(insp_pos)) + 34.0)
    cam_op.Set(Gf.Matrix4d().SetLookAt(look + cam_dir * dist, look, Gf.Vec3d(0, 0, 1)).GetInverse())


track_camera(p0)
for _ in range(30):
    sim.update()
viewport = get_active_viewport()
viewport.camera_path = "/World/DemoCam"


def hold_democam():
    # A late-connecting WebRTC client can reset the viewport to Perspective;
    # keep the chase camera in charge.
    if str(viewport.camera_path) != "/World/DemoCam":
        viewport.camera_path = "/World/DemoCam"


def cw_accel(p, v):
    return np.array([3 * n * n * p[0] + 2 * n * v[1], -2 * n * v[0], -n * n * p[2]])


def burn_force(t):
    f = np.zeros(3)
    for t0, t1, force in burns:
        if t0 <= t < t1:
            f += force
    return f


def deriv(t, state):
    p, v = state[:3], state[3:]
    return np.concatenate([v, cw_accel(p, v) + burn_force(t) / mass])


timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(3):
    sim.update()

xf_cache = UsdGeom.XformCache()
steps = int(sim_seconds * fps_physics)
ref = np.concatenate([p0.astype(float), v0])
prev_p = p0.astype(float)
insp_prim = insp.GetPrim()
for k in range(steps):
    t = k * dt
    p = np.array(xf_cache.GetLocalToWorldTransform(insp_prim).ExtractTranslation())
    xf_cache.Clear()
    v_est = (p - prev_p) / dt if k else v0.copy()
    prev_p = p
    track_camera(p)
    hold_democam()
    force_api.GetForceAttr().Set(Gf.Vec3f(*(cw_accel(p, v_est) * mass + burn_force(t))))
    k1 = deriv(t, ref); k2 = deriv(t + dt / 2, ref + dt / 2 * k1)
    k3 = deriv(t + dt / 2, ref + dt / 2 * k2); k4 = deriv(t + dt, ref + dt * k3)
    ref = ref + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    if k % 600 == 0:
        drift = float(np.linalg.norm(p - ref[:3]))
        print(f"t={t:6.1f}s pos={p.round(2).tolist()} |v|={np.linalg.norm(v_est):.3f} m/s "
              f"burn={np.linalg.norm(burn_force(t)):.2f} N  PhysX-vs-analytic drift={drift:.3f} m", flush=True)
    sim.update()
    if not sim.is_exiting() and not sim._app.is_running():
        break

drift = float(np.linalg.norm(prev_p - ref[:3]))
print(f"PHYSX DEMO COMPLETE after {sim_seconds:.0f} s: final PhysX-vs-analytic drift {drift:.3f} m. "
      f"Orbit continues under CW + station-keeping for {hold_seconds:.0f} s - keep watching or close the app.", flush=True)
# Station-keeping: the burns left a secular along-track drift (vy + 2n x != 0).
# A P-controller thruster force nulls it so the relative orbit stays bounded.
k_sk = float(os.environ.get("DEMO_SK_GAIN", "0.4"))
t_end = time.time() + hold_seconds
k = 0
while time.time() < t_end and sim._app.is_running() and not sim.is_exiting():
    p = np.array(xf_cache.GetLocalToWorldTransform(insp_prim).ExtractTranslation())
    xf_cache.Clear()
    v_est = (p - prev_p) / dt
    prev_p = p
    track_camera(p)
    hold_democam()
    f_sk = np.array([0.0, -k_sk * mass * (v_est[1] + 2 * n * p[0]), -k_sk * mass * v_est[2] * 0.2])
    force_api.GetForceAttr().Set(Gf.Vec3f(*(cw_accel(p, v_est) * mass + f_sk)))
    if k % 1800 == 0:
        print(f"holding orbit: pos={p.round(2).tolist()} |v|={np.linalg.norm(v_est):.3f} m/s "
              f"station-keeping force={np.linalg.norm(f_sk):.3f} N", flush=True)
    k += 1
    sim.update()
sim.close()
