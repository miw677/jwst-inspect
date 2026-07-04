# jwst-gui pilot: manually flyable inspector microsat around the JWST model.
# Runs inside the Isaac Sim container (streaming app, viewed over WebRTC).
# Zero-g PhysX dynamics with Clohessy-Wiltshire relative-orbit terms applied as
# real per-step forces (same validated pattern as the PhysX orbital demo), a
# first-person navigation camera, keyboard RCS thrusters, a sensor suite
# (rangefinder, photometer, TSIS-1-grounded solar exposure, radiation proxy,
# IMU), photo capture with state sidecars, Replicator sensor sweeps, an
# optional MMOD meteor-shower damage event, and an inspection report.
# Every tunable is an env knob with the default inline at the call site.
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from isaacsim import SimulationApp


def knob(name, default, cast=float):
    v = os.environ.get(name)
    return cast(v) if v not in (None, "") else default


W = knob("PILOT_W", 1920, int)
H = knob("PILOT_H", 1080, int)
EXPERIENCE = knob("PILOT_EXPERIENCE", "/isaac-sim/apps/isaacsim.exp.full.streaming.kit", str)
# hide_ui False keeps the viewport + HUD visible in the WebRTC stream
sim = SimulationApp({"headless": True, "hide_ui": False, "width": W, "height": H,
                     "window_width": W, "window_height": H}, experience=EXPERIENCE)

import carb
import carb.input
import numpy as np
import omni.appwindow
import omni.replicator.core as rep
import omni.timeline
import omni.ui as ui
import omni.usd
from isaacsim.core.utils.semantics import add_update_semantics
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from omni.physx import get_physx_scene_query_interface
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

import _pilot_event_mmod as mmod

# --- knobs -------------------------------------------------------------------
SEED = knob("SEED", 20260627, int)
OUT = knob("PILOT_OUT", f"/data/scratch/{os.environ.get('USER', 'unknown')}/pilot_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}", str)
JWST_SRC = knob("PILOT_JWST", "/data/shared/raw/jwst_geometry/model_a/nasa-jwst-model-a.usdc", str)
STARMAP = knob("PILOT_STARMAP", "/data/shared/raw/env_lighting/starmaps/starmap_2020_16k.exr", str)
EPH_CSV = knob("PILOT_EPH", "/data/shared/raw/jwst_ephemeris/jwst_horizons_vectors_sun_2022-01-01_2026-12-31_6h.csv", str)
TSIS_CSV = knob("PILOT_TSIS", "/data/shared/raw/env_lighting/solar/tsis1_hsrs.csv", str)
CRAFT_USD = knob("PILOT_CRAFT_USD", "", str)          # optional team-built craft asset
EPOCH = knob("PILOT_EPOCH", f"{datetime.now(timezone.utc):%Y-%m-%d}", str)
SUN_INTENSITY = knob("PILOT_SUN_INTENSITY", 3500.0)
SUN_ELEV_DEG = knob("PILOT_SUN_ELEV_DEG", 35.0)
SUN_AZ_OFFSET_DEG = knob("PILOT_SUN_AZ_OFFSET_DEG", 0.0)
DOME_INTENSITY = knob("PILOT_DOME_INTENSITY", 1500.0)
JWST_YAW_DEG = knob("PILOT_JWST_YAW_DEG", 0.0)

PHYS_HZ = knob("PILOT_PHYS_HZ", 60.0)
TIME_WARP = knob("PILOT_TIME_WARP", 1e5)
MASS = knob("FF_MASS_KG", 12.0)
THRUST_N = knob("PILOT_THRUST_N", 6.0)                # per-axis RCS authority
TORQUE_NM = knob("PILOT_TORQUE_NM", 0.6)
DV_BUDGET = knob("PILOT_DV_BUDGET_MPS", 20.0)
KEEPOUT_M = knob("JWST_KEEPOUT_M", 8.0)
# default start is probe-verified: boresight to origin locks on the telescope
# (raycast hit at 27.1 m, mirror-side view); override with PILOT_START=x,y,z
START = tuple(float(v) for v in knob("PILOT_START", "25.7,-3.0,23.5", str).split(","))
DAMP_KV = knob("PILOT_DAMP_KV", 0.8)                  # 1/s translational damp gain
DAMP_KW = knob("PILOT_DAMP_KW", 2.0)                  # 1/s rotational damp gain
AUTO_RATE_DAMP = knob("PILOT_AUTO_RATE_DAMP", 1, int)

SENSOR_HZ = knob("PILOT_SENSOR_HZ", 10.0)
HUD_HZ = knob("PILOT_HUD_HZ", 5.0)
HEARTBEAT_S = knob("PILOT_HEARTBEAT_S", 5.0)
FLUSH_S = knob("PILOT_LOG_FLUSH_S", 10.0)
RAD_BG_CPS = knob("PILOT_RAD_BG_CPS", 12.0)
RAD_SOLAR_FACTOR = knob("PILOT_RAD_SOLAR_FACTOR", 0.6)
SWEEP_W = knob("PILOT_SWEEP_W", 1920, int)
SWEEP_H = knob("PILOT_SWEEP_H", 1080, int)
SWEEP_SUBFRAMES = knob("PILOT_SWEEP_SUBFRAMES", 16, int)
PHOTOMETER_RES = knob("PILOT_PHOTOMETER_RES", 256, int)
RANGEFINDER_MAX_M = knob("PILOT_RANGEFINDER_MAX_M", 2000.0)
SUN_OCCLUDE_MAX_M = knob("PILOT_SUN_OCCLUDE_MAX_M", 400.0)
FPV_FOCAL_MM = knob("PILOT_FPV_FOCAL_MM", 18.0)
MAX_SESSION_S = knob("PILOT_MAX_SESSION_S", 14400.0)
EXPECT_GPUS = knob("PILOT_EXPECT_GPUS", 1, int)
EVENT = knob("PILOT_EVENT", "", str)                  # "shower" arms the MMOD event
SELFTEST = knob("PILOT_SELFTEST", 0, int)

for p in (JWST_SRC, STARMAP, EPH_CSV, TSIS_CSV):
    if not os.path.exists(p):
        raise SystemExit(f"ERROR: required input missing: {p}")
if CRAFT_USD and not os.path.exists(CRAFT_USD):
    raise SystemExit(f"ERROR: PILOT_CRAFT_USD set but missing: {CRAFT_USD}")
os.makedirs(os.path.join(OUT, "photos"), exist_ok=True)
os.makedirs(os.path.join(OUT, "sweeps"), exist_ok=True)

# GPU assert (hardware discipline: fail fast on mismatch, report detected vs expected)
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True)
except FileNotFoundError:
    raise SystemExit("ERROR: nvidia-smi not available inside the container; cannot assert GPU count")
gpus = [l for l in smi.stdout.strip().splitlines() if l.strip()]
if len(gpus) != EXPECT_GPUS:
    raise SystemExit(f"ERROR: expected {EXPECT_GPUS} GPU(s), detected {len(gpus)}: {gpus}")
print(f"PILOT gpu ok: {gpus[0]}", flush=True)

rng = np.random.default_rng(SEED)
rep.set_global_seed(SEED)

# --- solar constant from the TSIS-1 measured spectrum ------------------------
t0 = time.time()
tsis = np.loadtxt(TSIS_CSV, delimiter=",", skiprows=1, usecols=(0, 1))
TSI_1AU = float(np.trapz(tsis[:, 1], tsis[:, 0]))
print(f"PILOT TSIS-1 {os.path.basename(TSIS_CSV)}: {tsis.shape[0]} rows, band {tsis[0, 0]:.0f}-{tsis[-1, 0]:.0f} nm, "
      f"integrated TSI {TSI_1AU:.1f} W/m2 at 1 au ({time.time() - t0:.1f}s)", flush=True)

# --- ephemeris: sun azimuth + range for the epoch ----------------------------
epoch_horizons = datetime.strptime(EPOCH, "%Y-%m-%d").strftime("%Y-%b-%d")
best = None
with open(EPH_CSV) as fh:
    for row in csv.DictReader(fh):
        if epoch_horizons in row["datetime_str"]:
            best = row
            break
if best is None:
    raise SystemExit(f"ERROR: epoch {EPOCH} ({epoch_horizons}) not found in {EPH_CSV}")
r_vec = Gf.Vec3d(float(best["x"]), float(best["y"]), float(best["z"]))
R_AU = float(best["range"])
d_photon = r_vec.GetNormalized()  # photon travel direction sun -> JWST
az = math.degrees(math.atan2(d_photon[1], d_photon[0])) + SUN_AZ_OFFSET_DEG
SOLAR_WM2 = TSI_1AU / (R_AU * R_AU)
print(f"PILOT epoch {best['datetime_str']} range {R_AU:.6f} au; sun azimuth {az:.2f} deg "
      f"(elevation {SUN_ELEV_DEG:.1f} deg is a visualization knob); solar flux at range {SOLAR_WM2:.1f} W/m2", flush=True)

elev = math.radians(SUN_ELEV_DEG)
azr = math.radians(az)
SUN_DIR = np.array([math.cos(elev) * math.cos(azr), math.cos(elev) * math.sin(azr), -math.sin(elev)])  # photon dir
TO_SUN = -SUN_DIR

# --- stage composition --------------------------------------------------------
ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())

phys = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
phys.CreateGravityMagnitudeAttr(0.0)
PhysxSchema.PhysxSceneAPI.Apply(phys.GetPrim()).CreateTimeStepsPerSecondAttr(PHYS_HZ)
print(f"PILOT physics: gravity 0, {PHYS_HZ:.0f} steps/s", flush=True)

jwst = UsdGeom.Xform.Define(stage, "/World/JWST")
jwst.GetPrim().GetReferences().AddReference(JWST_SRC)
src = Usd.Stage.Open(JWST_SRC)
bcache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
jbox = bcache.ComputeWorldBound(src.GetPseudoRoot()).ComputeAlignedRange()
center = Gf.Vec3d(jbox.GetMidpoint())
jwst.AddRotateZOp().Set(JWST_YAW_DEG)
jwst.AddTranslateOp().Set(-center)
for stray in ("Cube", "Camera", "env_light", "Light"):
    pr = stage.GetPrimAtPath(f"/World/JWST/{stray}")
    if pr.IsValid():
        pr.SetActive(False)
JBOX_SIZE = np.array(jbox.GetSize())
print(f"PILOT JWST {os.path.basename(JWST_SRC)} recentered, yaw {JWST_YAW_DEG} deg, bbox {JBOX_SIZE.round(2).tolist()} m", flush=True)

t0 = time.time()
n_mesh = 0
for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/JWST")):
    if prim.IsA(UsdGeom.Mesh):
        UsdPhysics.CollisionAPI.Apply(prim)
        n_mesh += 1
print(f"PILOT colliders on {n_mesh} telescope meshes ({time.time() - t0:.2f}s)", flush=True)

ko = UsdGeom.Sphere.Define(stage, "/World/JWST_Keepout")
ko.CreateRadiusAttr(KEEPOUT_M)
UsdPhysics.CollisionAPI.Apply(ko.GetPrim())
UsdGeom.Imageable(ko.GetPrim()).MakeInvisible()
print(f"PILOT keep-out collider radius {KEEPOUT_M} m (hard shell; telescope meshes are also solid)", flush=True)

env = UsdGeom.Xform.Define(stage, "/World/Environment")
dome = UsdLux.DomeLight.Define(stage, "/World/Environment/Stars")
dome.CreateIntensityAttr(DOME_INTENSITY)
dome.CreateTextureFileAttr(STARMAP)
dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
sun = UsdLux.DistantLight.Define(stage, "/World/Environment/Sun")
sun.CreateIntensityAttr(SUN_INTENSITY)
sun.CreateAngleAttr(knob("PILOT_SUN_ANGLE_DEG", 0.53))
sun.CreateColorAttr(Gf.Vec3f(1.0, 0.985, 0.955))
zaxis = Gf.Vec3d(*(-SUN_DIR))
xaxis = Gf.Cross(Gf.Vec3d(0, 0, 1), zaxis).GetNormalized()
yaxis = Gf.Cross(zaxis, xaxis)
UsdGeom.Xformable(sun.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(
    xaxis[0], xaxis[1], xaxis[2], 0, yaxis[0], yaxis[1], yaxis[2], 0,
    zaxis[0], zaxis[1], zaxis[2], 0, 0, 0, 0, 1))
print(f"PILOT env: star dome {os.path.basename(STARMAP)} intensity {DOME_INTENSITY}, sun intensity {SUN_INTENSITY}", flush=True)

# --- craft -------------------------------------------------------------------
BX = knob("INSP_BUS_X", 0.5)
BY = knob("INSP_BUS_Y", 0.5)
BZ = knob("INSP_BUS_Z", 0.6)
craft = UsdGeom.Xform.Define(stage, "/World/Inspector")
craft_prim = craft.GetPrim()
p0 = np.array(START)
# initial attitude: body +Y (the FPV boresight) looks at the target center
fwd0 = -p0 / (np.linalg.norm(p0) + 1e-9)
if abs(fwd0[2]) > 0.98:
    raise SystemExit(f"ERROR: PILOT_START {START} is (anti)parallel to +Z; pick an off-axis start")
right0 = np.cross(fwd0, np.array([0.0, 0.0, 1.0]))
right0 /= np.linalg.norm(right0)
up0 = np.cross(right0, fwd0)
m0 = Gf.Matrix3d(right0[0], right0[1], right0[2],
                 fwd0[0], fwd0[1], fwd0[2],
                 up0[0], up0[1], up0[2])
q0 = m0.ExtractRotation().GetQuat()
craft.AddTranslateOp().Set(Gf.Vec3d(*p0))
craft.AddOrientOp().Set(Gf.Quatf(q0))

if CRAFT_USD:
    body_root = UsdGeom.Xform.Define(stage, "/World/Inspector/asset")
    body_root.GetPrim().GetReferences().AddReference(CRAFT_USD)
    n_cm = 0
    for prim in Usd.PrimRange(body_root.GetPrim()):
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube):
            UsdPhysics.CollisionAPI.Apply(prim)
            n_cm += 1
    print(f"PILOT craft asset referenced from {CRAFT_USD} ({n_cm} colliders)", flush=True)
else:
    def part(path, sx, sy, sz, tx, ty, tz, rgb):
        c = UsdGeom.Cube.Define(stage, path)
        c.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(c)
        xf.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
        xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
        c.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        return c
    part("/World/Inspector/bus", BX, BY, BZ, 0, 0, 0, (0.75, 0.62, 0.22))
    part("/World/Inspector/solar_array_pX", BX * 1.4, 0.02, BZ * 0.8, BX * 0.9, 0, 0, (0.06, 0.10, 0.32))
    part("/World/Inspector/solar_array_nX", BX * 1.4, 0.02, BZ * 0.8, -BX * 0.9, 0, 0, (0.06, 0.10, 0.32))
    part("/World/Inspector/reaction_wheels", BX * 0.4, BY * 0.4, BZ * 0.3, 0, 0, -BZ * 0.2, (0.35, 0.35, 0.38))

rb = UsdPhysics.RigidBodyAPI.Apply(craft_prim)
mass_api = UsdPhysics.MassAPI.Apply(craft_prim)
mass_api.CreateMassAttr(MASS)
# explicit box inertia from bus dims so torque authority is deterministic
IX = MASS / 12.0 * (BY * BY + BZ * BZ)
IY = MASS / 12.0 * (BX * BX + BZ * BZ)
IZ = MASS / 12.0 * (BX * BX + BY * BY)
mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(IX, IY, IZ))
for child in craft_prim.GetChildren():
    if child.IsA(UsdGeom.Cube):
        UsdPhysics.CollisionAPI.Apply(child)
force_api = PhysxSchema.PhysxForceAPI.Apply(craft_prim)
force_api.CreateModeAttr("force")
force_api.CreateWorldFrameEnabledAttr(True)
force_api.CreateForceAttr(Gf.Vec3f(0, 0, 0))
force_api.CreateTorqueAttr(Gf.Vec3f(0, 0, 0))
print(f"PILOT craft: mass {MASS} kg, inertia diag ({IX:.3f},{IY:.3f},{IZ:.3f}) kg m2, "
      f"start {p0.round(2).tolist()} boresight {fwd0.round(3).tolist()} (locked on target center)", flush=True)

# thruster plume indicators: one cone per face, visible while that face's thruster fires
PLUME_LEN = knob("PILOT_PLUME_LEN_M", 0.35)
plumes = {}
for ax_i, (ax, half) in enumerate((("x", BX / 2), ("y", BY / 2), ("z", BZ / 2))):
    for s in (1, -1):
        name = f"plume_{ax}{'p' if s > 0 else 'n'}"
        cone = UsdGeom.Cone.Define(stage, f"/World/Inspector/{name}")
        cone.CreateHeightAttr(PLUME_LEN)
        cone.CreateRadiusAttr(PLUME_LEN * 0.25)
        cone.CreateAxisAttr({"x": "X", "y": "Y", "z": "Z"}[ax])
        cone.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.75, 0.2)])
        t = [0.0, 0.0, 0.0]
        t[ax_i] = s * (half + PLUME_LEN / 2)
        xf = UsdGeom.Xformable(cone)
        xf.AddTranslateOp().Set(Gf.Vec3d(*t))
        if s < 0:
            xf.AddRotateXOp().Set(180.0) if ax == "z" else xf.AddRotateZOp().Set(180.0)
        UsdGeom.Imageable(cone.GetPrim()).MakeInvisible()
        plumes[(ax_i, s)] = cone.GetPrim()

fpv = UsdGeom.Camera.Define(stage, "/World/Inspector/fpv_cam")
fpv.CreateFocalLengthAttr(FPV_FOCAL_MM)
fpv.CreateClippingRangeAttr(Gf.Vec2f(0.05, 20000.0))
xf = UsdGeom.Xformable(fpv)
xf.AddTranslateOp().Set(Gf.Vec3d(0, BY / 2 + 0.02, 0.05))
xf.AddRotateXOp().Set(90.0)  # camera -Z looks along body +Y (forward)
chase = UsdGeom.Camera.Define(stage, "/World/Inspector/chase_cam")
chase.CreateFocalLengthAttr(knob("PILOT_CHASE_FOCAL_MM", 24.0))
chase.CreateClippingRangeAttr(Gf.Vec2f(0.05, 20000.0))
chase_eye = Gf.Vec3d(*(float(v) for v in knob("PILOT_CHASE_EYE", "0,-6,2", str).split(",")))
chase_view = Gf.Matrix4d().SetLookAt(chase_eye, Gf.Vec3d(0, 4, 0), Gf.Vec3d(0, 0, 1))
UsdGeom.Xformable(chase).AddTransformOp().Set(chase_view.GetInverse())

add_update_semantics(stage.GetPrimAtPath("/World/JWST"), "jwst")
add_update_semantics(craft_prim, "inspector")

# --- sensors: render products ------------------------------------------------
FPV_PATH = "/World/Inspector/fpv_cam"
photometer_rp = rep.create.render_product(FPV_PATH, (PHOTOMETER_RES, PHOTOMETER_RES))
photometer = rep.AnnotatorRegistry.get_annotator("rgb")
photometer.attach([photometer_rp])
sweep_rp = rep.create.render_product(FPV_PATH, (SWEEP_W, SWEEP_H))
print(f"PILOT sensors: photometer {PHOTOMETER_RES}px rgb annotator, sweep product {SWEEP_W}x{SWEEP_H}", flush=True)

# --- CW dynamics -------------------------------------------------------------
N_REAL = 2.0 * math.pi / (365.25 * 86400.0)
N_CW = N_REAL * TIME_WARP
DT = 1.0 / PHYS_HZ
print(f"PILOT orbital mechanics: n_real={N_REAL:.3e} rad/s x warp {TIME_WARP:.0e} -> n={N_CW:.4e} rad/s "
      f"(period {2 * math.pi / N_CW:.0f} s at warp), dt={DT:.4f} s", flush=True)


def cw_accel(p, v):
    # Hill frame: x radial, y along-track, z cross-track
    return np.array([3.0 * N_CW * N_CW * p[0] + 2.0 * N_CW * v[1],
                     -2.0 * N_CW * v[0],
                     -N_CW * N_CW * p[2]])


v0 = np.array([0.0, -2.0 * N_CW * p0[0], 0.0])
rb.CreateVelocityAttr(Gf.Vec3f(*v0))
rb.CreateAngularVelocityAttr(Gf.Vec3f(0, 0, 0))

# --- radiation proxy ---------------------------------------------------------
print(f"PILOT radiation proxy model: counts = Poisson(bg {RAD_BG_CPS} cps x (1 + {RAD_SOLAR_FACTOR} x sun_visible) "
      f"+ shower term) per sample; a documented proxy, not a physics claim", flush=True)

# --- input -------------------------------------------------------------------
KI = carb.input.KeyboardInput
held = set()
toggles = {"auto_damp": bool(AUTO_RATE_DAMP), "fpv": True, "quit": False,
            "photo_req": 0, "sweep_req": 0, "report_req": 0, "shower_req": EVENT == "shower"}
thrust_scale = [1.0]
SCALE_MIN = knob("PILOT_SCALE_MIN", 0.125)
SCALE_MAX = knob("PILOT_SCALE_MAX", 8.0)


def on_key(e):
    if e.type == carb.input.KeyboardEventType.KEY_PRESS:
        held.add(e.input)
        if e.input == KI.T:
            toggles["auto_damp"] = not toggles["auto_damp"]
            print(f"PILOT auto rate-damp {'ON' if toggles['auto_damp'] else 'OFF'}", flush=True)
        elif e.input == KI.V:
            toggles["fpv"] = not toggles["fpv"]
        elif e.input == KI.P:
            toggles["photo_req"] += 1
        elif e.input == KI.G:
            toggles["sweep_req"] += 1
        elif e.input == KI.M:
            toggles["shower_req"] = True
        elif e.input in (KI.ENTER, KI.NUMPAD_ENTER):
            toggles["report_req"] += 1
        elif e.input == KI.ESCAPE:
            toggles["quit"] = True
        elif e.input == KI.LEFT_BRACKET:
            thrust_scale[0] = max(SCALE_MIN, thrust_scale[0] * 0.5)
            print(f"PILOT thrust scale {thrust_scale[0]:.3f}", flush=True)
        elif e.input == KI.RIGHT_BRACKET:
            thrust_scale[0] = min(SCALE_MAX, thrust_scale[0] * 2.0)
            print(f"PILOT thrust scale {thrust_scale[0]:.3f}", flush=True)
    elif e.type == carb.input.KeyboardEventType.KEY_RELEASE:
        held.discard(e.input)
    return True


inp = carb.input.acquire_input_interface()
kb_sub = inp.subscribe_to_keyboard_events(omni.appwindow.get_default_app_window().get_keyboard(), on_key)
print("PILOT keys: W/S fwd/back, A/D left/right, R/F up/down, arrows pitch/yaw, Q/E roll, "
      "SPACE damp, T auto-damp, [ ] thrust scale, V view, P photo, G sweep, M shower, ENTER report, ESC quit", flush=True)

# --- HUD ---------------------------------------------------------------------
hud_win = ui.Window("Pilot HUD", width=knob("PILOT_HUD_W", 560, int), height=knob("PILOT_HUD_H", 300, int))
with hud_win.frame:
    hud_label = ui.Label("booting", style={"font_size": knob("PILOT_HUD_FONT", 16, int), "color": 0xFF00FF80})

viewport = get_active_viewport()
viewport.camera_path = FPV_PATH

# --- shower event ------------------------------------------------------------
shower = mmod.MeteorShower(stage, rng, knob, jbox_size=JBOX_SIZE, out_dir=OUT)

# --- logs (streamed to disk; flushed on a cadence so a crash loses <1 interval)
SENSOR_HEADER = ("t_s,utc,px,py,pz,qw,qx,qy,qz,vx,vy,vz,wx_dps,wy_dps,wz_dps,range_m,boresight_prim,"
                 "clearance_m,lum_mean,lum_max,sun_visible,solar_wm2,rad_cps,dv_mps,thrust_scale,shower_active,impacts")
FLIGHT_HEADER = "t_s,px,py,pz,rx,ry,rz,drift_m,cmd_fx,cmd_fy,cmd_fz,cmd_tx,cmd_ty,cmd_tz,dv_mps,speed_mps"
sensor_fh = open(os.path.join(OUT, "sensor_log.csv"), "w", buffering=1 << 16)
sensor_fh.write(SENSOR_HEADER + "\n")
flight_fh = open(os.path.join(OUT, "flight_log.csv"), "w", buffering=1 << 16)
flight_fh.write(FLIGHT_HEADER + "\n")


def flush_logs():
    sensor_fh.flush()
    flight_fh.flush()


def atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def now_utc():
    return f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%S.%f}Z"


# --- selftest script (same handlers as live keys; smoke-test harness) --------
SELFTEST_SCRIPT = [
    (1.0, "hold", KI.W, 3.0), (5.0, "press", KI.P, 0), (7.0, "press", KI.G, 0),
    (9.0, "hold", KI.RIGHT, 1.5), (11.0, "hold", KI.SPACE, 2.5), (16.0, "press", KI.M, 0),
    (39.0, "point", None, 0), (40.0, "press", KI.P, 0), (42.0, "press", KI.G, 0),
    (52.0, "press", KI.ENTER, 0), (58.0, "press", KI.ESCAPE, 0),
]
selftest_until = {}
selftest_fired = set()


def selftest_point_at_target():
    # test-harness teleport of attitude only: boresight back on the origin
    xfc_l = UsdGeom.XformCache()
    pl = np.array(xfc_l.GetLocalToWorldTransform(craft_prim).ExtractTranslation())
    fw = -pl / (np.linalg.norm(pl) + 1e-9)
    rt = np.cross(fw, np.array([0.0, 0.0, 1.0]))
    rt /= np.linalg.norm(rt) + 1e-9
    upv = np.cross(rt, fw)
    mm = Gf.Matrix3d(rt[0], rt[1], rt[2], fw[0], fw[1], fw[2], upv[0], upv[1], upv[2])
    craft.GetOrderedXformOps()[1].Set(Gf.Quatf(mm.ExtractRotation().GetQuat()))
    rb.GetAngularVelocityAttr().Set(Gf.Vec3f(0, 0, 0))
    print("SELFTEST point-at-target teleport applied", flush=True)


def selftest_drive(t):
    for i, (st, kind, key, dur) in enumerate(SELFTEST_SCRIPT):
        if t >= st and i not in selftest_fired:
            selftest_fired.add(i)
            if kind == "hold":
                held.add(key)
                selftest_until[key] = st + dur
            elif kind == "point":
                selftest_point_at_target()
            else:
                class _E:
                    pass
                ev = _E()
                ev.type = carb.input.KeyboardEventType.KEY_PRESS
                ev.input = key
                on_key(ev)
                ev2 = _E()
                ev2.type = carb.input.KeyboardEventType.KEY_RELEASE
                ev2.input = key
                on_key(ev2)
    for key, until in list(selftest_until.items()):
        if t >= until:
            held.discard(key)
            del selftest_until[key]


# --- main loop ---------------------------------------------------------------
timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(int(knob("PILOT_WARMUP_UPDATES", 60, int))):
    sim.update()
print(f"PILOT ready: outputs -> {OUT}", flush=True)

xfc = UsdGeom.XformCache()
sq = get_physx_scene_query_interface()
CRAFT_CLEAR = max(BX * 1.7, BY, BZ) / 2 + PLUME_LEN + 0.1  # ray origin offset past own colliders
# RK4 reference starts from the measured post-warmup state, not the authored
# start, so the drift metric measures integration disagreement only
_m = xfc.GetLocalToWorldTransform(craft_prim)
ref = np.concatenate([np.array(_m.ExtractTranslation()), np.array(rb.GetVelocityAttr().Get())])
dv_used = 0.0
step = 0
t_sim = 0.0
photo_n = 0
sweep_n = 0
drift = 0.0
drift_max = 0.0
drift_sumsq = 0.0
last_sensor = last_hud = last_beat = last_flush = -1e9
sensor_snapshot = {"range_m": -1.0, "boresight_prim": "", "clearance_m": -1.0, "lum_mean": 0.0,
                   "lum_max": 0.0, "sun_visible": 1, "solar_wm2": SOLAR_WM2, "rad_cps": 0.0}
report_written = False
sensor_int = 1.0 / SENSOR_HZ
hud_int = 1.0 / HUD_HZ
wall0 = time.time()

BODY_AXES = ((KI.D, 0, 1), (KI.A, 0, -1), (KI.W, 1, 1), (KI.S, 1, -1), (KI.R, 2, 1), (KI.F, 2, -1))
TORQUE_KEYS = ((KI.UP, 0, 1), (KI.DOWN, 0, -1), (KI.LEFT, 2, 1), (KI.RIGHT, 2, -1), (KI.Q, 1, 1), (KI.E, 1, -1))

while sim.is_running() and not toggles["quit"] and t_sim < MAX_SESSION_S:
    if SELFTEST:
        selftest_drive(t_sim)

    # SPACE doubles as the Kit play/pause hotkey in the streamed UI; flight
    # physics must not silently freeze, so force the timeline to keep playing
    if not timeline.is_playing():
        timeline.play()

    xfc.Clear()
    m = xfc.GetLocalToWorldTransform(craft_prim)
    p = np.array(m.ExtractTranslation())
    q = m.ExtractRotation().GetQuat()
    # pxr matrices act on row vectors; transpose+slice gives column-convention R
    R = np.array(Gf.Matrix4d().SetRotate(q))[:3, :3].T
    v = np.array(rb.GetVelocityAttr().Get())
    w_dps = np.array(rb.GetAngularVelocityAttr().Get())
    w_rad = np.radians(w_dps)

    # commanded body-frame thrust and torque
    cmd_f = np.zeros(3)
    for key, axis, sgn in BODY_AXES:
        if key in held:
            cmd_f[axis] += sgn
    cmd_t = np.zeros(3)
    for key, axis, sgn in TORQUE_KEYS:
        if key in held:
            cmd_t[axis] += sgn
    F_body = cmd_f * THRUST_N * thrust_scale[0]
    T_body = cmd_t * TORQUE_NM * thrust_scale[0]

    # damping thrusters (SPACE = full damp; auto rate-damp kills residual spin)
    v_body = R.T @ v
    if KI.SPACE in held:
        F_body += np.clip(-DAMP_KV * MASS * v_body, -THRUST_N * thrust_scale[0], THRUST_N * thrust_scale[0])
        T_body += np.clip(-DAMP_KW * np.array([IX, IY, IZ]) * (R.T @ w_rad),
                          -TORQUE_NM * thrust_scale[0], TORQUE_NM * thrust_scale[0])
    elif toggles["auto_damp"] and not any(k in held for k, _, _ in TORQUE_KEYS):
        T_body += np.clip(-DAMP_KW * np.array([IX, IY, IZ]) * (R.T @ w_rad),
                          -TORQUE_NM * thrust_scale[0], TORQUE_NM * thrust_scale[0])

    # delta-v budget: RCS cut when spent
    if dv_used >= DV_BUDGET:
        F_body[:] = 0.0
        T_body[:] = 0.0
    dv_used += float(np.linalg.norm(F_body)) / MASS * DT

    F_thr_world = R @ F_body
    T_world = R @ T_body
    F_total = cw_accel(p, v) * MASS + F_thr_world
    force_api.GetForceAttr().Set(Gf.Vec3f(*F_total))
    force_api.GetTorqueAttr().Set(Gf.Vec3f(*T_world))

    # plume indicators mirror the commanded thrust (plume fires opposite the push)
    for (ax_i, s), prim in plumes.items():
        want = F_body[ax_i] * s < -1e-6
        img = UsdGeom.Imageable(prim)
        img.MakeVisible() if want else img.MakeInvisible()

    # RK4 reference of the same ODE with the same applied thrust (drift evidence)
    def deriv(state):
        return np.concatenate([state[3:], cw_accel(state[:3], state[3:]) + F_thr_world / MASS])
    k1 = deriv(ref)
    k2 = deriv(ref + DT / 2 * k1)
    k3 = deriv(ref + DT / 2 * k2)
    k4 = deriv(ref + DT * k3)
    ref = ref + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    drift = float(np.linalg.norm(p - ref[:3]))

    # shower event
    if toggles["shower_req"] and not shower.armed:
        shower.arm(t_sim)
    rad_shower_cps = shower.step(t_sim, DT, sq)

    # sensors on cadence (rays hop past the invisible keep-out shell and own craft)
    if t_sim - last_sensor >= sensor_int:
        last_sensor = t_sim
        fwd = R @ np.array([0.0, 1.0, 0.0])
        o = p + fwd * CRAFT_CLEAR
        hit, hdist = mmod.raycast_filtered(sq, o, fwd, RANGEFINDER_MAX_M)
        rng_m = hdist + CRAFT_CLEAR if hit is not None else -1.0
        bs_prim = str(hit.get("collision", "")) if hit is not None else ""
        to_c = -p / (np.linalg.norm(p) + 1e-9)
        oc = p + to_c * CRAFT_CLEAR
        chit, cdist = mmod.raycast_filtered(sq, oc, to_c, RANGEFINDER_MAX_M)
        clear_m = cdist + CRAFT_CLEAR if chit is not None else -1.0
        os_ = p + TO_SUN * CRAFT_CLEAR
        shit, _sd = mmod.raycast_filtered(sq, os_, TO_SUN, SUN_OCCLUDE_MAX_M)
        sun_vis = 0 if shit is not None else 1
        px_data = photometer.get_data()
        if px_data is not None and getattr(px_data, "size", 0) > 0:
            rgbf = np.asarray(px_data)[:, :, :3].astype(np.float32)
            lum = 0.2126 * rgbf[:, :, 0] + 0.7152 * rgbf[:, :, 1] + 0.0722 * rgbf[:, :, 2]
            lum_mean, lum_max = float(lum.mean()), float(lum.max())
        else:
            lum_mean = lum_max = -1.0
        lam = (RAD_BG_CPS * (1.0 + RAD_SOLAR_FACTOR * sun_vis) + rad_shower_cps) / SENSOR_HZ
        rad_cps = float(rng.poisson(max(lam, 0.0))) * SENSOR_HZ
        sensor_snapshot = {"range_m": rng_m, "boresight_prim": bs_prim, "clearance_m": clear_m,
                           "lum_mean": lum_mean, "lum_max": lum_max, "sun_visible": sun_vis,
                           "solar_wm2": SOLAR_WM2 * sun_vis, "rad_cps": rad_cps}
        sensor_fh.write(f"{t_sim:.2f},{now_utc()},{p[0]:.3f},{p[1]:.3f},{p[2]:.3f},"
                        f"{q.GetReal():.5f},{q.GetImaginary()[0]:.5f},{q.GetImaginary()[1]:.5f},{q.GetImaginary()[2]:.5f},"
                        f"{v[0]:.4f},{v[1]:.4f},{v[2]:.4f},{w_dps[0]:.3f},{w_dps[1]:.3f},{w_dps[2]:.3f},"
                        f"{rng_m:.2f},{bs_prim},{clear_m:.2f},{lum_mean:.2f},{lum_max:.2f},{sun_vis},"
                        f"{SOLAR_WM2 * sun_vis:.1f},{rad_cps:.1f},{dv_used:.3f},{thrust_scale[0]:.3f},"
                        f"{int(shower.active)},{len(shower.impacts)}\n")

    flight_fh.write(f"{t_sim:.3f},{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},{ref[0]:.4f},{ref[1]:.4f},{ref[2]:.4f},"
                    f"{drift:.4f},{F_body[0]:.3f},{F_body[1]:.3f},{F_body[2]:.3f},"
                    f"{T_body[0]:.4f},{T_body[1]:.4f},{T_body[2]:.4f},{dv_used:.4f},{np.linalg.norm(v):.4f}\n")
    drift_max = max(drift_max, drift)
    drift_sumsq += drift * drift

    # photo request
    while toggles["photo_req"] > 0:
        toggles["photo_req"] -= 1
        photo_path = os.path.join(OUT, "photos", f"{photo_n:03d}.png")
        capture_viewport_to_file(viewport, photo_path)
        meta = {"file": photo_path, "utc": now_utc(), "t_s": round(t_sim, 3),
                "pos": p.round(4).tolist(), "quat_wxyz": [round(q.GetReal(), 5)] + [round(c, 5) for c in q.GetImaginary()],
                "vel": v.round(4).tolist(), "range_m": sensor_snapshot["range_m"],
                "boresight_prim": sensor_snapshot["boresight_prim"], "sun_azimuth_deg": round(az, 2),
                "epoch": EPOCH, "renderer": "rtx_realtime_streamed", "dv_used_mps": round(dv_used, 3),
                "focal_mm": FPV_FOCAL_MM,
                "hfov_deg": round(math.degrees(2 * math.atan((fpv.GetHorizontalApertureAttr().Get() / 2) / FPV_FOCAL_MM)), 2),
                "view": "fpv" if toggles["fpv"] else "chase", "sensors": sensor_snapshot.copy()}
        atomic_write(photo_path + ".json", json.dumps(meta, indent=1))
        photo_n += 1
        print(f"PILOT photo {photo_n} -> {photo_path} range={sensor_snapshot['range_m']:.1f} m", flush=True)

    # sweep request (freeze-frame Replicator capture)
    while toggles["sweep_req"] > 0:
        toggles["sweep_req"] -= 1
        sweep_dir = os.path.join(OUT, "sweeps", f"{sweep_n:03d}")
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=sweep_dir, rgb=True, distance_to_camera=True,
                          semantic_segmentation=True, colorize_semantic_segmentation=True)
        writer.attach([sweep_rp])
        t_w = time.time()
        # step() is synchronous (wait_for_render=True); an extra
        # wait_until_complete() here deadlocks the app - do not add one
        rep.orchestrator.step(rt_subframes=SWEEP_SUBFRAMES, pause_timeline=True)
        writer.detach()
        meta = {"dir": sweep_dir, "utc": now_utc(), "t_s": round(t_sim, 3), "pos": p.round(4).tolist(),
                "quat_wxyz": [round(q.GetReal(), 5)] + [round(c, 5) for c in q.GetImaginary()],
                "boresight_prim": sensor_snapshot["boresight_prim"], "range_m": sensor_snapshot["range_m"],
                "annotators": ["rgb", "distance_to_camera", "semantic_segmentation"],
                "resolution": [SWEEP_W, SWEEP_H], "rt_subframes": SWEEP_SUBFRAMES,
                "focal_mm": FPV_FOCAL_MM, "sensors": sensor_snapshot.copy()}
        atomic_write(os.path.join(sweep_dir, "sidecar.json"), json.dumps(meta, indent=1))
        sweep_n += 1
        print(f"PILOT sweep {sweep_n} -> {sweep_dir} ({time.time() - t_w:.1f}s)", flush=True)

    # report request
    if toggles["report_req"] > 0:
        toggles["report_req"] = 0
        flush_logs()
        rpt = mmod.write_inspection_report(OUT, epoch=EPOCH, seed=SEED, warp=TIME_WARP,
                                           shower=shower, dv_used=dv_used, tsi_wm2=TSI_1AU, r_au=R_AU)
        report_written = True
        print(f"PILOT report -> {rpt}", flush=True)

    # view toggle
    want_cam = FPV_PATH if toggles["fpv"] else "/World/Inspector/chase_cam"
    if viewport.camera_path != want_cam:
        viewport.camera_path = want_cam

    # HUD
    if t_sim - last_hud >= hud_int:
        last_hud = t_sim
        rng_d = np.linalg.norm(p)
        closing = float(np.dot(v, -p / (rng_d + 1e-9)))
        hud_label.text = (
            f"t {t_sim:7.1f} s   range {rng_d:7.2f} m   closing {closing:+.3f} m/s   speed {np.linalg.norm(v):.3f} m/s\n"
            f"body rates {w_dps.round(2).tolist()} deg/s   thrust scale {thrust_scale[0]:.3f}   "
            f"auto-damp {'ON' if toggles['auto_damp'] else 'OFF'}\n"
            f"dv {dv_used:6.2f} / {DV_BUDGET:.1f} m/s {'  DV DEPLETED' if dv_used >= DV_BUDGET else ''}\n"
            f"keep-out margin {rng_d - KEEPOUT_M:7.2f} m   clearance {sensor_snapshot['clearance_m']:.1f} m\n"
            f"rangefinder {sensor_snapshot['range_m']:8.2f} m -> {sensor_snapshot['boresight_prim'].split('/')[-1][:40]}\n"
            f"light mean/max {sensor_snapshot['lum_mean']:.1f}/{sensor_snapshot['lum_max']:.1f}   "
            f"sun {'VISIBLE' if sensor_snapshot['sun_visible'] else 'ECLIPSED'} {sensor_snapshot['solar_wm2']:.0f} W/m2   "
            f"radiation {sensor_snapshot['rad_cps']:.0f} cps\n"
            f"epoch {EPOCH}   warp {TIME_WARP:.0e}   photos {photo_n}   sweeps {sweep_n}\n"
            f"{shower.hud_line()}")

    # heartbeat + periodic flush
    if t_sim - last_beat >= HEARTBEAT_S:
        last_beat = t_sim
        print(f"PILOT t={t_sim:7.1f}s p={p.round(2).tolist()} |v|={np.linalg.norm(v):.3f} m/s dv={dv_used:.2f} "
              f"drift={drift:.3f} m rad={sensor_snapshot['rad_cps']:.0f}cps photos={photo_n} sweeps={sweep_n} "
              f"impacts={len(shower.impacts)} fps={step / max(time.time() - wall0, 1e-9):.0f}", flush=True)
    if t_sim - last_flush >= FLUSH_S:
        last_flush = t_sim
        flush_logs()

    sim.update()
    step += 1
    t_sim = step * DT

# --- shutdown ----------------------------------------------------------------
elapsed = time.time() - wall0
sensor_fh.close()
flight_fh.close()
if shower.impacts:
    shower.save_damaged_stage(stage)
    if not report_written:
        rpt = mmod.write_inspection_report(OUT, epoch=EPOCH, seed=SEED, warp=TIME_WARP,
                                           shower=shower, dv_used=dv_used, tsi_wm2=TSI_1AU, r_au=R_AU)
        print(f"PILOT report (auto) -> {rpt}", flush=True)
print(f"PILOT DONE steps={step} sim={t_sim:.1f}s wall={elapsed:.1f}s ({step / max(elapsed, 1e-9):.1f} steps/s) "
      f"dv={dv_used:.2f}/{DV_BUDGET:.0f} m/s photos={photo_n} sweeps={sweep_n} impacts={len(shower.impacts)}", flush=True)
print(f"PILOT physx-vs-RK4 drift: final={drift:.3f} m max={drift_max:.3f} m "
      f"rms={math.sqrt(drift_sumsq / max(step, 1)):.3f} m (thruster use and collisions legitimately diverge the reference)", flush=True)
print(f"PILOT outputs: {OUT} (flight_log.csv, sensor_log.csv, photos/, sweeps/"
      f"{', impacts.csv, stage_damaged.usd, inspection_report.md' if shower.impacts else ''})", flush=True)

if SELFTEST:
    checks = {"flight_log": os.path.getsize(os.path.join(OUT, "flight_log.csv")) > 0,
              "sensor_log": os.path.getsize(os.path.join(OUT, "sensor_log.csv")) > 0,
              "photos": photo_n >= 2, "sweeps": sweep_n >= 2, "impacts": len(shower.impacts) > 0,
              "report": os.path.exists(os.path.join(OUT, "inspection_report.md")),
              "damaged_stage": os.path.exists(os.path.join(OUT, "stage_damaged.usd")),
              "dv_spent": dv_used > 0.0}
    for k, ok in checks.items():
        print(f"SELFTEST {'PASS' if ok else 'FAIL'} {k}", flush=True)
    fail = [k for k, ok in checks.items() if not ok]
    print(f"SELFTEST DONE fails={fail}", flush=True)
    timeline.stop()
    sim.close()
    sys.exit(1 if fail else 0)

timeline.stop()
sim.close()
