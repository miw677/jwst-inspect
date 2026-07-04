# Helper for `jwst-gui render`: open a USD stage headless, add a sun + star dome
# if it has no lights, path-trace one frame from a framed camera, write a PNG.
# Runs inside the Isaac Sim container. Inputs from env (set by jwst-gui).
import os

from isaacsim import SimulationApp

sim = SimulationApp({"headless": True})

import carb
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdLux

stage_path = os.environ["RENDER_STAGE"]
out_path = os.environ["RENDER_OUT"]
width = int(os.environ.get("RENDER_W", "1920"))
height = int(os.environ.get("RENDER_H", "1080"))
spp = int(os.environ.get("RENDER_SPP", "128"))
star = os.environ.get("RENDER_STARMAP", "/data/shared/raw/env_lighting/starmaps/starmap_2020_8k.exr")

ctx = omni.usd.get_context()
assert ctx.open_stage(stage_path), f"could not open {stage_path}"
stage = ctx.get_stage()
s = carb.settings.get_settings()
s.set("/rtx/rendermode", "PathTracing")
s.set("/rtx/pathtracing/spp", spp)
s.set("/rtx/pathtracing/totalSpp", spp)

# Bounds for framing; add lighting only if the stage carries none.
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
box = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
center = Gf.Vec3d(box.GetMidpoint())
diag = max(Gf.Vec3d(box.GetSize()).GetLength(), 1e-3)

# Add a view sun + star dome unless the stage already has a strong enough light
# (many converted GLB/USDZ carry only a token light and render near-black).
min_intensity = float(os.environ.get("RENDER_MIN_LIGHT", "500"))
existing = [
    float(p.GetAttribute("inputs:intensity").Get() or 0.0)
    for p in stage.Traverse()
    if "Light" in p.GetTypeName() and p.GetAttribute("inputs:intensity")
]
if not existing or max(existing) < min_intensity:
    sun = UsdLux.DistantLight.Define(stage, "/World/_ViewSun")
    sun.CreateIntensityAttr(float(os.environ.get("RENDER_SUN", "3500")))
    sun.CreateAngleAttr(0.53)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45, 25, 0))
    dome = UsdLux.DomeLight.Define(stage, "/World/_ViewStars")
    dome.CreateIntensityAttr(float(os.environ.get("RENDER_DOME", "2500")))
    if os.path.exists(star):
        dome.CreateTextureFileAttr(star)
        dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    print(f"added a view sun + star dome (stage lights {existing} below {min_intensity})", flush=True)

up = UsdGeom.GetStageUpAxis(stage)
upv = Gf.Vec3d(0, 0, 1) if up == UsdGeom.Tokens.z else Gf.Vec3d(0, 1, 0)
off = (Gf.Vec3d(0.7, -0.9, 0.5) if up == UsdGeom.Tokens.z else Gf.Vec3d(0.7, 0.5, 0.9)).GetNormalized()
eye = center + off * diag * 1.1
cam = UsdGeom.Camera.Define(stage, "/World/_ViewCam")
cam.CreateClippingRangeAttr(Gf.Vec2f(max(diag * 1e-4, 0.01), float(diag * 20)))
UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(
    Gf.Matrix4d().SetLookAt(eye, center, upv).GetInverse())

for _ in range(120):
    sim.update()

out_dir = os.path.dirname(out_path)
rp = rep.create.render_product("/World/_ViewCam", (width, height))
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=out_dir, rgb=True)
writer.attach([rp])
with rep.trigger.on_frame(num_frames=1):
    pass
rep.orchestrator.run()
rep.orchestrator.wait_until_complete()

# BasicWriter names it rgb_0000.png; rename to the requested output.
src = os.path.join(out_dir, "rgb_0000.png")
if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(out_path):
    os.replace(src, out_path)
print(f"wrote {out_path}", flush=True)
sim.close()
