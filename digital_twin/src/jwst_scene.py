#!/usr/bin/env python3
"""Build the JWST-Inspect OpenUSD benchmark scene from the public NASA geometry.

Group 1 (Digital Twin) core. Loads the NASA 3D Resources GLB seed (data_plan.md
Dataset 1), authors a clean USD stage with the JWST component hierarchy + semantic
class tags (groups/digital_twin/interface/semantic_labels_v0.json), parameterized
safety regions (keep-out, standoff shell, approach corridor), lighting variants,
and a viewing camera. Writes jwst_digital_twin_stub.usd (geometry-light, fast to
load) and jwst_inspect_scene_v1.usd (full), plus asset_manifest.csv and a sidecar.

Everything dimensional is an env knob with an inline default (no hardcoded science
constants in source). Run in the jwst-usd env (provides pxr + trimesh):
    JWST_GLB=/data/shared/raw/jwst_geometry/James_Webb_Space_Telescope_B.glb \
    OUT_DIR=/data/shared/assets python jwst_scene.py
"""
import csv
import datetime as dt
import hashlib
import json
import os
import sys

from pxr import Usd, UsdGeom, UsdLux, UsdShade, Gf, Sdf, Vt

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS_JSON = os.path.join(HERE, "..", "interface", "semantic_labels_v0.json")


def knob(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v is not None and v != "" else default


def log(msg):
    print(f"[{dt.datetime.utcnow():%H:%M:%S}Z] {msg}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_components():
    with open(LABELS_JSON) as f:
        spec = json.load(f)
    return spec


def map_node(name, components):
    """Return (component_path, class_name) for a GLB node by substring match."""
    low = name.lower()
    for comp in components:
        for token in comp["match"]:
            if token in low:
                return comp["path"], comp["class"]
    return "geometry", "background"


def author_mesh(stage, parent_path, name, tm, scale, semantic_class):
    """Author a UsdGeom.Mesh from a trimesh geometry under parent_path."""
    safe = "".join(c if c.isalnum() else "_" for c in name)[:48] or "mesh"
    path = f"{parent_path}/{safe}"
    mesh = UsdGeom.Mesh.Define(stage, path)
    verts = [Gf.Vec3f(float(v[0]) * scale, float(v[1]) * scale, float(v[2]) * scale)
             for v in tm.vertices]
    mesh.CreatePointsAttr(Vt.Vec3fArray(verts))
    counts = [3] * len(tm.faces)
    idx = [int(i) for f in tm.faces for i in f]
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    prim = mesh.GetPrim()
    prim.CreateAttribute("semanticClass", Sdf.ValueTypeNames.Token).Set(semantic_class)
    return path, len(verts), len(counts)


_CLASS_MTL = {
    "primary_mirror_segment": "gold_mirror", "secondary_mirror": "gold_mirror",
    "spacecraft_bus": "mli_blanket", "instrument_housing": "mli_blanket",
    "sunshield_layer": "sunshield",
    "truss": "foil_truss", "secondary_support_strut": "foil_truss", "backplane": "foil_truss",
}


def author_materials(stage):
    """MDL material variants for the specular/thermal surfaces (gold mirror, MLI,
    sunshield, foil/truss). Uses OmniPBR.mdl, evaluated by the RTX path tracer; the
    gold specular reflectance is central to the rasterized-to-path-traced claim.
    Reflectance constants are env knobs."""
    UsdGeom.Scope.Define(stage, "/World/Looks")
    defs = {
        "gold_mirror": dict(color=(1.0, 0.84, 0.30),
                            metallic=1.0, rough=knob("JWST_GOLD_ROUGHNESS", 0.04, float)),
        "mli_blanket": dict(color=(0.85, 0.78, 0.45), metallic=0.8, rough=0.35),
        "sunshield": dict(color=(0.70, 0.65, 0.55), metallic=0.6, rough=0.5),
        "foil_truss": dict(color=(0.60, 0.60, 0.62), metallic=0.9, rough=0.2),
    }
    mats = {}
    for name, p in defs.items():
        mp = f"/World/Looks/{name}"
        mtl = UsdShade.Material.Define(stage, mp)
        sh = UsdShade.Shader.Define(stage, mp + "/Shader")
        sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*p["color"]))
        sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(float(p["metallic"]))
        sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(p["rough"]))
        mtl.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        mats[name] = mtl
    return mats


def bind_materials(stage, mats):
    n = 0
    for prim in stage.Traverse():
        a = prim.GetAttribute("semanticClass")
        cls = a.Get() if a.IsValid() else None
        mname = _CLASS_MTL.get(cls)
        if mname and prim.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(prim)
            UsdShade.MaterialBindingAPI(prim).Bind(mats[mname])
            n += 1
    return n


def author_primary_mirror_segments(stage, scale):
    """Author the 18 gold hexagonal primary-mirror segments in a honeycomb layout
    (the real JWST primary is 18 segments, no center). Flat-to-flat, inter-segment
    gap, and mirror thickness are env knobs. Each segment is a labeled mesh that
    bind_materials() then binds to the gold MDL - the specular target for the R2P
    study. The NASA GLB is unsegmented, so these labeled segments are authored
    parametrically rather than guessed from GLB node names."""
    import math
    f = knob("JWST_SEG_FLAT_M", 1.32, float) * scale
    gap = knob("JWST_SEG_GAP_M", 0.007, float) * scale
    thick = knob("JWST_SEG_THICK_M", 0.06, float) * scale
    pitch = f + gap
    R = f / math.sqrt(3.0)
    coords = [(q, r) for q in range(-2, 3) for r in range(-2, 3)
              if abs(q + r) <= 2 and not (q == 0 and r == 0)][:18]
    ang = [math.radians(60 * k) for k in range(6)]
    top = [(R * math.cos(a), thick / 2, R * math.sin(a)) for a in ang]
    bot = [(R * math.cos(a), -thick / 2, R * math.sin(a)) for a in ang]
    pts = [Gf.Vec3f(*p) for p in (top + bot)]
    faces = [[0, 1, 2, 3, 4, 5], [11, 10, 9, 8, 7, 6]]
    for k in range(6):
        nx = (k + 1) % 6
        faces.append([k, nx, 6 + nx, 6 + k])
    counts = Vt.IntArray([len(fc) for fc in faces])
    idx = Vt.IntArray([i for fc in faces for i in fc])
    n = 0
    for (q, r) in coords:
        x = pitch * (q + r / 2.0)
        z = pitch * (math.sqrt(3.0) / 2.0) * r
        m = UsdGeom.Mesh.Define(stage, f"/World/JWST/primary_mirror/inst_{n:02d}/hex")
        m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(p[0] + x, p[1], p[2] + z) for p in pts]))
        m.CreateFaceVertexCountsAttr(counts)
        m.CreateFaceVertexIndicesAttr(idx)
        m.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        m.GetPrim().CreateAttribute("semanticClass", Sdf.ValueTypeNames.Token).Set("primary_mirror_segment")
        n += 1
    return n


def author_safety_regions(stage):
    """Approach corridor (cone), keep-out (sphere), standoff (sphere shell marker).

    All dimensions are env knobs (meters). Authored with purpose=guide so they are
    scene metadata, not rendered solids, and tagged with a semanticClass.
    """
    keepout_m = knob("JWST_KEEPOUT_M", 8.0, float)
    standoff_m = knob("JWST_STANDOFF_M", 15.0, float)
    corridor_len_m = knob("JWST_CORRIDOR_LEN_M", 30.0, float)
    corridor_half_deg = knob("JWST_CORRIDOR_HALF_DEG", 20.0, float)

    scope = UsdGeom.Scope.Define(stage, "/World/SafetyRegions")
    scope.GetPrim().CreateAttribute("semanticClass", Sdf.ValueTypeNames.Token).Set("safety")

    ko = UsdGeom.Sphere.Define(stage, "/World/SafetyRegions/keep_out_zone")
    ko.CreateRadiusAttr(keepout_m)
    ko.CreatePurposeAttr(UsdGeom.Tokens.guide)
    ko.GetPrim().CreateAttribute("semanticClass", Sdf.ValueTypeNames.Token).Set("keep_out_zone")

    so = UsdGeom.Sphere.Define(stage, "/World/SafetyRegions/standoff_shell")
    so.CreateRadiusAttr(standoff_m)
    so.CreatePurposeAttr(UsdGeom.Tokens.guide)
    so.GetPrim().CreateAttribute("semanticClass", Sdf.ValueTypeNames.Token).Set("standoff_shell")

    import math
    radius = corridor_len_m * math.tan(math.radians(corridor_half_deg))
    cone = UsdGeom.Cone.Define(stage, "/World/SafetyRegions/approach_corridor")
    cone.CreateHeightAttr(corridor_len_m)
    cone.CreateRadiusAttr(radius)
    cone.CreateAxisAttr(UsdGeom.Tokens.z)
    cone.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdGeom.Xformable(cone).AddTranslateOp().Set(Gf.Vec3d(0, 0, standoff_m + corridor_len_m / 2.0))
    cone.GetPrim().CreateAttribute("semanticClass", Sdf.ValueTypeNames.Token).Set("approach_corridor")
    return dict(keepout_m=keepout_m, standoff_m=standoff_m,
                corridor_len_m=corridor_len_m, corridor_half_deg=corridor_half_deg)


def author_lighting(stage):
    """Sun (distant) + dome fill, with a 'lighting' variant set: sun_high/low/eclipse."""
    lights = UsdGeom.Scope.Define(stage, "/World/Lighting").GetPrim()
    sun = UsdLux.DistantLight.Define(stage, "/World/Lighting/Sun")
    dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/Fill")
    sun_hi = knob("JWST_SUN_INTENSITY", 8.0, float)
    vset = lights.GetVariantSets().AddVariantSet("lighting")
    for name, sun_i, dome_i in [("sun_high", sun_hi, 0.3),
                                ("sun_low", sun_hi * 0.25, 0.15),
                                ("eclipse", 0.0, 0.05)]:
        vset.AddVariant(name)
        vset.SetVariantSelection(name)
        with vset.GetVariantEditContext():
            sun.CreateIntensityAttr(sun_i)
            dome.CreateIntensityAttr(dome_i)
    vset.SetVariantSelection("sun_high")


def author_camera(stage):
    standoff_m = knob("JWST_STANDOFF_M", 15.0, float)
    cam = UsdGeom.Camera.Define(stage, "/World/InspectorCam")
    cam.CreateFocalLengthAttr(knob("JWST_CAM_FOCAL_MM", 24.0, float))
    UsdGeom.Xformable(cam).AddTranslateOp().Set(Gf.Vec3d(0, -standoff_m, standoff_m * 0.3))


def build(stage_path, components_spec, glb_meshes, scale, full):
    stage = Usd.Stage.CreateNew(stage_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    jwst = UsdGeom.Xform.Define(stage, "/World/JWST")
    jwst.GetPrim().CreateAttribute("semanticClass", Sdf.ValueTypeNames.Token).Set("jwst")

    # Author the full component taxonomy as labeled Xforms (the reusable structure),
    # incl. the 18 primary-mirror segment frames, regardless of GLB granularity.
    comp_paths = {}
    for comp in components_spec["components"]:
        p = f"/World/JWST/{comp['path']}"
        x = UsdGeom.Xform.Define(stage, p)
        x.GetPrim().CreateAttribute("semanticClass", Sdf.ValueTypeNames.Token).Set(comp["class"])
        comp_paths[comp["path"]] = p
        for i in range(comp.get("instances", 1)):
            if comp.get("instances", 1) > 1:
                seg = UsdGeom.Xform.Define(stage, f"{p}/inst_{i:02d}")
                seg.GetPrim().CreateAttribute("semanticClass", Sdf.ValueTypeNames.Token).Set(comp["class"])

    seg_n = author_primary_mirror_segments(stage, scale)   # 18 labeled gold hex segments
    meshes_authored = seg_n
    verts_total = seg_n * 12
    if full:
        for name, tm in glb_meshes:
            cpath, cls = map_node(name, components_spec["components"])
            parent = comp_paths.get(cpath, "/World/JWST")
            _, nv, _ = author_mesh(stage, parent, name, tm, scale, cls)
            meshes_authored += 1
            verts_total += nv
    mats = author_materials(stage)
    bound = bind_materials(stage, mats)
    log(f"MDL materials bound to {bound} meshes (incl. {seg_n} gold mirror segments)")

    safety = author_safety_regions(stage)
    author_lighting(stage)
    author_camera(stage)
    # Inspector is published as a separate asset; reference it if present.
    insp = os.path.join(os.path.dirname(stage_path), "inspector_microsat.usd")
    if os.path.exists(insp):
        ip = UsdGeom.Xform.Define(stage, "/World/Inspector")
        ip.GetPrim().GetReferences().AddReference(f"./inspector_microsat.usd")
        UsdGeom.Xformable(ip).AddTranslateOp().Set(Gf.Vec3d(0, -safety["standoff_m"], 0))

    stage.GetRootLayer().Save()
    return meshes_authored, verts_total, safety


def main():
    glb = knob("JWST_GLB", "/data/shared/raw/jwst_geometry/James_Webb_Space_Telescope_B.glb")
    out_dir = knob("OUT_DIR", "/data/shared/assets")
    scale = knob("JWST_SCALE", 1.0, float)
    os.makedirs(out_dir, exist_ok=True)
    spec = load_components()

    log(f"loading NASA GLB seed: {glb}")
    glb_meshes = []
    if os.path.exists(glb):
        import trimesh
        scene = trimesh.load(glb, force="scene")
        for name, geom in scene.geometry.items():
            glb_meshes.append((name, geom))
        log(f"GLB nodes: {len(glb_meshes)}  total faces: {sum(len(g.faces) for _, g in glb_meshes)}")
    else:
        log(f"WARNING: GLB not found at {glb} - authoring labeled structure + safety regions only")

    stub = os.path.join(out_dir, "jwst_digital_twin_stub.usd")
    full = os.path.join(out_dir, "jwst_inspect_scene_v1.usd")
    for path in (stub, full):
        if os.path.exists(path):
            os.remove(path)

    log("authoring stub stage (structure + safety regions, geometry-light)")
    build(stub, spec, glb_meshes, scale, full=False)
    log("authoring full v1 stage (with GLB geometry)")
    m, v, safety = build(full, spec, glb_meshes, scale, full=True)
    log(f"v1: meshes={m} verts={v} safety={safety}")

    # asset manifest + sidecar
    manifest = os.path.join(out_dir, "asset_manifest.csv")
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["asset", "bytes", "sha256", "provenance", "built_utc"])
        now = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        for path, prov in [(stub, "authored from NASA 3D Resources GLB + label schema"),
                           (full, "authored from NASA 3D Resources GLB + label schema")]:
            w.writerow([os.path.basename(path), os.path.getsize(path),
                        sha256_file(path), prov, now])
        if os.path.exists(glb):
            w.writerow([os.path.basename(glb), os.path.getsize(glb), sha256_file(glb),
                        "NASA 3D Resources (public, see data_plan.md Dataset 1)", now])
    sidecar = os.path.join(out_dir, "jwst_inspect_scene_v1.sidecar.json")
    with open(sidecar, "w") as f:
        json.dump({"scene": os.path.basename(full), "meshes": m, "verts": v,
                   "scale": scale, "safety_regions": safety,
                   "glb_source": glb, "schema": "semantic_labels_v0.json",
                   "built_utc": now}, f, indent=2)
    log(f"published: {full}")
    log(f"manifest: {manifest}  sidecar: {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
