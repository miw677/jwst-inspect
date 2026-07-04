#!/usr/bin/env python3
"""Acceptance test for the published JWST-Inspect scene (Group 1 definition of done).

Loads the scene with usd-core and asserts: it opens without composition errors, the
JWST component hierarchy + semantic tags are present, the 18 primary-mirror segment
frames exist, the three safety regions are authored, the lighting variant set is
present, and a camera exists. Exits non-zero on any failure (this is a gate). Run:
    SCENE=/data/shared/assets/jwst_inspect_scene_v1.usd python scene_load_test.py
"""
import os
import re
import sys

from pxr import Usd, UsdGeom


def knob(name, default):
    v = os.environ.get(name)
    return v if v else default


def main():
    scene = knob("SCENE", "/data/shared/assets/jwst_inspect_scene_v1.usd")
    print(f"loading {scene}", flush=True)
    if not os.path.exists(scene):
        print(f"FAIL: scene not found: {scene}")
        return 1
    stage = Usd.Stage.Open(scene)
    if stage is None:
        print("FAIL: stage did not open")
        return 1

    checks, failed = [], 0

    def ck(name, cond):
        nonlocal failed
        ok = bool(cond)
        checks.append((name, ok))
        if not ok:
            failed += 1
        print(f"  [{'OK' if ok else 'FAIL'}] {name}", flush=True)

    ck("/World defaultPrim", stage.GetDefaultPrim() and stage.GetDefaultPrim().GetPath() == "/World")
    jwst = stage.GetPrimAtPath("/World/JWST")
    ck("/World/JWST present", jwst.IsValid())
    ck("JWST semanticClass=jwst", jwst.IsValid() and jwst.GetAttribute("semanticClass").Get() == "jwst")

    # 18 primary-mirror segment frames (the inst_NN Xforms, not their hex children)
    segs = [p for p in stage.Traverse()
            if re.fullmatch(r"/World/JWST/primary_mirror/inst_\d{2}", p.GetPath().pathString)]
    ck("18 primary-mirror segment frames", len(segs) == 18)
    gold = [p for p in stage.Traverse()
            if p.IsA(UsdGeom.Mesh) and p.GetAttribute("semanticClass").Get() == "primary_mirror_segment"]
    ck("18 gold mirror-segment meshes", len(gold) == 18)

    for region in ("keep_out_zone", "standoff_shell", "approach_corridor"):
        ck(f"safety region {region}", stage.GetPrimAtPath(f"/World/SafetyRegions/{region}").IsValid())

    lighting = stage.GetPrimAtPath("/World/Lighting")
    ck("lighting variant set", lighting.IsValid() and "lighting" in lighting.GetVariantSets().GetNames())

    cams = [p for p in stage.Traverse() if p.IsA(UsdGeom.Camera)]
    ck("at least one camera", len(cams) >= 1)

    labeled = [p for p in stage.Traverse() if p.GetAttribute("semanticClass").IsValid()]
    ck("semantic tags authored (>=20)", len(labeled) >= 20)

    print(f"\nPASS {sum(1 for _, o in checks if o)}/{len(checks)}; FAIL {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
