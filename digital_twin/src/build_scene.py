#!/usr/bin/env python3
"""Assemble the JWST-Inspect OpenUSD scene from public geometry + labels + zones.

Skeleton for the Digital Twin team. Runs with usd-core (jwst-usd conda env) for
pure-USD authoring, or inside the Isaac Sim / Kit container when RTX features are
needed. Follows the project rules: every tunable is an env knob with an inline
default; every step prints what it did; content (paths, labels) comes from files,
not hardcoded literals.

Pipeline (fill in the TODOs against the definition of done in our README):
  1. import NASA GLB/STL seed from /data/shared/raw/jwst_geometry
  2. build the component hierarchy + apply semantic labels (semantic_labels_v0.json)
  3. author coordinate frames + safety regions (coordinate_frames.md)
  4. attach MDL material variants + lighting variants
  5. export jwst_inspect_scene_v1.usd and copy to /data/shared/assets/
"""
import json
import os
from pathlib import Path


def knob(name: str, default: str) -> str:
    return os.environ.get(name, default)


def main() -> int:
    raw = Path(knob("JWST_RAW", "/data/shared/raw/jwst_geometry"))
    here = Path(__file__).resolve().parent.parent / "interface"
    out = Path(knob("JWST_SCENE_OUT", "/data/shared/assets/jwst_inspect_scene_v1.usda"))
    labels = json.loads((here / "semantic_labels_v0.json").read_text())
    print(f"build_scene: raw={raw} out={out} classes={len(labels['classes'])}")

    from pxr import Usd, UsdGeom  # noqa: F401  (import here so --help works without USD)

    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out)) if not out.exists() else Usd.Stage.Open(str(out))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # TODO 1: import + clean NASA geometry from `raw` (usdz/gltf import or mesh rebuild).
    # TODO 2: build /World/JWST hierarchy; tag each prim with its semantic class.
    # TODO 3: author /World/SafetyRegions per coordinate_frames.md.
    # TODO 4: bind MDL materials (gold, MLI, sunshield, foil) + lighting variants.
    print("build_scene: TODO sections are stubs - implement against the definition of done in our README")

    stage.GetRootLayer().Save()
    print(f"build_scene: wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
