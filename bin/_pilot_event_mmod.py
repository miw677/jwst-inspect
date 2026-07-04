# MMOD meteor-shower event + inspection report for jwst-gui pilot.
# Grounded in the real JWST C3 segment micrometeoroid strike class documented in
# the commissioning report (staged at /data/shared/raw/inspector_refs/); shower
# rates, sizes, and speeds are visual-scale knobs printed for provenance, not a
# flux-model claim. Meteoroid motion is exact zero-g ballistics (constant
# velocity) applied kinematically so meteoroids cannot shove the craft or the
# telescope; impacts are detected by sweeping a ray along each meteoroid's path
# every physics step against the telescope collision meshes. Every hit authors
# a labeled scar prim and a CSV row.
import json
import os
from datetime import datetime, timezone

import carb
import numpy as np
from isaacsim.core.utils.semantics import add_update_semantics
from pxr import Gf, Sdf, UsdGeom

IMPACTS_HEADER = "impact_id,t_s,utc,hit_prim,part,x,y,z,speed_mps,meteoroid_r_m,meteoroid_kg,energy_j,crater_r_m,scar_prim"
# prims that physics rays must see through: the invisible keep-out shell, the
# craft itself, scars, and the meteoroids
RAY_IGNORE_PREFIXES = ("/World/JWST_Keepout", "/World/Inspector", "/World/JWST_Damage", "/World/ShowerEvent")


def raycast_filtered(sq, origin, direction, max_dist, ignore_prefixes=RAY_IGNORE_PREFIXES):
    """Closest hit whose collider is not in ignore_prefixes, via raycast_all.

    Returns (hit_dict, distance) or (None, -1.0). raycast_all reports every
    collider along the ray in one call, so ignored prims (the invisible
    keep-out shell, the craft, scars, meteoroids) are simply skipped.
    """
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    best = {}

    def on_hit(h):
        prim = str(h.collision)
        if any(prim.startswith(p) for p in ignore_prefixes):
            return True
        if not best or float(h.distance) < best["distance"]:
            best.update({"collision": prim, "distance": float(h.distance),
                         "position": tuple(h.position), "normal": tuple(h.normal)})
        return True

    sq.raycast_all(carb.Float3(*np.asarray(origin, float)), carb.Float3(*d), float(max_dist), on_hit)
    if not best:
        return None, -1.0
    return best, best["distance"]


class MeteorShower:
    """Seeded meteoroid stream with swept-ray impact detection and scar authoring."""

    def __init__(self, stage, rng, knob, jbox_size, out_dir):
        self.stage = stage
        self.rng = rng
        self.out = out_dir
        self.n_total = knob("PILOT_SHOWER_N", 24, int)
        self.aim_frac = knob("PILOT_SHOWER_AIM_FRAC", 0.5)
        self.v_mps = knob("PILOT_SHOWER_VMPS", 40.0)
        self.duration_s = knob("PILOT_SHOWER_DURATION_S", 20.0)
        self.spawn_range_m = knob("PILOT_SHOWER_SPAWN_M", 120.0)
        self.rad_cps_active = knob("PILOT_SHOWER_RAD_CPS", 220.0)
        self.density_kgm3 = knob("PILOT_SHOWER_DENSITY_KGM3", 2500.0)   # stony meteoroid
        self.v_jitter = knob("PILOT_SHOWER_V_JITTER", 0.3)
        self.r_min = knob("PILOT_SHOWER_R_MIN_M", 0.06)
        self.r_max = knob("PILOT_SHOWER_R_MAX_M", 0.22)
        self.crater_scale = knob("PILOT_CRATER_SCALE", 0.35)
        self.crater_ref_j = knob("PILOT_CRATER_REF_J", 2000.0)
        self.crater_min_m = knob("PILOT_CRATER_MIN_M", 0.12)
        self.crater_max_m = knob("PILOT_CRATER_MAX_M", 0.9)
        self.jbox_half = float(np.max(jbox_size)) / 2.0
        self.aim_spread = knob("PILOT_SHOWER_AIM_SPREAD", 0.35)
        # radiant: seeded direction, biased above the sunshield plane
        u = self.rng.normal(size=3)
        u[2] = abs(u[2]) * 0.5 + 0.3
        self.radiant = u / np.linalg.norm(u)
        self.armed = False
        self.active = False
        self.t_arm = -1.0
        self.spawned = 0
        self.meteors = []          # dicts: prim, xform_op, pos, vel, alive
        self.impacts = []          # dicts matching IMPACTS_HEADER
        UsdGeom.Xform.Define(stage, "/World/ShowerEvent")
        UsdGeom.Xform.Define(stage, "/World/JWST_Damage")
        print(f"MMOD shower ready (idle): N={self.n_total} aim_frac={self.aim_frac} v={self.v_mps} m/s "
              f"duration={self.duration_s}s radiant={self.radiant.round(3).tolist()} "
              f"(visual-scale knobs; reference class: JWST C3 MMOD strike)", flush=True)

    def arm(self, t_sim):
        self.armed = True
        self.active = True
        self.t_arm = t_sim
        print(f"MMOD SHOWER ARMED t={t_sim:.1f}s: {self.n_total} meteoroids over {self.duration_s}s, "
              f"~{self.aim_frac * 100:.0f}% aimed at the telescope", flush=True)

    def _spawn_one(self, idx):
        aimed = self.rng.uniform() < self.aim_frac
        if aimed:
            aim = self.rng.uniform(-self.aim_spread, self.aim_spread, size=3) * self.jbox_half
        else:
            off = self.rng.normal(size=3)
            off /= np.linalg.norm(off) + 1e-12
            aim = off * self.jbox_half * self.rng.uniform(2.5, 5.0)
        pos = aim + self.radiant * self.spawn_range_m
        speed = self.v_mps * float(self.rng.uniform(1.0 - self.v_jitter, 1.0 + self.v_jitter))
        vel = -self.radiant * speed
        r = float(self.rng.uniform(self.r_min, self.r_max))
        mass = self.density_kgm3 * 4.0 / 3.0 * np.pi * r ** 3
        path = f"/World/ShowerEvent/meteor_{idx:03d}"
        s = UsdGeom.Sphere.Define(self.stage, path)
        s.CreateRadiusAttr(r)
        s.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.5, 0.45)])
        op = UsdGeom.Xformable(s).AddTranslateOp()
        op.Set(Gf.Vec3d(*pos))
        self.meteors.append({"prim": s.GetPrim(), "op": op, "pos": pos, "vel": vel,
                             "radius_m": r, "mass_kg": mass, "alive": True})

    def step(self, t_sim, dt, sq):
        """Spawn on schedule, advance ballistics, sweep-detect impacts. Returns radiation cps term."""
        if not self.active:
            return 0.0
        frac = np.clip((t_sim - self.t_arm) / self.duration_s, 0.0, 1.0)
        want = int(round(frac * self.n_total))
        while self.spawned < want:
            self._spawn_one(self.spawned)
            self.spawned += 1
        alive = 0
        for m in self.meteors:
            if not m["alive"]:
                continue
            step_len = float(np.linalg.norm(m["vel"])) * dt
            hit, dist = raycast_filtered(sq, m["pos"], m["vel"], step_len * 2.0)
            if hit is not None and str(hit.get("collision", "")).startswith("/World/JWST"):
                self._impact(t_sim, m, hit)
                m["alive"] = False
                m["prim"].SetActive(False)
                continue
            m["pos"] = m["pos"] + m["vel"] * dt
            m["op"].Set(Gf.Vec3d(*m["pos"]))
            if np.linalg.norm(m["pos"]) > self.spawn_range_m * 3:
                m["alive"] = False
                m["prim"].SetActive(False)
                continue
            alive += 1
        if frac >= 1.0 and alive == 0 and self.active:
            self.active = False
            print(f"MMOD shower complete t={t_sim:.1f}s: {len(self.impacts)} impacts recorded", flush=True)
            self.write_impacts_csv()
        return self.rad_cps_active if self.active else 0.0

    def _impact(self, t_sim, m, hit):
        idx = len(self.impacts)
        pos = np.array([hit["position"][0], hit["position"][1], hit["position"][2]])
        nrm = np.array([hit["normal"][0], hit["normal"][1], hit["normal"][2]])
        if np.linalg.norm(nrm) < 1e-6:
            nrm = -m["vel"] / (np.linalg.norm(m["vel"]) + 1e-12)
        speed = float(np.linalg.norm(m["vel"]))
        energy = 0.5 * m["mass_kg"] * speed * speed
        crater_r = float(np.clip(self.crater_scale * (energy / self.crater_ref_j) ** (1.0 / 3.0),
                                 self.crater_min_m, self.crater_max_m))
        hit_prim = str(hit.get("collision", ""))
        part = hit_prim.rstrip("/").split("/")[-1] if hit_prim else "unknown"
        scar_path = f"/World/JWST_Damage/scar_{idx:03d}"
        # crater: flattened dark disc oriented to the surface normal, slightly proud
        scar = UsdGeom.Sphere.Define(self.stage, scar_path)
        scar.CreateRadiusAttr(crater_r)
        scar.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.04, 0.03)])
        zw = Gf.Vec3d(*nrm)
        xw = Gf.Cross(Gf.Vec3d(0, 0, 1) if abs(nrm[2]) < 0.95 else Gf.Vec3d(1, 0, 0), zw)
        xw = xw.GetNormalized()
        yw = Gf.Cross(zw, xw)
        rot = Gf.Matrix4d(xw[0], xw[1], xw[2], 0, yw[0], yw[1], yw[2], 0, zw[0], zw[1], zw[2], 0, 0, 0, 0, 1)
        xf = UsdGeom.Xformable(scar)
        xf.AddTranslateOp().Set(Gf.Vec3d(*(pos + nrm * 0.01)))
        xf.AddTransformOp(opSuffix="orient").Set(rot)
        xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 0.12))
        prim = scar.GetPrim()
        prim.CreateAttribute("semanticClass", Sdf.ValueTypeNames.Token).Set("anomaly_mmod")
        add_update_semantics(prim, "anomaly_mmod")
        prim.CreateAttribute("mmod:hitPrim", Sdf.ValueTypeNames.String).Set(hit_prim)
        prim.CreateAttribute("mmod:energyJ", Sdf.ValueTypeNames.Float).Set(energy)
        rec = {"impact_id": idx, "t_s": round(t_sim, 3),
               "utc": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%S.%f}Z",
               "hit_prim": hit_prim, "part": part,
               "x": round(float(pos[0]), 3), "y": round(float(pos[1]), 3), "z": round(float(pos[2]), 3),
               "speed_mps": round(speed, 2), "meteoroid_r_m": round(m["radius_m"], 3),
               "meteoroid_kg": round(m["mass_kg"], 3), "energy_j": round(energy, 2),
               "crater_r_m": round(crater_r, 3), "scar_prim": scar_path}
        self.impacts.append(rec)
        print(f"MMOD IMPACT {idx}: part={part} p={pos.round(2).tolist()} v={speed:.1f} m/s "
              f"r={m['radius_m']:.2f} m E={energy:.0f} J crater_r={crater_r:.2f} m -> {scar_path}", flush=True)

    def hud_line(self):
        if not self.armed:
            return "shower: idle (M to arm)"
        state = "ACTIVE" if self.active else "complete"
        return f"shower: {state}   meteoroids {self.spawned}/{self.n_total}   IMPACTS {len(self.impacts)}"

    def write_impacts_csv(self):
        path = os.path.join(self.out, "impacts.csv")
        rows = [IMPACTS_HEADER] + [
            f"{r['impact_id']},{r['t_s']},{r['utc']},{r['hit_prim']},{r['part']},{r['x']},{r['y']},{r['z']},"
            f"{r['speed_mps']},{r['meteoroid_r_m']},{r['meteoroid_kg']},{r['energy_j']},{r['crater_r_m']},{r['scar_prim']}"
            for r in self.impacts]
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            fh.write("\n".join(rows) + "\n")
        os.replace(tmp, path)
        print(f"MMOD impacts.csv written: {len(self.impacts)} rows -> {path}", flush=True)
        return path

    def save_damaged_stage(self, stage):
        # session layer only: references stay live paths, scars and poses included
        path = os.path.join(self.out, "stage_damaged.usd")
        ok = stage.GetRootLayer().Export(path)
        if not ok or not os.path.exists(path):
            raise RuntimeError(f"ERROR: damaged stage export failed: {path}")
        print(f"MMOD damaged stage saved -> {path} ({os.path.getsize(path)} bytes)", flush=True)
        return path


def _same_part(a, b):
    """True when two collider paths identify the same part: equal, nested, or siblings."""
    if not a or not b:
        return False
    if a == b or a.startswith(b) or b.startswith(a):
        return True
    return os.path.dirname(a) == os.path.dirname(b)


def write_inspection_report(out_dir, epoch, seed, warp, shower, dv_used, tsi_wm2, r_au):
    """Cross-link impacts with photo/sweep sidecars and the sensor log into markdown."""
    impacts = shower.impacts
    if impacts:
        shower.write_impacts_csv()
    photos = []
    pdir = os.path.join(out_dir, "photos")
    for name in sorted(os.listdir(pdir)) if os.path.isdir(pdir) else []:
        if name.endswith(".json"):
            with open(os.path.join(pdir, name)) as fh:
                photos.append(json.load(fh))
    sweeps = []
    sdir = os.path.join(out_dir, "sweeps")
    for name in sorted(os.listdir(sdir)) if os.path.isdir(sdir) else []:
        sc = os.path.join(sdir, name, "sidecar.json")
        if os.path.exists(sc):
            with open(sc) as fh:
                sweeps.append(json.load(fh))

    def evidence_for(rec):
        ph = [p for p in photos if _same_part(str(p.get("boresight_prim", "")), rec["hit_prim"])]
        sw = [s for s in sweeps if _same_part(str(s.get("boresight_prim", "")), rec["hit_prim"])]
        return ph, sw

    # radiation spike window from the sensor log
    spike = ""
    slog = os.path.join(out_dir, "sensor_log.csv")
    if os.path.exists(slog):
        with open(slog) as fh:
            header = fh.readline().strip().split(",")
            i_t, i_rad, i_sh = header.index("t_s"), header.index("rad_cps"), header.index("shower_active")
            ts, rads, act = [], [], []
            for line in fh:
                c = line.rstrip("\n").split(",")
                if len(c) != len(header):
                    continue
                ts.append(float(c[i_t]))
                rads.append(float(c[i_rad]))
                act.append(int(c[i_sh]))
        if any(act):
            on = [t for t, a in zip(ts, act) if a]
            r_on = [r for r, a in zip(rads, act) if a]
            r_off = [r for r, a in zip(rads, act) if not a] or [0.0]
            spike = (f"Radiation spiked during the shower window t=[{min(on):.1f}, {max(on):.1f}] s: "
                     f"mean {np.mean(r_on):.0f} cps active vs {np.mean(r_off):.0f} cps background "
                     f"(peak {max(rads):.0f} cps). Source: sensor_log.csv columns rad_cps/shower_active.")

    lines = []
    lines.append("# Inspection report: MMOD shower damage survey")
    lines.append("")
    lines.append(f"Generated {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} by jwst-gui pilot. "
                 f"Epoch {epoch}, seed {seed}, time warp {warp:.0e}, TSIS-1 TSI {tsi_wm2:.1f} W/m2 at 1 au, "
                 f"sun range {r_au:.4f} au. Reference damage class: JWST C3 segment MMOD strike "
                 f"(commissioning report, /data/shared/raw/inspector_refs/).")
    lines.append("")
    if not impacts:
        lines.append("No impacts recorded. Arm the shower with M (or PILOT_EVENT=shower) before requesting a report.")
    else:
        details = []
        covered = 0
        lines.append(f"## Impact summary: {len(impacts)} recorded impacts")
        lines.append("")
        lines.append("| id | t (s) | part | speed (m/s) | meteoroid r (m) | energy (J) | crater r (m) | photos | sweeps |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for rec in impacts:
            ph, sw = evidence_for(rec)
            covered += 1 if (ph or sw) else 0
            lines.append(f"| {rec['impact_id']} | {rec['t_s']} | {rec['part']} | {rec['speed_mps']} | "
                         f"{rec['meteoroid_r_m']} | {rec['energy_j']} | {rec['crater_r_m']} | {len(ph)} | {len(sw)} |")
            details.append((rec, ph, sw))
        lines.append("")
        lines.append(f"Survey coverage: {covered}/{len(impacts)} impacts have at least one photo or sweep "
                     f"whose boresight was on the damaged part. Delta-v spent this flight: {dv_used:.2f} m/s.")
        lines.append("")
        if spike:
            lines.append("## Sensor evidence")
            lines.append("")
            lines.append(spike)
            lines.append("")
        lines.append("## Per-impact detail")
        lines.append("")
        for rec, ph, sw in details:
            lines.append(f"### Impact {rec['impact_id']}: {rec['part']}")
            lines.append("")
            lines.append(f"- Hit prim: `{rec['hit_prim']}`")
            lines.append(f"- Scar prim: `{rec['scar_prim']}` (semanticClass anomaly_mmod)")
            lines.append(f"- Position ({rec['x']}, {rec['y']}, {rec['z']}) m, speed {rec['speed_mps']} m/s, "
                         f"meteoroid radius {rec['meteoroid_r_m']} m ({rec['meteoroid_kg']} kg), "
                         f"energy proxy {rec['energy_j']} J, crater radius {rec['crater_r_m']} m")
            for p in ph:
                lines.append(f"- Photo: {os.path.relpath(p['file'], out_dir)} (t={p['t_s']} s, "
                             f"range {p.get('range_m', -1):.1f} m)")
            for s in sw:
                lines.append(f"- Sweep: {os.path.relpath(s['dir'], out_dir)}/ (t={s['t_s']} s, rgb+depth+semantic)")
            if not ph and not sw:
                lines.append("- NOT PHOTOGRAPHED: no capture had its boresight on this part; fly back and document it")
            lines.append("")
    lines.append("## Full data")
    lines.append("")
    lines.append("- impacts.csv: every impact row (id, time, prim, position, speed, energy, crater, scar)")
    lines.append("- sensor_log.csv: continuous telemetry (pose, rates, rangefinder, light, radiation, dv)")
    lines.append("- flight_log.csv: per-step trajectory vs RK4 reference, commands, dv")
    lines.append("- photos/NNN.png + .json: FPV captures with full state sidecars")
    lines.append("- sweeps/NNN/: Replicator rgb + distance_to_camera + semantic_segmentation + sidecar.json")
    lines.append("- stage_damaged.usd: the damaged scene with labeled anomaly_mmod scars, reloadable for re-inspection")
    path = os.path.join(out_dir, "inspection_report.md")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    print(f"MMOD inspection report written -> {path}", flush=True)
    return path
