#!/usr/bin/env python3
"""Paired rasterized-vs-path-traced (R2P) evaluation harness (Group 3).

Scores each available policy (scripted / behavior-cloning / PPO) on held-out seeds
under two appearance-fidelity conditions and exports coverage, mean standoff error,
safety violations, abort rate, and relative velocity, plus the R2P gap (rasterized
minus path-traced). The conditions model the appearance/sensor gap that path tracing
exposes on the specular gold optics: 'rasterized' = clean low-noise observations a
fast renderer yields; 'path_traced' = specular-glare sensor noise + intermittent
depth dropout. Writes raster_vs_path_traced_eval.csv, failure_taxonomy.md, and
policy_eval_report.md. (The image-in-the-loop version, feeding Replicator-rendered
frames to a visual policy, is the scale-up driven by eval_r2p.sbatch.)

  OUT=/data/shared/checkpoints python eval_r2p.py
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from free_flyer_env import FreeFlyerVecEnv, FreeFlyerConfig, knob
from scripted_policy import ScriptedInspector

OUT = knob("OUT", "/data/shared/checkpoints", str)
EVAL_SEED = knob("EVAL_SEED", 4242, int)
EVAL_ENVS = knob("EVAL_ENVS", 128, int)
EVAL_STEPS = knob("EVAL_STEPS", 1200, int)
RASTER_NOISE = knob("FF_RASTER_NOISE", 0.0)
PT_NOISE = knob("FF_PT_NOISE", 0.06)          # specular-glare-driven obs noise
PT_DROPOUT = knob("FF_PT_DROPOUT", 0.1)        # intermittent depth/standoff dropout


def load_torch_policy(path):
    import torch
    ckpt = torch.load(path, map_location="cpu")
    obs, act = ckpt["obs_dim"], ckpt["act_dim"]
    import torch.nn as nn
    if ckpt.get("kind") == "bc":
        net = nn.Sequential(nn.Linear(obs, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(),
                            nn.Linear(256, act), nn.Tanh())
        net.load_state_dict({k.replace("net.", ""): v for k, v in ckpt["model"].items()})
        fwd = lambda o: net(o)
    else:  # ppo actor-critic: use the action mean
        from train_ppo import ActorCritic
        net = ActorCritic(obs, act)
        net.load_state_dict(ckpt["model"])
        fwd = lambda o: net(o)[0]
    net.eval()
    return net, fwd


def run(policy_kind, fwd, env, steps):
    import numpy as np
    keepout = 0; aborts = 0; timeouts = 0
    cov_final = []; standoff = []; speed = []
    obs = env.reset()
    expert = ScriptedInspector(env.cfg) if policy_kind == "scripted" else None
    for _ in range(steps):
        if policy_kind == "scripted":
            a = expert.act(env)
        else:
            import torch
            with torch.no_grad():
                a = fwd(torch.as_tensor(obs, dtype=torch.float32)).numpy()
            a = np.clip(a, -1, 1)
        obs, _, done, info = env.step(a)
        keepout += int(info["in_keepout"].sum())
        aborts += int((info["abort"] & ~info["timeout"]).sum())
        timeouts += int(info["timeout"].sum())
        standoff.append(float(info["standoff_err"].mean()))
        speed.append(float(info["speed"].mean()))
        cov_final.append(float(info["coverage"].mean()))
    episodes = max(1, aborts + timeouts)
    return dict(final_coverage=float(np.mean(cov_final[-50:])),
                mean_standoff_err=float(np.mean(standoff)),
                safety_violations=keepout, abort_rate=aborts / episodes,
                mean_rel_velocity=float(np.mean(speed)))


def make_env(condition):
    cfg = FreeFlyerConfig()
    cfg.sensor_noise = RASTER_NOISE if condition == "rasterized" else PT_NOISE
    env = FreeFlyerVecEnv(num_envs=EVAL_ENVS, cfg=cfg, seed=EVAL_SEED)
    if condition == "path_traced" and PT_DROPOUT > 0:
        # wrap step to intermittently drop the depth-derived obs (standoff/range)
        base_step = env.step
        rng = np.random.default_rng(EVAL_SEED + 1)

        def step(a):
            o, r, d, i = base_step(a)
            mask = rng.random(o.shape[0]) < PT_DROPOUT
            o[mask, 9] = 0.0; o[mask, 12] = 0.0   # standoff_err, range columns
            return o, r, d, i
        env.step = step
    return env


def main():
    os.makedirs(OUT, exist_ok=True)
    policies = [("scripted", None)]
    for kind, ck in [("bc", os.path.join(OUT, "bc", "bc_policy_checkpoint.pt")),
                     ("ppo", os.path.join(OUT, "ppo", "ppo_policy_checkpoint.pt"))]:
        if os.path.exists(ck):
            policies.append((kind, ck))

    rows = []
    metrics_keys = ["final_coverage", "mean_standoff_err", "safety_violations",
                    "abort_rate", "mean_rel_velocity"]
    results = {}
    for kind, ck in policies:
        fwd = None
        if ck:
            _, fwd = load_torch_policy(ck)
        results[kind] = {}
        for cond in ("rasterized", "path_traced"):
            m = run(kind, fwd, make_env(cond), EVAL_STEPS)
            results[kind][cond] = m
            rows.append(dict(policy=kind, condition=cond, **m))
            print(f"{kind:9s} {cond:12s} cov={m['final_coverage']:.3f} "
                  f"standoff_err={m['mean_standoff_err']:.3f} safety={m['safety_violations']} "
                  f"abort={m['abort_rate']:.3f} vel={m['mean_rel_velocity']:.3f}", flush=True)
        # R2P gap row
        gap = {k: results[kind]["rasterized"][k] - results[kind]["path_traced"][k] for k in metrics_keys}
        rows.append(dict(policy=kind, condition="r2p_gap", **gap))

    csv_path = os.path.join(OUT, "raster_vs_path_traced_eval.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "condition"] + metrics_keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(os.path.join(OUT, "failure_taxonomy.md"), "w") as f:
        f.write("# Failure-mode taxonomy (JWST-Inspect autonomous)\n\n"
                "Observed/instrumented failure modes for the inspection policies:\n\n"
                "- keep_out_breach: inspector entered the keep-out sphere (`safety_violations`).\n"
                "- abort_overspeed: relative velocity exceeded 2x the limit -> abort.\n"
                "- abort_range: drifted past the max range or inside the abort boundary.\n"
                "- coverage_stall: viewpoint coverage plateaued below target.\n"
                "- standoff_drift: sustained standoff error under sensor noise.\n\n"
                "The R2P-specific failures (induced by path-traced specular glare / depth\n"
                "dropout, absent under the rasterized condition) are quantified by the\n"
                "rasterized-minus-path_traced gap in raster_vs_path_traced_eval.csv.\n")

    with open(os.path.join(OUT, "policy_eval_report.md"), "w") as f:
        f.write("# Policy evaluation report (scripted vs learned, R2P)\n\n"
                "Metrics cite raster_vs_path_traced_eval.csv (this run). Conditions: a\n"
                "clean 'rasterized' sensor model vs a 'path_traced' model with specular-glare\n"
                "noise + depth dropout. The R2P gap is rasterized minus path-traced.\n\n")
        for kind in results:
            f.write(f"## {kind}\n\n")
            for cond in ("rasterized", "path_traced"):
                m = results[kind][cond]
                f.write(f"- {cond}: coverage {m['final_coverage']:.3f}, standoff_err "
                        f"{m['mean_standoff_err']:.3f} m, safety {m['safety_violations']}, "
                        f"abort {m['abort_rate']:.3f}, rel_vel {m['mean_rel_velocity']:.3f} m/s\n")
            f.write("\n")
    json.dump(results, open(os.path.join(OUT, "r2p_results.json"), "w"), indent=2)
    print(f"wrote {csv_path}, failure_taxonomy.md, policy_eval_report.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
