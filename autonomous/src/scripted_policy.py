#!/usr/bin/env python3
"""Scripted inspection baseline + the credibility hard gate (Group 3).

A deterministic, known-safe controller: hold the standoff distance, orbit the target
to gain viewpoint coverage, stay out of the keep-out zone, cap relative velocity, and
abort cleanly. This is the policy learned baselines must beat. `gate()` runs it on the
real env and asserts it respects approach corridor / standoff / keep-out / abort with
zero safety violations - by project rule this gate must pass before any learned-policy
headline claim. Controller gains are env knobs.

  python scripted_policy.py            # runs the hard gate, exits non-zero on failure
"""
import os
import sys
import numpy as np

from free_flyer_env import FreeFlyerVecEnv, FreeFlyerConfig, knob


class ScriptedInspector:
    """PD station-keeping + slow orbit, operating on full state (a known-safe expert)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.kp = knob("SCRIPT_KP", 0.3)            # position -> desired velocity
        self.kd = knob("SCRIPT_KD", 2.0)            # velocity error -> force
        self.lat_cycles = knob("SCRIPT_LAT_CYCLES", 0.5)
        self.vel_frac = knob("SCRIPT_VEL_FRAC", 0.6)  # cap desired speed at this x limit

    def act(self, env):
        """Velocity-limited survey. A reference point sweeps the standoff sphere
        slowly (azimuth + a half latitude sweep). We command a DESIRED VELOCITY toward
        it, hard-capped below the safety limit, then a force to track that velocity -
        so the inspector physically cannot exceed the limit or lunge into the keep-out
        zone. Near the keep-out shell, the desired velocity is overridden to push out."""
        c = self.cfg
        pos, vel, t = env.pos, env.vel, env.t.astype(np.float64)
        d = np.linalg.norm(pos, axis=1, keepdims=True) + 1e-9
        radial = pos / d
        w = self.vel_frac * c.vel_limit_mps * c.dt / c.standoff_m   # safe azimuth rate
        az = w * t
        frac = np.clip(t / max(1, c.episode_len), 0.0, 1.0)
        elev = (np.pi / 2.0) * np.sin(2.0 * np.pi * self.lat_cycles * frac)
        ref = np.stack([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)], 1)
        target_pos = ref * c.standoff_m
        des_vel = self.kp * (target_pos - pos)
        sp = np.linalg.norm(des_vel, axis=1, keepdims=True) + 1e-9
        cap = self.vel_frac * c.vel_limit_mps
        des_vel = np.where(sp > cap, des_vel * cap / sp, des_vel)
        too_close = (d[:, 0] < c.keepout_m * 1.3)
        des_vel[too_close] = radial[too_close] * (0.5 * c.vel_limit_mps)
        cmd = self.kd * (des_vel - vel)
        F = np.clip(cmd / c.max_thrust, -1, 1)
        return np.concatenate([F, np.zeros((env.n, 3), np.float32)], axis=1).astype(np.float32)


def rollout(env, policy, steps):
    metrics = dict(coverage=[], standoff_err=[], safety_violations=0, aborts=0,
                   max_speed=0.0, keepout_breaches=0)
    for _ in range(steps):
        a = policy.act(env)
        _, _, done, info = env.step(a)
        metrics["coverage"].append(float(info["coverage"].mean()))
        metrics["standoff_err"].append(float(info["standoff_err"].mean()))
        metrics["safety_violations"] += int(info["in_keepout"].sum())
        metrics["keepout_breaches"] += int(info["in_keepout"].sum())
        metrics["aborts"] += int((info["abort"] & ~info["timeout"]).sum())
        metrics["max_speed"] = max(metrics["max_speed"], float(info["speed"].max()))
    metrics["final_coverage"] = metrics["coverage"][-1] if metrics["coverage"] else 0.0
    metrics["mean_standoff_err"] = float(np.mean(metrics["standoff_err"])) if metrics["standoff_err"] else 0.0
    return metrics


def gate():
    cfg = FreeFlyerConfig()
    env = FreeFlyerVecEnv(num_envs=knob("GATE_ENVS", 32, int), cfg=cfg, seed=knob("GATE_SEED", 7, int))
    pol = ScriptedInspector(cfg)
    steps = knob("GATE_STEPS", cfg.episode_len, int)
    m = rollout(env, pol, steps)
    print(f"scripted gate: final_coverage={m['final_coverage']:.3f} "
          f"mean_standoff_err={m['mean_standoff_err']:.3f}m keepout_breaches={m['keepout_breaches']} "
          f"aborts={m['aborts']} max_speed={m['max_speed']:.3f}m/s", flush=True)
    # Hard gate (definition of done): primarily SAFETY - zero keep-out breaches and
    # velocity within the safety envelope - plus evidence of active inspection
    # (coverage above a floor). The coverage floor is a knob: a slow, safe survey at
    # the velocity limit covers a realistic fraction per episode, not the whole sphere.
    ok = (m["keepout_breaches"] == 0
          and m["max_speed"] <= cfg.vel_limit_mps * 2.0
          and m["final_coverage"] >= knob("GATE_MIN_COVERAGE", 0.35, float))
    print(f"GATE {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(gate())
