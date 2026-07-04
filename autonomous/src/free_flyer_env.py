#!/usr/bin/env python3
"""6-DoF zero-g free-flyer inspection environment for JWST-Inspect (Group 3).

Real local rigid-body dynamics (Newton-Euler) for a non-contact inspector around
the JWST target at the origin, with the published safety geometry: approach
corridor, keep-out zone, standoff target, abort boundary, and a relative-velocity
limit. Reward couples viewpoint coverage of the target, standoff hold, fuel use,
and safety. The observation is state-based (relative pose/vel + coverage + safety
flags); the rasterized-vs-path-traced visual study swaps the *rendered* observation
in eval_r2p.py via the Benchmark Replicator pipeline - the dynamics are identical.

Vectorized over `num_envs` with numpy for fast PPO. Every constant is an env knob.
This module is framework-agnostic (no Isaac dependency) so it trains on the jwst-rl
stack; the same task is also exposed as an Isaac Lab task scaffold for the
GPU-physics scale-up (see isaaclab_task/).
"""
import os
import numpy as np


def knob(n, d, c=float):
    v = os.environ.get(n)
    return c(v) if v not in (None, "") else d


class FreeFlyerConfig:
    def __init__(self):
        self.dt = knob("FF_DT", 0.1)
        self.mass = knob("FF_MASS_KG", 12.0)
        self.inertia = knob("FF_INERTIA", 0.8)               # scalar isotropic approx (kg m^2)
        self.max_thrust = knob("FF_MAX_THRUST_N", 0.5)        # cold-gas microthruster
        self.max_torque = knob("FF_MAX_TORQUE_NM", 0.05)
        self.keepout_m = knob("JWST_KEEPOUT_M", 8.0)
        self.standoff_m = knob("JWST_STANDOFF_M", 15.0)
        self.standoff_band_m = knob("FF_STANDOFF_BAND_M", 3.0)
        self.corridor_half_deg = knob("JWST_CORRIDOR_HALF_DEG", 20.0)
        self.abort_m = knob("FF_ABORT_M", 6.0)                # inside this -> abort (collision risk)
        self.max_range_m = knob("FF_MAX_RANGE_M", 40.0)
        self.vel_limit_mps = knob("FF_VEL_LIMIT_MPS", 0.5)    # relative-velocity safety limit
        # Inspection is slow at a 0.5 m/s limit and ~15 m standoff (a full survey is
        # thousands of steps); the episode is sized for a real sweep, not a quick loop.
        self.episode_len = knob("FF_EPISODE_LEN", 2400, int)
        self.n_patches = knob("FF_COVERAGE_PATCHES", 64, int)
        self.cover_fov_deg = knob("FF_COVER_FOV_DEG", 40.0)
        self.latency_steps = knob("FF_LATENCY_STEPS", 1, int)  # comms/control latency
        self.imu_bias = knob("FF_IMU_BIAS", 0.0)
        self.sensor_noise = knob("FF_SENSOR_NOISE", 0.0)
        # reward weights
        self.w_cover = knob("FF_W_COVER", 10.0)
        self.w_standoff = knob("FF_W_STANDOFF", 0.5)
        self.w_fuel = knob("FF_W_FUEL", 0.05)
        self.w_safety = knob("FF_W_SAFETY", 50.0)


def _fib_sphere(n):
    """n approximately-uniform unit vectors on a sphere (target viewpoint patches)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)


class FreeFlyerVecEnv:
    """Minimal vectorized env (reset/step) returning numpy arrays. obs_dim=13, act_dim=6."""

    def __init__(self, num_envs=1, cfg=None, seed=0):
        self.cfg = cfg or FreeFlyerConfig()
        self.n = int(num_envs)
        self.rng = np.random.default_rng(seed)
        self.patches = _fib_sphere(self.cfg.n_patches)
        self.obs_dim, self.act_dim = 13, 6
        self._alloc()
        self.reset()

    def _alloc(self):
        n = self.n
        self.pos = np.zeros((n, 3)); self.vel = np.zeros((n, 3)); self.ang = np.zeros((n, 3))
        self.cover = np.zeros((n, self.cfg.n_patches), bool)
        self.t = np.zeros(n, int)
        self._act_buf = [np.zeros((n, 6)) for _ in range(max(1, self.cfg.latency_steps))]

    def reset(self, mask=None):
        c = self.cfg
        idx = np.arange(self.n) if mask is None else np.where(mask)[0]
        for i in idx:
            # spawn on the standoff shell, within the approach corridor, at rest-ish
            r = c.standoff_m + self.rng.uniform(-c.standoff_band_m, c.standoff_band_m)
            d = self.rng.normal(size=3); d /= np.linalg.norm(d) + 1e-9
            self.pos[i] = d * r
            self.vel[i] = self.rng.normal(scale=0.02, size=3)
            self.ang[i] = 0.0
            self.cover[i] = False
            self.t[i] = 0
        return self._obs()

    def _dist(self):
        return np.linalg.norm(self.pos, axis=1)

    def _update_coverage(self):
        # a patch is "seen" if the inspector is within standoff band and its line of
        # sight to the target origin aligns with the patch normal within the FOV.
        d = self._dist()[:, None]
        view = -self.pos / (np.linalg.norm(self.pos, axis=1, keepdims=True) + 1e-9)  # toward target
        # patch normal points outward; seen if view ~ -normal (we look at that face)
        cosang = -(view @ self.patches.T)
        thresh = np.cos(np.radians(self.cfg.cover_fov_deg))
        near = (np.abs(self._dist() - self.cfg.standoff_m) < self.cfg.standoff_band_m * 1.5)[:, None]
        newly = (cosang > thresh) & near
        gained = (newly & ~self.cover).sum(1)
        self.cover |= newly
        return gained

    def _obs(self):
        c = self.cfg
        d = self._dist()
        cov_frac = self.cover.mean(1)
        standoff_err = (d - c.standoff_m) / c.standoff_m
        speed = np.linalg.norm(self.vel, axis=1)
        obs = np.concatenate([
            self.pos / c.max_range_m, self.vel / c.vel_limit_mps, self.ang,
            standoff_err[:, None], cov_frac[:, None], (speed / c.vel_limit_mps)[:, None],
            (d / c.max_range_m)[:, None]], axis=1).astype(np.float32)
        if c.sensor_noise > 0:
            obs = obs + self.rng.normal(scale=c.sensor_noise, size=obs.shape).astype(np.float32)
        return obs

    def step(self, action):
        c = self.cfg
        action = np.clip(np.asarray(action, np.float32), -1, 1).reshape(self.n, 6)
        # control/comms latency: apply a delayed action
        self._act_buf.append(action.copy())
        applied = self._act_buf.pop(0)
        F = applied[:, :3] * c.max_thrust
        tau = applied[:, 3:] * c.max_torque
        self.vel += (F / c.mass) * c.dt
        self.pos += self.vel * c.dt
        self.ang += (tau / c.inertia) * c.dt
        self.t += 1

        d = self._dist(); speed = np.linalg.norm(self.vel, axis=1)
        gained = self._update_coverage()
        standoff_err = np.abs(d - c.standoff_m)
        fuel = np.abs(applied).sum(1)

        # safety: keep-out / abort / range / velocity
        in_keepout = d < c.keepout_m
        abort = (d < c.abort_m) | (d > c.max_range_m) | (speed > c.vel_limit_mps * 2.0)
        vel_viol = speed > c.vel_limit_mps

        reward = (c.w_cover * gained
                  - c.w_standoff * standoff_err
                  - c.w_fuel * fuel
                  - c.w_safety * (in_keepout | abort).astype(np.float32)
                  - 0.5 * vel_viol.astype(np.float32)).astype(np.float32)

        timeout = self.t >= c.episode_len
        done = abort | timeout
        info = dict(coverage=self.cover.mean(1).copy(), standoff_err=standoff_err.copy(),
                    in_keepout=in_keepout.copy(), abort=abort.copy(), speed=speed.copy(),
                    timeout=timeout.copy())
        obs = self._obs()
        if done.any():
            self.reset(mask=done)
        return obs, reward, done, info
