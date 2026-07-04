#!/usr/bin/env python3
"""Behavior cloning from the scripted expert (Group 3).

Collects (observation, scripted-action) pairs by rolling out the known-safe scripted
controller, then trains an MLP policy to imitate it from observations alone (a
deployable visual-state policy, unlike the state-omniscient scripted expert).
Checkpoints bc_policy_checkpoint.pt. Knobs are env vars. Runs in the jwst-rl env.

  OUT=/data/shared/checkpoints/bc BC_EPISODES=200 python behavior_cloning.py
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from free_flyer_env import FreeFlyerVecEnv, FreeFlyerConfig, knob
from scripted_policy import ScriptedInspector


class MLPPolicy(nn.Module):
    def __init__(self, obs, act, hid=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(),
                                 nn.Linear(hid, act), nn.Tanh())

    def forward(self, x):
        return self.net(x)


def main():
    OUT = knob("OUT", "/data/shared/checkpoints/bc", str)
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seed = knob("SEED", 20260627, int)
    torch.manual_seed(seed); np.random.seed(seed)
    n_envs = knob("NUM_ENVS", 128, int)
    horizon = knob("BC_HORIZON", 600, int)
    epochs = knob("BC_EPOCHS", 40, int)

    cfg = FreeFlyerConfig()
    env = FreeFlyerVecEnv(num_envs=n_envs, cfg=cfg, seed=seed)
    expert = ScriptedInspector(cfg)

    log_("collecting expert demonstrations")
    O, A = [], []
    obs = env.reset()
    for _ in range(horizon):
        a = expert.act(env)            # expert acts on full state
        O.append(obs.copy()); A.append(a.copy())
        obs, _, _, _ = env.step(a)
    Ob = torch.as_tensor(np.concatenate(O), dtype=torch.float32, device=dev)
    Ac = torch.as_tensor(np.concatenate(A), dtype=torch.float32, device=dev)
    log_(f"dataset: {Ob.shape[0]} (obs, action) pairs")

    net = MLPPolicy(env.obs_dim, env.act_dim).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=knob("BC_LR", 1e-3))
    bs = knob("BC_BATCH", 4096, int)
    N = Ob.shape[0]
    losses = []
    for ep in range(epochs):
        perm = torch.randperm(N, device=dev); tot = 0.0
        for s in range(0, N, bs):
            j = perm[s:s + bs]
            loss = ((net(Ob[j]) - Ac[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        losses.append(tot)
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            log_(f"epoch {ep+1}/{epochs} mse={tot/max(1,N//bs):.5f}")
    torch.save({"model": net.state_dict(), "obs_dim": env.obs_dim, "act_dim": env.act_dim,
                "kind": "bc"}, os.path.join(OUT, "bc_policy_checkpoint.pt"))
    json.dump({"epochs": epochs, "pairs": int(N), "final_mse": losses[-1] / max(1, N // bs)},
              open(os.path.join(OUT, "bc_train_log.json"), "w"), indent=2)
    log_(f"BC checkpoint saved -> {OUT}/bc_policy_checkpoint.pt")
    return 0


def log_(m):
    import datetime as dt
    print(f"[{dt.datetime.utcnow():%H:%M:%S}Z] {m}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
