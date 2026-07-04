#!/usr/bin/env python3
"""PPO training for the JWST-Inspect free-flyer (Group 3).

A real clipped-PPO with GAE on the vectorized 6-DoF env. MLP Gaussian actor-critic.
Checkpoints ppo_policy_checkpoint.pt and logs episode return + coverage + safety to
a metrics JSON (and TensorBoard if available). All hyperparameters are env knobs.
Runs on GPU in the jwst-rl env. The learned policy must beat the scripted baseline
and clear the safety floor (checked in eval_r2p.py) before any headline claim.

  OUT=/data/shared/checkpoints/ppo TOTAL_STEPS=5000000 python train_ppo.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from free_flyer_env import FreeFlyerVecEnv, FreeFlyerConfig, knob


class ActorCritic(nn.Module):
    def __init__(self, obs, act, hid=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh())
        self.mu = nn.Linear(hid, act)
        self.v = nn.Linear(hid, 1)
        self.log_std = nn.Parameter(torch.zeros(act) - 0.5)

    def forward(self, x):
        h = self.body(x)
        return self.mu(h), self.log_std.exp(), self.v(h).squeeze(-1)

    def dist(self, x):
        mu, std, v = self(x)
        return torch.distributions.Normal(mu, std), v


def main():
    OUT = knob("OUT", "/data/shared/checkpoints/ppo", str)
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seed = knob("SEED", 20260627, int)
    torch.manual_seed(seed); np.random.seed(seed)

    n_envs = knob("NUM_ENVS", 256, int)
    rollout = knob("ROLLOUT_STEPS", 64, int)
    total = knob("TOTAL_STEPS", 5_000_000, int)
    epochs = knob("PPO_EPOCHS", 4, int)
    minibatches = knob("PPO_MINIBATCHES", 8, int)
    clip = knob("PPO_CLIP", 0.2)
    gamma = knob("PPO_GAMMA", 0.99)
    lam = knob("PPO_LAMBDA", 0.95)
    lr = knob("PPO_LR", 3e-4)
    ent = knob("PPO_ENT", 0.0)

    cfg = FreeFlyerConfig()
    env = FreeFlyerVecEnv(num_envs=n_envs, cfg=cfg, seed=seed)
    net = ActorCritic(env.obs_dim, env.act_dim).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    obs = torch.as_tensor(env.reset(), device=dev)
    ep_ret = np.zeros(n_envs, np.float32)
    log = {"updates": [], "step": [], "mean_return": [], "mean_coverage": [], "abort_rate": []}
    iters = total // (n_envs * rollout)
    t0 = time.time()
    for it in range(iters):
        O, A, LP, R, D, V = [], [], [], [], [], []
        ep_returns, ep_cov, ep_abort = [], [], []
        for _ in range(rollout):
            with torch.no_grad():
                dist, v = net.dist(obs)
                a = dist.sample()
                lp = dist.log_prob(a).sum(-1)
            no, r, d, info = env.step(a.cpu().numpy())
            O.append(obs); A.append(a); LP.append(lp); V.append(v)
            R.append(torch.as_tensor(r, device=dev)); D.append(torch.as_tensor(d.astype(np.float32), device=dev))
            ep_ret += r
            for i in np.where(d)[0]:
                ep_returns.append(ep_ret[i]); ep_ret[i] = 0.0
            ep_cov.append(info["coverage"].mean()); ep_abort.append((info["abort"] & ~info["timeout"]).mean())
            obs = torch.as_tensor(no, device=dev)
        with torch.no_grad():
            _, last_v = net.dist(obs)
        # GAE
        adv = torch.zeros(rollout, n_envs, device=dev); gae = torch.zeros(n_envs, device=dev)
        for t in reversed(range(rollout)):
            nextv = last_v if t == rollout - 1 else V[t + 1]
            delta = R[t] + gamma * nextv * (1 - D[t]) - V[t]
            gae = delta + gamma * lam * (1 - D[t]) * gae
            adv[t] = gae
        Ob = torch.stack(O).reshape(-1, env.obs_dim); Ac = torch.stack(A).reshape(-1, env.act_dim)
        LPb = torch.stack(LP).reshape(-1); Vb = torch.stack(V).reshape(-1)
        Advb = adv.reshape(-1); Retb = Advb + Vb
        Advb = (Advb - Advb.mean()) / (Advb.std() + 1e-8)
        N = Ob.shape[0]; mb = N // minibatches
        for _ in range(epochs):
            perm = torch.randperm(N, device=dev)
            for s in range(0, N, mb):
                j = perm[s:s + mb]
                dist, v = net.dist(Ob[j])
                lp = dist.log_prob(Ac[j]).sum(-1)
                ratio = (lp - LPb[j]).exp()
                s1 = ratio * Advb[j]; s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * Advb[j]
                pg = -torch.min(s1, s2).mean()
                vl = (v - Retb[j]).pow(2).mean()
                loss = pg + 0.5 * vl - ent * dist.entropy().sum(-1).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        if it % max(1, iters // 50) == 0 or it == iters - 1:
            mr = float(np.mean(ep_returns)) if ep_returns else 0.0
            mc = float(np.mean(ep_cov)); ab = float(np.mean(ep_abort))
            step = (it + 1) * n_envs * rollout
            log["updates"].append(it); log["step"].append(step)
            log["mean_return"].append(mr); log["mean_coverage"].append(mc); log["abort_rate"].append(ab)
            print(f"it {it}/{iters} step {step} return {mr:.2f} coverage {mc:.3f} "
                  f"abort {ab:.3f} ({step/max(1,time.time()-t0):.0f} sps)", flush=True)
            torch.save({"model": net.state_dict(), "obs_dim": env.obs_dim, "act_dim": env.act_dim,
                        "cfg": vars(cfg)}, os.path.join(OUT, "ppo_policy_checkpoint.pt"))
            json.dump(log, open(os.path.join(OUT, "ppo_train_log.json"), "w"), indent=2)
    print("PPO training complete; checkpoint at", os.path.join(OUT, "ppo_policy_checkpoint.pt"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
