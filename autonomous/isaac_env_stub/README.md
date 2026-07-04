# isaac_env_stub - JWST inspection environment package

The Isaac Lab task package for JWST inspection. Registers `Isaac-JWST-Inspect-v0`
with: 6-DoF zero-g rigid-body dynamics for the inspector, observation space
(RGB/depth features + inspector state + standoff/keep-out distances), action
space (cold-gas thrust + reaction-wheel torque), reward shaping (coverage,
standoff hold, safety, inspection completion), and termination (abort boundary,
keep-out violation, success).

Start from bounding-geometry JWST so this team is unblocked; swap in the Digital
Twin scene from `/data/shared/assets/` when available. Build the task as a proper
Python package (`pyproject.toml` + `isaac_env/__init__.py` registering the gym
id) following the Isaac Lab task examples in `${ENV_DIR}/IsaacLab` - confirm the
registration API against the pinned Isaac Lab version (do not guess task APIs).

Files to add here:
- `isaac_env/__init__.py` (gym registration)
- `isaac_env/jwst_inspect_env.py` (env + reward + termination)
- `isaac_env/mdp/` (observations, rewards, terminations, events)
- `pyproject.toml`
