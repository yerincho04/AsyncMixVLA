"""Environment accounting and helpers shared by deployed AsyncMixVLA runners."""

import hashlib
import numpy as np

from asyncmixvla.perturbation import apply_displacement_step
from asyncmixvla.runtime_io import CHUNK_SIZE, call_policy, prepare_observation, process_action


class EpisodeEnv:
    """Apply the disturbance before every action and account for every policy step."""

    def __init__(self, env, obs, horizon, hook=None, holder=None):
        self.base, self.obs, self.horizon = env, obs, horizon
        self.hook, self.holder = hook, holder
        self.steps = 0
        self.done = False
        self.policy = "adapter"
        self.oft_starts_at = None
        self.counts = {"adapter": 0, "oft": 0}
        self.actions = []
        self.digest = hashlib.sha256()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def step(self, action):
        if self.steps >= self.horizon or self.done:
            raise RuntimeError("step beyond horizon or terminal success")
        if self.hook is not None:
            self.holder["obs"] = self.obs
            self.hook(self.steps)
        policy = self.policy
        if self.oft_starts_at is not None and self.steps >= self.oft_starts_at:
            policy = "oft"
        array = np.asarray(action, dtype=np.float64)
        self.digest.update(array.tobytes())
        self.actions.append({"step": self.steps, "policy": policy, "action": array.tolist()})
        self.obs, reward, self.done, info = self.base.step(action)
        self.counts[policy] += 1
        self.steps += 1
        return self.obs, reward, self.done, info


def make_displacement_stepper(env, displacement_schedule, active_displacements,
                              source_initial_z_map, diagnostics, obs_holder, task_id):
    """Apply the recorded disturbance before every real environment step."""
    state = {"last_step": None}

    def hook(step_idx):
        if state["last_step"] is not None and step_idx > state["last_step"]:
            for body_name in list(active_displacements):
                active_displacements[body_name]["remaining"] -= 1
                if active_displacements[body_name]["remaining"] <= 0:
                    del active_displacements[body_name]
        apply_displacement_step(
            env, step_idx, displacement_schedule, active_displacements,
            source_initial_z_map, obs_holder["obs"], diagnostics, task_id=task_id,
        )
        state["last_step"] = step_idx

    return hook


def continue_with_oft_until_done(env, task_description, obs, done, t_start, max_steps):
    """Continue with fresh-observation OFT queries until success or timeout."""
    action_log = []
    t = t_start
    success = bool(done)
    success_step = t_start if done else None
    while not success and t < max_steps:
        actions, _ = call_policy("oft", prepare_observation(obs), task_description)
        for raw_action in actions[:CHUNK_SIZE]:
            if t >= max_steps:
                break
            obs, _, done, _ = env.step(process_action(raw_action).tolist())
            action_log.append({"t": t, "model": "oft"})
            t += 1
            if done:
                success, success_step = True, t
                break
    return {"success": success, "success_step": success_step, "final_step": t,
            "action_log": action_log}
