"""Zero-shot perturbation families for the generalization eval.

These are NEW disturbance types the trigger/handoff were never developed or
calibrated against. The mechanics here only move scene bodies; they never
touch the AsyncMixVLA pipeline. Each family fires ONCE, spread over `duration`
consecutive control steps starting at `trigger_step`, exactly like the frozen
translation perturbation (schedule_from_manifest_event).

Free-joint qpos layout for a movable body: [x, y, z, qw, qx, qy, qz].
"""
import json

import numpy as np

from experiments.robot.libero.env_perturbations import (
    get_body_position, get_free_joint_info_for_body,
)


def _quat_mul(a, b):
    """Hamilton product, MuJoCo wxyz order."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def translate_body(env, body_name, delta_xyz, zero_velocity=True):
    info = get_free_joint_info_for_body(env, body_name)
    if info is None:
        return False
    sim = env.sim
    q = info["qpos_adr"]
    d = np.asarray(delta_xyz, float)
    sim.data.qpos[q + 0] += float(d[0])
    sim.data.qpos[q + 1] += float(d[1])
    if d.size > 2:
        sim.data.qpos[q + 2] += float(d[2])
    if zero_velocity:
        dof, dl = info["dof_adr"], info["dof_len"]
        sim.data.qvel[dof:dof + min(dl, 6)] = 0.0
    sim.forward()
    return True


def rotate_body_yaw(env, body_name, dtheta_rad, zero_velocity=True):
    info = get_free_joint_info_for_body(env, body_name)
    if info is None:
        return False
    sim = env.sim
    q = info["qpos_adr"]
    cur = np.array(sim.data.qpos[q + 3:q + 7], float)
    half = 0.5 * float(dtheta_rad)
    yaw = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])  # rotation about world +z
    new = _quat_mul(yaw, cur)
    n = np.linalg.norm(new)
    if n < 1e-9:
        return False
    sim.data.qpos[q + 3:q + 7] = new / n
    if zero_velocity:
        dof, dl = info["dof_adr"], info["dof_len"]
        sim.data.qvel[dof:dof + min(dl, 6)] = 0.0
    sim.forward()
    return True


class ZeroShotPerturbation:
    """pre_step_hook-compatible: build the hook via .make_hook(obs_holder).
    The hook takes the episode step index and applies at most one increment
    per step, only within [trigger_step, trigger_step + duration)."""

    def __init__(self, event, diagnostics=None):
        self.event = event
        self.mode = event["mode"]
        self.body = event["body_name"]
        self.trigger_step = int(event["trigger_step"])
        self.duration = int(event["duration"])
        self.diag = diagnostics if diagnostics is not None else {
            "body_name": self.body, "mode": self.mode, "fired": False,
            "applied_steps": [], "position_before": None, "position_after": None,
            "no_free_joint": False,
        }
        if self.mode in ("translate_xy",):
            dxy = np.asarray(event["delta_xy"], float)
            self._per_step = np.array([dxy[0], dxy[1], 0.0]) / self.duration
        elif self.mode == "slip_xyz":
            self._per_step = np.asarray(event["delta_xyz"], float) / self.duration
        elif self.mode == "rotate_yaw":
            self._per_step = float(event["dtheta_rad"]) / self.duration
        else:
            raise ValueError(f"unknown zero-shot perturbation mode {self.mode!r}")
        self._applied = set()
        self._first_done = False

    def make_hook(self, obs_holder):
        def hook(step_idx):
            if step_idx < self.trigger_step or step_idx >= self.trigger_step + self.duration:
                return
            if step_idx in self._applied:
                return
            self._applied.add(step_idx)
            try:
                before = get_body_position(self.env, self.body).tolist()
            except Exception:
                before = None
            if not self._first_done:
                self.diag["position_before"] = before
                self._first_done = True
            if self.mode == "rotate_yaw":
                ok = rotate_body_yaw(self.env, self.body, self._per_step)
            else:
                ok = translate_body(self.env, self.body, self._per_step)
            if not ok:
                self.diag["no_free_joint"] = True
                return
            self.diag["fired"] = True
            self.diag["applied_steps"].append(int(step_idx))
            try:
                self.diag["position_after"] = get_body_position(self.env, self.body).tolist()
            except Exception:
                pass
        self._hook = hook
        return hook

    # env is attached by setup (kept off __init__ so callers match setup_perturbation)
    env = None


def load_event(zeroshot_manifest_path, trial_id):
    with open(zeroshot_manifest_path) as f:
        m = json.load(f)
    entry = m[str(trial_id)]
    return entry


def setup_zeroshot_perturbation(zeroshot_manifest_path, trial_id, env, obs):
    """Mirror of run_asyncmixvla.setup_perturbation's return contract:
    (displacement_hook, diagnostics, obs_holder, manifest_event).

    manifest_event is returned as None on purpose: the frozen learned trigger's
    optional 'displacement_dot' feature takes a perturbation-direction hint from
    manifest_event['delta_xy']; a genuinely zero-shot deployment does not know
    the disturbance, so that feature stays at its clean-episode value (0.0) for
    every novel family. The seen translation baseline is run through the normal
    setup_perturbation path elsewhere and keeps its hint.
    """
    entry = load_event(zeroshot_manifest_path, trial_id)
    if not entry.get("feasible") or entry.get("event") is None:
        raise ValueError(f"zero-shot manifest trial {trial_id}: not feasible "
                         f"({entry.get('feasible_reason')})")
    pert = ZeroShotPerturbation(entry["event"])
    pert.env = env
    obs_holder = {"obs": obs}
    hook = pert.make_hook(obs_holder)
    return hook, pert.diag, obs_holder, None
