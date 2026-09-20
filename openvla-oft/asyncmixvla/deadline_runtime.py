"""Opt-in real-time deadline model for the step-based LIBERO runtime.

LIBERO normally freezes simulation while a policy request is blocking.  That
makes a synchronous handoff's wall-clock pause visible in latency metrics but
physically free in task success.  ``DeadlineAwareEnv`` preserves the existing
bridge implementation and inserts the control ticks that elapsed between the
last Adapter bridge action and the first OFT action.

The wrapper is experimental and disabled unless a caller explicitly creates
it.  Every inserted tick still routes through ``EpisodeEnv.step`` so the task
perturbation hook, horizon accounting, action digest, and success predicate all
advance exactly as they do for policy actions.
"""

import math
import time

import numpy as np

from experiments.robot.libero.env_perturbations import get_control_dt


HOLD_MODES = ("zero_arm_hold_gripper", "repeat_last")


def missed_control_ticks(wait_s, control_dt_s):
    """Ticks missed after allowing one normal control period for the response."""
    wait_s = max(0.0, float(wait_s))
    control_dt_s = float(control_dt_s)
    if control_dt_s <= 0:
        raise ValueError("control_dt_s must be positive")
    return max(0, int(math.ceil(max(0.0, wait_s - control_dt_s) / control_dt_s)))


def hold_action(last_action, mode):
    action = np.asarray(last_action, dtype=np.float64).copy()
    if action.shape != (7,):
        raise ValueError(f"expected a processed 7-D action, got {action.shape}")
    if mode == "zero_arm_hold_gripper":
        action[:6] = 0.0
    elif mode != "repeat_last":
        raise ValueError(f"unknown deadline hold mode: {mode!r}")
    return action


class DeadlineAwareEnv:
    """Proxy that materializes missed control ticks immediately before OFT takeover."""

    def __init__(self, wrapped, mode="zero_arm_hold_gripper", latency_override_s=None):
        if mode not in HOLD_MODES:
            raise ValueError(f"mode must be one of {HOLD_MODES}")
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_mode", mode)
        object.__setattr__(self, "_latency_override_s", latency_override_s)
        object.__setattr__(self, "_last_step_end", None)
        object.__setattr__(self, "_last_action", None)
        object.__setattr__(self, "_injected_for_takeover", False)
        object.__setattr__(self, "deadline_events", [])

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def __setattr__(self, name, value):
        if name.startswith("_") or name == "deadline_events":
            object.__setattr__(self, name, value)
        else:
            setattr(self._wrapped, name, value)
            if name == "oft_starts_at" and value is not None:
                object.__setattr__(self, "_injected_for_takeover", False)

    def step(self, action):
        now = time.perf_counter()
        takeover_due = (
            self._wrapped.oft_starts_at is not None
            and self._wrapped.steps >= self._wrapped.oft_starts_at
            and not self._injected_for_takeover
            and self._last_step_end is not None
            and self._last_action is not None
        )
        if takeover_due:
            observed_wait = max(0.0, now - self._last_step_end)
            modeled_wait = (
                observed_wait if self._latency_override_s is None else float(self._latency_override_s)
            )
            control_dt = get_control_dt(self._wrapped)
            requested = missed_control_ticks(modeled_wait, control_dt)
            hold = hold_action(self._last_action, self._mode)
            executed = 0
            terminal_result = None
            start_step = int(self._wrapped.steps)
            for _ in range(requested):
                if self._wrapped.done or self._wrapped.steps >= self._wrapped.horizon:
                    break
                terminal_result = self._wrapped.step(hold.tolist())
                executed += 1
                if terminal_result[2]:
                    break
            self.deadline_events.append({
                "takeover_step": start_step,
                "observed_wait_s": observed_wait,
                "modeled_wait_s": modeled_wait,
                "control_dt_s": control_dt,
                "requested_hold_ticks": requested,
                "executed_hold_ticks": executed,
                "hold_mode": self._mode,
                "hold_action": hold.tolist(),
                "success_during_hold": bool(terminal_result and terminal_result[2]),
                "horizon_exhausted_during_hold": bool(
                    not self._wrapped.done and self._wrapped.steps >= self._wrapped.horizon
                ),
            })
            object.__setattr__(self, "_injected_for_takeover", True)
            if terminal_result is not None and terminal_result[2]:
                object.__setattr__(self, "_last_step_end", time.perf_counter())
                object.__setattr__(self, "_last_action", hold)
                return terminal_result
            if self._wrapped.steps >= self._wrapped.horizon:
                # No control slot remains for the requested OFT action.  Return
                # the current observation without stepping past the locked
                # evaluation horizon; the outer loop will terminate on steps.
                return self._wrapped.obs, 0.0, False, {
                    "deadline_horizon_exhausted": True,
                    "oft_action_executed": False,
                }

        result = self._wrapped.step(action)
        object.__setattr__(self, "_last_step_end", time.perf_counter())
        object.__setattr__(self, "_last_action", np.asarray(action, dtype=np.float64).copy())
        return result
