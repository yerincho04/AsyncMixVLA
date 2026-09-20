"""Causal vision conditioning used by the deployed async handoff."""
import numpy as np


def align_observation(mode, *, stale_obs, prepare_observation_fn, future_state=None, future_obs=None):
    """Prepare the trigger-time image and optional predicted future proprioception."""
    if mode not in ("stale", "action_consistency"):
        raise ValueError(f"unknown deployed vision_alignment mode: {mode!r}")
    prepared = dict(prepare_observation_fn(stale_obs))
    if future_state is not None:
        prepared["state"] = np.asarray(future_state, dtype=np.float32)
    if mode == "action_consistency":
        prepared["use_action_consistency"] = True
    return prepared
