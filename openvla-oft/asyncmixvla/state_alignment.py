"""Modular proprioception-conditioning for the async OFT request.

The controller-aware calibration found that uncorrected roll-forward
systematically overshoots by a controller-physics-explained, empirically
constant ~4x factor (OSC_POSE is a PD/impedance controller whose goal
resets every step -- it never fully converges within one policy action),
and that a single TRAIN-calibrated scalar gain per component (position/
orientation/gripper) removes nearly all of that error on DEV (96.7% of
handoff points beat stale, vs 0% for the naive additive model).

Gains are loaded from a calibration artifact (asyncmixvla/calibration/
vlash_gain_calibration.json), not hardcoded here, so they can be
recalibrated (e.g. if a different controller/env config is ever used)
without touching this module or callers.
"""
import json
import os

import numpy as np

from asyncmixvla.state_prediction import roll_forward_proprio

DEFAULT_CALIBRATION_PATH = os.environ.get(
    "VLASH_GAIN_CALIBRATION",
    os.path.join(os.path.dirname(__file__), "calibration", "vlash_gain_calibration.json"),
)

_CALIBRATION_CACHE = {}


def load_gain_calibration(path=None):
    path = path or DEFAULT_CALIBRATION_PATH
    if path not in _CALIBRATION_CACHE:
        with open(path) as f:
            _CALIBRATION_CACHE[path] = json.load(f)
    return _CALIBRATION_CACHE[path]


def align_state(mode, *, current_obs, bridge_actions, env, open_bounds, closed_bounds,
                 prepare_observation_fn, calibration=None):
    """Returns the 8-D proprioception state (pos[3]+axis-angle[3]+gripper[2])
    to use as the OFT observation's "state" field for the handoff.

    mode="stale": current_obs's own state, unchanged -- no roll-forward at
      all (the Naive Async baseline's proprio).
    mode="vlash_additive": roll_forward_proprio's naive prediction, called
      without gain correction.
    mode="vlash_gain_corrected": vlash_additive's prediction blended toward
      the current state via TRAIN-derived calibrated gains (see module
      docstring / calibration artifact). Position and orientation use their
      own fitted gains; gripper's fitted gain is ~0 (see calibration
      artifact's derivation note), so its prediction is effectively stale
      unless a future recalibration says otherwise -- read from the
      artifact, not assumed here.
    """
    if mode == "stale":
        return np.asarray(prepare_observation_fn(current_obs)["state"], dtype=np.float32)

    naive_state, _ = roll_forward_proprio(env, current_obs, bridge_actions, open_bounds, closed_bounds)
    if mode == "vlash_additive":
        return naive_state

    if mode == "vlash_gain_corrected":
        cal = calibration or load_gain_calibration()
        s_t = np.asarray(prepare_observation_fn(current_obs)["state"], dtype=np.float64)
        naive = np.asarray(naive_state, dtype=np.float64)
        pos = s_t[0:3] + cal["gain_pos"] * (naive[0:3] - s_t[0:3])
        ori = s_t[3:6] + cal["gain_ori"] * (naive[3:6] - s_t[3:6])
        grip = s_t[6:8] + cal["gain_gripper"] * (naive[6:8] - s_t[6:8])
        return np.concatenate([pos, ori, grip]).astype(np.float32)

    raise ValueError(f"unknown state_alignment mode: {mode!r}")
