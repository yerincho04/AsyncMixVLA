"""Causal, observable-only features for an Adapter switching detector.

One update describes the observation *before* executed action ``episode_step``.
``adapter_action`` is the already processed Adapter action about to execute; the previous
record's action explains the newly measured state change. Nothing here processes
or changes a gripper command. Call ``reset()`` at every episode boundary.

The caller passes policy-oriented RGB arrays (``prepare_observation`` already
rotates LIBERO images by 180 degrees) and its measured eight-dimensional state:
eef xyz, eef axis-angle xyz in radians, and two gripper joint positions. Camera
images may be downsampled to 64x64 by the caller. This module does not receive an
environment, task identifier, perturbation metadata, or outcome information.

All features are available at decision time. Episode step is used only to enforce
alignment, never as a model feature. A schema change requires a new version and a
refit; model files must persist and check ``feature_names`` and ``schema_version``.
"""

from collections import deque
from dataclasses import dataclass
from numbers import Integral

import numpy as np


SCHEMA_VERSION = "observable_v1"
CAMERAS = ("front", "wrist")
CAMERA_LAGS = (1, 4, 16)
ACTION_WINDOWS = (5, 16)
STATE_NAMES = ("eef_x", "eef_y", "eef_z", "eef_ax", "eef_ay", "eef_az", "gripper_left", "gripper_right")
ACTION_NAMES = ("dx", "dy", "dz", "dax", "day", "daz", "gripper")


@dataclass(frozen=True, slots=True)
class ObservableInput:
    """Explicit deployment inputs, with no optional hidden state channels.

    RGB arrays must be uint8 HxWx3; height and width must be multiples of 16.
    ``chunk_position`` is the executed Adapter chunk position, an integer 0..7.
    The first observation after warm-up has ``episode_step=0``. It does not
    describe a warm-up or idle action, and must precede the first evaluated action.
    """

    full_image: np.ndarray
    wrist_image: np.ndarray
    robot_state: np.ndarray
    adapter_action: np.ndarray
    episode_step: int
    chunk_position: int


def _feature_names():
    names = [f"state.{name}" for name in STATE_NAMES]
    for prefix in ("action", "previous_action", "action_delta"):
        names.extend(f"{prefix}.{name}" for name in ACTION_NAMES)
    names.extend(f"eef_delta.{axis}" for axis in "xyz")
    names.extend(f"rotation_delta.{axis}" for axis in "xyz")
    names.extend(("gripper_delta.left", "gripper_delta.right"))
    names.extend((
        "has_previous", "chunk_position_fraction", "eef_motion_norm",
        "rotation_motion_angle", "gripper_motion_norm", "previous_translation_command_norm",
        "previous_rotation_command_norm", "translation_command_alignment",
        "rotation_command_alignment", "translation_motion_per_command",
        "rotation_motion_per_command",
    ))
    for window in ACTION_WINDOWS:
        prefix = f"history{window}"
        for statistic in ("action_mean", "action_std"):
            names.extend(f"{prefix}.{statistic}.{name}" for name in ACTION_NAMES)
        names.extend(f"{prefix}.{name}" for name in (
            "action_change_mean", "gripper_flip_fraction", "eef_displacement",
            "eef_path_length", "rotation_path_length", "gripper_path_length", "coverage",
        ))
    names.extend(f"camera.lag{lag}.available" for lag in CAMERA_LAGS)
    for camera in CAMERAS:
        names.extend((f"{camera}.luma_std", f"{camera}.edge_energy"))
        for lag in CAMERA_LAGS:
            prefix = f"{camera}.lag{lag}"
            names.extend(f"{prefix}.residual_grid.r{row}c{col}" for row in range(4) for col in range(4))
            names.extend(f"{prefix}.median_shift.{channel}" for channel in "rgb")
            names.extend(f"{prefix}.{name}" for name in (
                "rgb_abs_mean", "luma_abs_mean", "residual_mean", "residual_grid_max",
            ))
    return tuple(names)


FEATURE_NAMES = _feature_names()
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def _vector(value, size, name):
    array = np.asarray(value)
    if array.shape != (size,) or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a numeric vector of shape ({size},)")
    array = array.astype(np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _camera(value, name):
    image = np.asarray(value)
    if image.dtype != np.uint8:
        raise ValueError(f"{name} must have dtype uint8 and RGB values in [0,255]")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must have shape (height,width,3)")
    height, width, _ = image.shape
    if height < 16 or width < 16 or height % 16 or width % 16:
        raise ValueError(f"{name} height and width must be positive multiples of 16")
    # Area pooling preserves localized motion while bounding CPU and history size.
    return image.reshape(16, height // 16, 16, width // 16, 3).mean(axis=(1, 3)) / 255.0


def _quaternion(axis_angle):
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-12:
        return np.concatenate((0.5 * axis_angle, [1.0]))
    return np.concatenate((axis_angle * (np.sin(angle / 2) / angle), [np.cos(angle / 2)]))


def _rotation_delta(previous, current):
    """Shortest relative rotation vector, avoiding axis-angle branch artifacts."""
    prev = _quaternion(previous)
    curr = _quaternion(current)
    xyz = prev[3] * curr[:3] - curr[3] * prev[:3] - np.cross(curr[:3], prev[:3])
    scalar = float(curr[3] * prev[3] + np.dot(curr[:3], prev[:3]))
    if scalar < 0:
        xyz, scalar = -xyz, -scalar
    norm = np.linalg.norm(xyz)
    if norm < 1e-12:
        return 2.0 * xyz
    return xyz * (2.0 * np.arctan2(norm, scalar) / norm)


def _alignment(motion, command):
    scale = np.linalg.norm(motion) * np.linalg.norm(command)
    return float(np.dot(motion, command) / scale) if scale > 1e-12 else 0.0


def _motion_per_command(motion, command):
    # This is an observable ratio, not an OSC prediction: no control gain is
    # assumed. A denominator floor bounds near-zero commands without hiding them.
    return float(np.linalg.norm(motion) / max(float(np.linalg.norm(command)), 1e-3))


class ObservableFeatures:
    """Stateful deterministic extractor with a fixed, versioned float32 schema."""

    schema_version = SCHEMA_VERSION
    feature_names = FEATURE_NAMES
    num_features = len(FEATURE_NAMES)

    def __init__(self):
        self.reset()

    def reset(self):
        self._history = deque(maxlen=max(CAMERA_LAGS))
        self._next_step = 0

    def update(self, record: ObservableInput) -> np.ndarray:
        if not isinstance(record, ObservableInput):
            raise TypeError("record must be an ObservableInput")
        if isinstance(record.episode_step, (bool, np.bool_)) or not isinstance(record.episode_step, Integral):
            raise ValueError("episode_step must be an integer")
        if record.episode_step != self._next_step:
            raise ValueError(f"Expected episode_step {self._next_step}, received {record.episode_step}; reset at episode boundaries")
        if isinstance(record.chunk_position, (bool, np.bool_)) or not isinstance(record.chunk_position, Integral):
            raise ValueError("chunk_position must be an integer in [0,7]")
        if not 0 <= record.chunk_position < 8:
            raise ValueError("chunk_position must be an integer in [0,7]")
        state = _vector(record.robot_state, 8, "state")
        action = _vector(record.adapter_action, 7, "action")
        cameras = (_camera(record.full_image, "full_image"), _camera(record.wrist_image, "wrist_image"))
        current = (state, action, cameras)
        previous = self._history[-1] if self._history else None
        previous_action = previous[1] if previous is not None else np.zeros(7)
        eef_delta = state[:3] - previous[0][:3] if previous is not None else np.zeros(3)
        rotation_delta = _rotation_delta(previous[0][3:6], state[3:6]) if previous is not None else np.zeros(3)
        gripper_delta = state[6:] - previous[0][6:] if previous is not None else np.zeros(2)
        action_delta = action - previous_action if previous is not None else np.zeros(7)
        values = [*state, *action, *previous_action, *action_delta, *eef_delta, *rotation_delta, *gripper_delta]
        values.extend((
            float(previous is not None), record.chunk_position / 7.0,
            np.linalg.norm(eef_delta), np.linalg.norm(rotation_delta), np.linalg.norm(gripper_delta),
            np.linalg.norm(previous_action[:3]), np.linalg.norm(previous_action[3:6]),
            _alignment(eef_delta, previous_action[:3]), _alignment(rotation_delta, previous_action[3:6]),
            _motion_per_command(eef_delta, previous_action[:3]), _motion_per_command(rotation_delta, previous_action[3:6]),
        ))
        history = [*self._history, current]
        for window in ACTION_WINDOWS:
            recent = history[-window:]
            states = np.stack([item[0] for item in recent])
            actions = np.stack([item[1] for item in recent])
            values.extend(actions.mean(axis=0))
            values.extend(actions.std(axis=0))
            changes = np.diff(actions, axis=0)
            values.extend((
                np.linalg.norm(changes, axis=1).mean() if len(changes) else 0.0,
                np.mean(actions[1:, -1] * actions[:-1, -1] < 0) if len(changes) else 0.0,
                np.linalg.norm(states[-1, :3] - states[0, :3]),
                np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1).sum(),
                sum(np.linalg.norm(_rotation_delta(a[3:6], b[3:6])) for a, b in zip(states[:-1], states[1:])),
                np.linalg.norm(np.diff(states[:, 6:], axis=0), axis=1).sum(),
                len(recent) / window,
            ))
        values.extend(float(len(self._history) >= lag) for lag in CAMERA_LAGS)
        for camera_index, camera in enumerate(cameras):
            luma = camera @ _LUMA
            values.extend((luma.std(), (np.abs(np.diff(luma, axis=0)).mean() + np.abs(np.diff(luma, axis=1)).mean()) / 2))
            for lag in CAMERA_LAGS:
                if len(self._history) < lag:
                    values.extend([0.0] * 23)
                    continue
                delta = camera - self._history[-lag][2][camera_index]
                median_shift = np.median(delta, axis=(0, 1))
                residual = np.abs(delta - median_shift).mean(axis=2)
                grid = residual.reshape(4, 4, 4, 4).mean(axis=(1, 3))
                values.extend(grid.ravel())
                values.extend(median_shift)
                values.extend((np.abs(delta).mean(), np.abs(delta @ _LUMA).mean(), residual.mean(), grid.max()))
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (self.num_features,) or not np.isfinite(result).all():
            raise ValueError("Feature vector has an invalid shape or non-finite value")
        # Commit only once all validation succeeds. External mutation of the input
        # cannot change history: vectors are copies and pooled images are new arrays.
        self._history.append(current)
        self._next_step += 1
        return result


def camera64(rgb):
    """Downsample an orientation-correct RGB frame exactly as during trigger fitting."""
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected orientation-correct RGB uint8 image")
    yy = np.minimum(((np.arange(64) + 0.5) * image.shape[0] / 64).astype(int), image.shape[0] - 1)
    xx = np.minimum(((np.arange(64) + 0.5) * image.shape[1] / 64).astype(int), image.shape[1] - 1)
    return image[yy[:, None], xx[None, :]].copy()


def observable(prepared, action, step, chunk_position):
    return ObservableInput(
        full_image=camera64(prepared["full_image"]),
        wrist_image=camera64(prepared["wrist_image"]),
        robot_state=np.asarray(prepared["state"], dtype=np.float32).copy(),
        adapter_action=np.asarray(action, dtype=np.float32).copy(),
        episode_step=step,
        chunk_position=chunk_position,
    )
