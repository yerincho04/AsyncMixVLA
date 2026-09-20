"""Controller-aware future proprioception used at the deployed handoff."""

import numpy as np
from experiments.robot.libero.libero_utils import quat2axisangle
from robosuite.utils.transform_utils import axisangle2quat, quat_multiply

from asyncmixvla.runtime_io import process_action


def gripper_open_closed_bounds(env):
    robot = env.robots[0]
    ranges = np.array(
        [env.sim.model.jnt_range[env.sim.model.joint_name2id(name)] for name in robot.gripper_joints],
        dtype=np.float64,
    )
    lo, hi = ranges[:, 0], ranges[:, 1]
    closed = np.where(np.abs(lo) < np.abs(hi), lo, hi)
    opened = np.where(np.abs(lo) < np.abs(hi), hi, lo)
    return opened, closed


def roll_forward_proprio(env, base_obs, remaining_actions, open_bounds, closed_bounds):
    controller = env.robots[0].controller
    position = np.asarray(base_obs["robot0_eef_pos"], dtype=np.float64).copy()
    quaternion = np.asarray(base_obs["robot0_eef_quat"], dtype=np.float64).copy()
    for raw_action in remaining_actions:
        scaled = np.asarray(
            controller.scale_action(np.asarray(raw_action, dtype=np.float64)[:6]), dtype=np.float64
        )
        position += scaled[:3]
        quaternion = quat_multiply(axisangle2quat(scaled[3:6]), quaternion)
    gripper_command = process_action(remaining_actions[-1])[6]
    gripper = np.asarray(open_bounds if gripper_command < 0 else closed_bounds, dtype=np.float64)
    axis_angle = quat2axisangle(quaternion).astype(np.float64)
    state = np.concatenate([position, axis_angle, gripper]).astype(np.float32)
    return state, gripper
