"""I/O and LIBERO environment helpers used by the deployed runtime."""

import os
import time

import msgpack
import msgpack_numpy as m
import numpy as np
import requests
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.robot_utils import normalize_gripper_action, invert_gripper_action

m.patch()

NUM_STEPS_WAIT = 10
CHUNK_SIZE = 8
MAX_STEPS = 520

ENDPOINTS = {
    "adapter": f"http://127.0.0.1:{os.environ.get('ADAPTER_PORT', '8002')}/act",
    "oft": f"http://127.0.0.1:{os.environ.get('OFT_PORT', '8001')}/act",
    "oft_action_consistency": (
        f"http://127.0.0.1:{os.environ.get('OFT_PORT', '8001')}/act_with_action_consistency"
    ),
}
SESSION = requests.Session()


def prepare_observation(obs):
    state = np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    ).astype(np.float32)
    return {
        "full_image": get_libero_image(obs),
        "wrist_image": get_libero_wrist_image(obs),
        "state": state,
    }


def process_action(action):
    action = normalize_gripper_action(action, binarize=True)
    return invert_gripper_action(action)


def _post(endpoint, payload, timeout):
    packed = msgpack.packb(payload, default=m.encode, use_bin_type=True)
    start = time.perf_counter()
    response = SESSION.post(
        endpoint, data=packed, headers={"Content-Type": "application/msgpack"}, timeout=timeout
    )
    response.raise_for_status()
    latency = time.perf_counter() - start
    output = msgpack.unpackb(response.content, object_hook=m.decode, raw=False)
    actions = np.asarray(output["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] != 7:
        raise ValueError(f"Expected [T,7], got {actions.shape}")
    return actions, latency


def call_policy(model, observation, task_description, timeout=120):
    payload = {
        "full_image": observation["full_image"],
        "wrist_image": observation["wrist_image"],
        "state": observation["state"],
        "task_description": task_description,
    }
    return _post(ENDPOINTS[model], payload, timeout)


def call_policy_action_consistency(observation, task_description, bridge_actions, timeout=120):
    payload = {
        "full_image": observation["full_image"],
        "wrist_image": observation["wrist_image"],
        "state": observation["state"],
        "task_description": task_description,
        "bridge_actions": np.asarray(bridge_actions, dtype=np.float64),
    }
    return _post(ENDPOINTS["oft_action_consistency"], payload, timeout)


def get_env_and_task(task_id):
    task_suite = benchmark.get_benchmark_dict()["libero_10"]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = get_libero_env(task, "openvla", resolution=256)
    return env, task_description, initial_states


def initialize_episode(env, initial_state):
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
        if done:
            raise RuntimeError("Episode succeeded during initialization wait")
    return obs


def warm_up(env, task_description, initial_state, count):
    obs = initialize_episode(env, initial_state)
    observation = prepare_observation(obs)
    for index in range(count):
        _, adapter_latency = call_policy("adapter", observation, task_description)
        _, oft_latency = call_policy("oft", observation, task_description)
        print(
            f"[warmup {index + 1}/{count}] adapter={adapter_latency:.4f}s oft={oft_latency:.4f}s",
            flush=True,
        )
