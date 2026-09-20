"""OFT driven asynchronously for the WHOLE episode -- the setting VLASH targets.

Why this exists
---------------
In the frozen AsyncMixVLA runtime, asynchrony happens exactly once per
episode (the Adapter->OFT handoff). After it, OFT runs synchronously via
continue_with_oft_until_done: it queries, the simulator waits, the chunk
executes. Staleness therefore exists at one moment, and a future-state
alignment can influence a single action. Even a perfect acausal future
observation (vision_alignment="oracle_context") did not beat stale there.

VLASH's gains come from a different regime: the policy runs asynchronously
at EVERY chunk boundary, so every chunk starts from a stale state and the
misalignment recurs hundreds of times per episode. This module reproduces
that regime so Naive vs VLASH-style alignment can be compared where the
alignment has something to correct.

Timing model (step-based, repeatable)
-------------------------------------
Each chunk has CHUNK_SIZE actions. With delay d (0 <= d < CHUNK_SIZE):

  1. execute the chunk's first CHUNK_SIZE - d actions normally;
  2. capture the observation, launch the next OFT query on a background
     thread, and execute the chunk's last d actions (the bridge) while it
     computes;
  3. start the new chunk at index 0.

The new chunk was conditioned on an observation exactly d steps old when its
first action executes. Staleness is defined by d, not by wall-clock speed, so
results do not depend on node load. If OFT is not ready when the bridge ends,
the robot waits (recorded as stall), exactly like the frozen bridge.

d = 0 degenerates to synchronous chunked OFT and must reproduce mode=oft_only.
The first chunk is always a synchronous query: the robot is at rest after the
settle steps, so there is nothing to be stale about.

Alignment (same modules as the frozen handoff, reused unmodified)
------------------------------------------------------------------
  state_alignment : "stale" (Naive) | "vlash_additive" (VLASH roll-forward)
                    | "vlash_gain_corrected" (Ours)
  vision_alignment: "stale" | "action_consistency" (Ours)

Bridge actions are OFT's own RAW chunk actions. roll_forward_proprio and the
action-consistency endpoint both expect raw [-1, 1] actions (they apply
process_action themselves), and process_action is applied exactly once, at
env.step. Note: the action-consistency residual was trained on Adapter bridge
actions; OFT actions share the action space but come from a different policy.
"""
import copy
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from run_seamless_handoff import gripper_open_closed_bounds
from run_test0_switch_timing import (
    CHUNK_SIZE,
    call_policy,
    call_policy_action_consistency,
    prepare_observation,
    process_action,
)
from asyncmixvla.state_alignment import align_state
from asyncmixvla.vision_alignment import align_observation

STATE_MODES = ("stale", "vlash_additive", "vlash_gain_corrected")
VISION_MODES = ("stale", "action_consistency")


def _query_raw_chunk(raw_obs, task_description, future_state, vision_alignment, bridge_raw):
    """Background-thread OFT query returning the RAW action chunk."""
    t0 = time.perf_counter()
    observation = align_observation(vision_alignment, stale_obs=raw_obs,
                                     prepare_observation_fn=prepare_observation, future_state=future_state)
    use_ac = observation.pop("use_action_consistency", False)
    if use_ac:
        actions, _ = call_policy_action_consistency(observation, task_description, bridge_raw)
    else:
        actions, _ = call_policy("oft", observation, task_description)
    return np.asarray(actions[:CHUNK_SIZE], dtype=np.float32), time.perf_counter() - t0


def run_oft_continuous_async(env, task_description, obs, *, delay, state_alignment, vision_alignment,
                             pre_step_hook=None, max_steps, start_step=0):
    if not 0 <= int(delay) < CHUNK_SIZE:
        raise ValueError(f"delay must be in [0, {CHUNK_SIZE - 1}], got {delay}")
    if state_alignment not in STATE_MODES:
        raise ValueError(f"unsupported state_alignment {state_alignment!r}")
    if vision_alignment not in VISION_MODES:
        raise ValueError(f"unsupported vision_alignment {vision_alignment!r}")
    delay = int(delay)
    open_bounds, closed_bounds = gripper_open_closed_bounds(env)

    step = start_step
    done = False
    stats = {"delay": delay, "n_chunks": 0, "n_async_boundaries": 0,
             "query_latency_s": [], "stall_s": []}

    def execute(raw_action):
        nonlocal obs, done, step
        if pre_step_hook is not None:
            pre_step_hook(step, obs)
        obs, _, done, _ = env.step(process_action(raw_action).tolist())
        step += 1

    chunk, _ = call_policy("oft", prepare_observation(obs), task_description)
    chunk = np.asarray(chunk[:CHUNK_SIZE], dtype=np.float32)
    stats["n_chunks"] += 1

    while not done and step < max_steps:
        for raw_action in chunk[:CHUNK_SIZE - delay]:
            if done or step >= max_steps:
                break
            execute(raw_action)
        if done or step >= max_steps:
            break

        if delay == 0:
            chunk, _ = call_policy("oft", prepare_observation(obs), task_description)
            chunk = np.asarray(chunk[:CHUNK_SIZE], dtype=np.float32)
            stats["n_chunks"] += 1
            continue

        bridge_raw = [np.asarray(a, dtype=np.float64) for a in chunk[CHUNK_SIZE - delay:]]
        raw_obs = copy.deepcopy(obs)
        future_state = None
        if state_alignment != "stale":
            future_state = align_state(state_alignment, current_obs=obs, bridge_actions=bridge_raw, env=env,
                                       open_bounds=open_bounds, closed_bounds=closed_bounds,
                                       prepare_observation_fn=prepare_observation)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_query_raw_chunk, raw_obs, task_description, future_state,
                                     vision_alignment, bridge_raw)
            for raw_action in bridge_raw:
                if done or step >= max_steps:
                    break
                execute(raw_action)
            t_bridge_end = time.perf_counter()
            new_chunk, latency = future.result()
            t_ready = time.perf_counter()
        stats["n_async_boundaries"] += 1
        stats["query_latency_s"].append(latency)
        stats["stall_s"].append(max(0.0, t_ready - t_bridge_end))
        chunk = new_chunk
        stats["n_chunks"] += 1

    stalls = stats["stall_s"]
    stats["median_stall_s"] = float(np.median(stalls)) if stalls else None
    stats["frac_boundaries_stalled"] = float(np.mean([s > 0.05 for s in stalls])) if stalls else None
    return {"done": bool(done), "final_step": step, "obs": obs, "stats": stats}
