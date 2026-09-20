"""Deployed asynchronous Adapter-to-OFT bridge and handoff.

The bridge rolls proprioception forward, optionally applies the causal
action-consistency visual residual, and overlaps the OFT request with the
remaining committed Adapter actions.

Locked definitions (spec section 8, do not change):
  T_predict = perf_counter() the instant the trigger fires (before any
              roll-forward/query work begins).
  T_switch  = perf_counter() the instant the final already-committed bridge
              action's env.step() returns.
  T_ready   = perf_counter() the instant the async OFT response is fully
              parsed and ready to execute.
  T_action  = perf_counter() the instant the first OFT action is actually
              issued to env.step() (== max(T_switch, T_ready) in practice).
  B         = T_switch - T_predict   (available bridge budget)
  C_async   = T_ready   - T_predict  (actual OFT async readiness latency)
  S         = T_switch  - T_ready    (slack; algebraically == B - C_async)
  L_handoff = T_action  - T_switch   (exposed robot wait -- THE robot-wait
              metric, not T_switch - T_predict)
  no_stall  = T_ready <= T_switch + control_dt (no missed next control
              deadline -- NOT "L_handoff == 0")
"""
import copy
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from experiments.robot.libero.env_perturbations import get_control_dt
from asyncmixvla.state_prediction import gripper_open_closed_bounds
from asyncmixvla.runtime_io import (
    CHUNK_SIZE,
    call_policy,
    call_policy_action_consistency,
    prepare_observation,
    process_action,
)
from asyncmixvla.state_alignment import align_state
from asyncmixvla.vision_alignment import align_observation


def oft_query_pipeline(raw_observation, task_description, origin, future_state=None,
                        vision_alignment="stale", bridge_actions_raw=None, future_obs=None):
    """Prepare the causal handoff context and query OFT on the background thread."""
    t_start = time.perf_counter()
    observation = align_observation(vision_alignment, stale_obs=raw_observation,
                                     prepare_observation_fn=prepare_observation,
                                     future_state=future_state, future_obs=future_obs)
    t_prep_end = time.perf_counter()
    use_action_consistency = observation.pop("use_action_consistency", False)
    if use_action_consistency:
        actions, http_latency = call_policy_action_consistency(observation, task_description, bridge_actions_raw)
    else:
        actions, http_latency = call_policy("oft", observation, task_description)
    actions = actions[:CHUNK_SIZE]
    first_action = process_action(actions[0])
    remaining_chunk_actions = [process_action(a) for a in actions[1:CHUNK_SIZE]]
    t_ready = time.perf_counter()
    return {
        "T_oft_start_abs": t_start, "T_prep_end_abs": t_prep_end, "T_ready_abs": t_ready,
        "dynamic_context_prep_s": t_prep_end - t_start, "oft_http_latency_s": http_latency,
        "oft_client_postprocess_s": t_ready - (t_prep_end + http_latency),
        "first_oft_action": first_action, "remaining_chunk_actions": remaining_chunk_actions,
        "T_oft_start_s": t_start - origin, "T_prep_end_s": t_prep_end - origin, "T_ready_s": t_ready - origin,
    }


def run_async_bridge(env, task_description, obs, bridge_actions_raw, origin, mechanism,
                      action_number_offset, open_bounds=None, closed_bounds=None, pre_step_hook=None,
                      state_alignment="vlash_gain_corrected", vision_alignment="stale",
                      adaptive_bridge=False, adaptive_max_extra_chunks=3):
    """mechanism in {"naive_async", "vlash_async"}. bridge_actions_raw: the
    EXACT already-generated Adapter actions from the current chunk that had
    not yet executed when the trigger fired -- passed in by the caller's
    state machine, never regenerated or altered here. Launches OFT
    asynchronously at T_predict=now, executes bridge_actions_raw for real
    while it computes, hands off at T_switch.

    The deployed method uses ``vlash_gain_corrected`` state alignment and
    ``action_consistency`` vision alignment. Baseline calls use ``stale``."""
    if mechanism not in ("naive_async", "vlash_async"):
        raise ValueError(f"run_async_bridge: unsupported mechanism {mechanism!r}")
    if len(bridge_actions_raw) == 0:
        raise ValueError("run_async_bridge: bridge_actions_raw must be non-empty")

    t_predict = time.perf_counter()
    raw_observation = copy.deepcopy(obs)

    future_state = None
    if mechanism == "vlash_async":
        if open_bounds is None or closed_bounds is None:
            open_bounds, closed_bounds = gripper_open_closed_bounds(env)
        # State-alignment compute happens here, between T_predict and the
        # executor.submit() call below -- its cost is on the critical path
        # (counted in C_async), matching the already-validated VLASH
        # pipeline structure (run_matched_evaluation_vlash.py). Include the
        # gain-corrected blend's (negligible) extra arithmetic in this same
        # window for a fair timing comparison -- no free preprocessing.
        future_state = align_state(
            state_alignment, current_obs=obs, bridge_actions=bridge_actions_raw, env=env,
            open_bounds=open_bounds, closed_bounds=closed_bounds, prepare_observation_fn=prepare_observation,
        )

    action_timeline = []
    n_extra_chunks = 0
    extra_bridge_actions = 0

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(oft_query_pipeline, raw_observation, task_description, origin,
                                  future_state, vision_alignment, bridge_actions_raw)

        t_switch = None
        for offset, raw_action in enumerate(bridge_actions_raw):
            action = process_action(raw_action)
            episode_step = action_number_offset + offset
            if pre_step_hook is not None:
                pre_step_hook(episode_step, obs)
            t_start = time.perf_counter()
            obs, _, done, _ = env.step(action.tolist())
            t_end = time.perf_counter()
            action_timeline.append({"episode_step": episode_step, "start_s": t_start - origin,
                                     "end_s": t_end - origin, "phase": "bridge",
                                     "raw_action": np.asarray(raw_action, dtype=np.float64).tolist()})
            if offset == len(bridge_actions_raw) - 1:
                t_switch = t_end  # locked boundary: last committed bridge action's completion
            if done:
                return {"valid": False, "reason": "adapter_succeeded_during_bridge",
                        "action_timeline": action_timeline, "obs": obs, "done": True}

        # --- adaptive bridge extension -------------------------------------
        # The default bridge is whatever remained of the Adapter's CURRENT
        # chunk when the trigger fired, i.e. 8 - chunk_position actions. That
        # budget B is an accident of chunk phase, not a function of how long
        # OFT actually needs: measured over 110 real switches, bridge length 1
        # gives B~105ms against C_async~231ms and stalls 92% of the time,
        # while length >=4 essentially never stalls. When enabled, this keeps
        # the Adapter driving (fetching further chunks) until OFT is actually
        # ready, so the robot never holds. The Adapter costs ~57ms / 48 GFLOPs
        # per chunk, negligible next to the 7.5B incoming policy.
        #
        # Opt-in (adaptive_bridge=False by default) so every previously
        # recorded result stays bit-reproducible. Note the state/vision
        # alignment for this handoff was computed at T_predict against the
        # ORIGINAL bridge_actions_raw; extending the bridge makes the real
        # T_switch later than that prediction assumed. Harmless for
        # stale/stale, a documented approximation for the predicted modes.
        if adaptive_bridge:
            while (not future.done()) and n_extra_chunks < adaptive_max_extra_chunks:
                extra_chunk, _ = call_policy("adapter", prepare_observation(obs), task_description)
                n_extra_chunks += 1
                for raw_action in extra_chunk[:CHUNK_SIZE]:
                    if future.done():
                        break
                    action = process_action(raw_action)
                    episode_step = action_number_offset + len(action_timeline)
                    if pre_step_hook is not None:
                        pre_step_hook(episode_step, obs)
                    t_start = time.perf_counter()
                    obs, _, done, _ = env.step(action.tolist())
                    t_end = time.perf_counter()
                    action_timeline.append({"episode_step": episode_step, "start_s": t_start - origin,
                                             "end_s": t_end - origin, "phase": "bridge_extended",
                                             "raw_action": np.asarray(raw_action, dtype=np.float64).tolist()})
                    t_switch = t_end   # locked boundary moves with the real last committed action
                    extra_bridge_actions += 1
                    if done:
                        return {"valid": False, "reason": "adapter_succeeded_during_bridge",
                                "action_timeline": action_timeline, "obs": obs, "done": True}
                if done:
                    break
        oft = future.result()

    t_ready = oft["T_ready_abs"]
    first_oft_action = oft.pop("first_oft_action")
    remaining_chunk_actions = oft.pop("remaining_chunk_actions")

    # Handoff: if OFT wasn't ready by T_switch, this env.step() call itself
    # is what we're timing -- future.result() above already blocked until
    # T_ready, so by the time we get here OFT is guaranteed ready; T_action
    # is simply "now" (== max(T_switch, T_ready) in wall-clock terms).
    t_action = time.perf_counter()
    obs, _, first_action_done, _ = env.step(first_oft_action.tolist())
    t_action_complete = time.perf_counter()

    control_dt = get_control_dt(env)
    B = t_switch - t_predict
    C_async = t_ready - t_predict
    S = t_switch - t_ready
    L_handoff = t_action - t_switch
    first_control_deadline = t_switch + control_dt
    missed_ticks = max(0, math.ceil(max(0.0, t_ready - first_control_deadline) / control_dt))
    no_stall = bool(t_ready <= first_control_deadline)

    return {
        "valid": True, "mechanism": mechanism, "state_alignment": state_alignment if mechanism == "vlash_async" else None,
        "adaptive_bridge": bool(adaptive_bridge), "adaptive_extra_chunks": n_extra_chunks,
        "adaptive_extra_bridge_actions": extra_bridge_actions,
        "vision_alignment": vision_alignment,
        "T_predict_s": t_predict - origin, "T_switch_s": t_switch - origin,
        "T_ready_s": t_ready - origin, "T_action_s": t_action - origin,
        "T_action_complete_s": t_action_complete - origin,
        "B_s": B, "C_async_s": C_async, "S_s": S, "L_handoff_s": L_handoff,
        "no_stall": no_stall, "missed_control_ticks": missed_ticks,
        "nominal_control_dt_s": control_dt,
        "oft_http_latency_s": oft["oft_http_latency_s"],
        "dynamic_context_prep_s": oft["dynamic_context_prep_s"],
        "oft_client_postprocess_s": oft["oft_client_postprocess_s"],
        "T_oft_start_s": oft["T_oft_start_s"], "T_prep_end_s": oft["T_prep_end_s"],
        "action_timeline": action_timeline,
        "vlash_future_state": future_state.tolist() if future_state is not None else None,
        "vlash_prediction_bridge_actions": [np.asarray(a, dtype=np.float64).tolist() for a in bridge_actions_raw]
                                            if mechanism == "vlash_async" else None,
        "first_oft_action_success": bool(first_action_done),
        "obs": obs, "done": bool(first_action_done),
        "remaining_oft_chunk_actions": [a.tolist() for a in remaining_chunk_actions],
        # Additive (benchmark instrumentation): the processed OFT action chunk
        # produced at the handoff -- first_oft_action is the action actually
        # executed at T_action; the full chunk is first + the 7 already-fetched
        # remaining actions (post process_action). No behavior change.
        "first_oft_action": first_oft_action.tolist(),
        "oft_chunk_actions": [first_oft_action.tolist()] + [a.tolist() for a in remaining_chunk_actions],
    }


def run_sync_handoff(env, task_description, obs, bridge_actions_raw, origin, action_number_offset,
                      pre_step_hook=None):
    """mechanism="sync_full_oft": finish the rest of the current chunk as
    plain Adapter (no OFT work starts until the chunk is done), THEN issue
    one blocking OFT query. T_predict is still the trigger-fire timestamp;
    T_ready/T_action coincide since the query is synchronous."""
    if len(bridge_actions_raw) == 0:
        raise ValueError("run_sync_handoff: bridge_actions_raw must be non-empty")

    t_predict = time.perf_counter()
    action_timeline = []
    t_switch = None
    for offset, raw_action in enumerate(bridge_actions_raw):
        action = process_action(raw_action)
        episode_step = action_number_offset + offset
        if pre_step_hook is not None:
            pre_step_hook(episode_step, obs)
        t_start = time.perf_counter()
        obs, _, done, _ = env.step(action.tolist())
        t_end = time.perf_counter()
        action_timeline.append({"episode_step": episode_step, "start_s": t_start - origin,
                                 "end_s": t_end - origin, "phase": "bridge_sync_no_oft_work",
                                 "raw_action": np.asarray(raw_action, dtype=np.float64).tolist()})
        if offset == len(bridge_actions_raw) - 1:
            t_switch = t_end
        if done:
            return {"valid": False, "reason": "adapter_succeeded_during_bridge",
                    "action_timeline": action_timeline, "obs": obs, "done": True}

    raw_observation = copy.deepcopy(obs)
    oft = oft_query_pipeline(raw_observation, task_description, origin, future_state=None)
    t_ready = oft["T_ready_abs"]
    first_oft_action = oft.pop("first_oft_action")
    remaining_chunk_actions = oft.pop("remaining_chunk_actions")

    t_action = time.perf_counter()
    obs, _, first_action_done, _ = env.step(first_oft_action.tolist())
    t_action_complete = time.perf_counter()

    control_dt = get_control_dt(env)
    B = t_switch - t_predict
    C_async = t_ready - t_predict  # not truly "async" here, but same formula: time from trigger to ready
    S = t_switch - t_ready
    L_handoff = t_action - t_switch
    first_control_deadline = t_switch + control_dt
    missed_ticks = max(0, math.ceil(max(0.0, t_ready - first_control_deadline) / control_dt))
    no_stall = bool(t_ready <= first_control_deadline)

    return {
        "valid": True, "mechanism": "sync_full_oft",
        "T_predict_s": t_predict - origin, "T_switch_s": t_switch - origin,
        "T_ready_s": t_ready - origin, "T_action_s": t_action - origin,
        "T_action_complete_s": t_action_complete - origin,
        "B_s": B, "C_async_s": C_async, "S_s": S, "L_handoff_s": L_handoff,
        "no_stall": no_stall, "missed_control_ticks": missed_ticks,
        "nominal_control_dt_s": control_dt,
        "oft_http_latency_s": oft["oft_http_latency_s"],
        "dynamic_context_prep_s": oft["dynamic_context_prep_s"],
        "oft_client_postprocess_s": oft["oft_client_postprocess_s"],
        "T_oft_start_s": oft["T_oft_start_s"], "T_prep_end_s": oft["T_prep_end_s"],
        "action_timeline": action_timeline,
        "vlash_future_state": None, "vlash_prediction_bridge_actions": None,
        "first_oft_action_success": bool(first_action_done),
        "obs": obs, "done": bool(first_action_done),
        "remaining_oft_chunk_actions": [a.tolist() for a in remaining_chunk_actions],
        "first_oft_action": first_oft_action.tolist(),
        "oft_chunk_actions": [first_oft_action.tolist()] + [a.tolist() for a in remaining_chunk_actions],
    }
