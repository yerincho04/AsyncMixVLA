"""Measure the Sync and Async-naive MixVLA handoff latency protocol.

This implements the timestamp definitions in
``Async_MixVLA_Latency_Formulation_Report.pdf``.  Async-naive deliberately
uses the observation available at T_predict; it is not presented as a
future-state method.  A future-prediction condition must be added only when a
real predictor is available.
"""

import argparse
import json
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from experiments.robot.libero.env_perturbations import get_control_dt
from experiments.robot.libero.libero_utils import get_libero_dummy_action
from run_test0_switch_timing import (
    CHUNK_SIZE,
    NUM_STEPS_WAIT,
    call_policy,
    get_env_and_task,
    prepare_observation,
    process_action,
)


def percentile(values, q):
    values = sorted(float(v) for v in values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def summarize(values):
    values = [float(v) for v in values]
    return {
        "n": len(values),
        "median_s": statistics.median(values) if values else None,
        "p95_s": percentile(values, 0.95),
        "mean_s": statistics.mean(values) if values else None,
        "min_s": min(values) if values else None,
        "max_s": max(values) if values else None,
    }


def initialize_episode(env, initial_state):
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
        if done:
            raise RuntimeError("Episode succeeded during initialization wait")
    return obs


def query_adapter_chunk(obs, task_description):
    actions, latency = call_policy("adapter", prepare_observation(obs), task_description)
    actions = actions[:CHUNK_SIZE]
    if len(actions) != CHUNK_SIZE:
        raise ValueError(f"Adapter returned {len(actions)} actions; expected {CHUNK_SIZE}")
    return actions, latency


def execute_adapter_chunk(env, obs, actions):
    step_durations = []
    for raw_action in actions:
        action = process_action(raw_action)
        t0 = time.perf_counter()
        obs, _, done, _ = env.step(action.tolist())
        step_durations.append(time.perf_counter() - t0)
        if done:
            return obs, True, step_durations
    return obs, False, step_durations


def oft_pipeline(observation, task_description, origin):
    """Run and timestamp the complete client-side OFT critical path."""
    t_start = time.perf_counter()
    # The observation is already a detached numpy payload.  Keep this boundary
    # explicit so a future predictor/context builder can be inserted here.
    t_prep_end = time.perf_counter()
    actions, http_latency = call_policy("oft", observation, task_description)
    first_action = process_action(actions[0])
    t_ready = time.perf_counter()
    return {
        "T_oft_start_s": t_start - origin,
        "T_prep_end_s": t_prep_end - origin,
        "T_ready_s": t_ready - origin,
        "dynamic_prep_s": t_prep_end - t_start,
        "oft_http_latency_s": http_latency,
        "oft_client_non_http_overhead_s": t_ready - (t_prep_end + http_latency),
        "first_oft_action": first_action,
    }


def run_trial(env, task_description, initial_state, condition):
    """Run one matched trial with a complete final Adapter chunk."""
    obs = initialize_episode(env, initial_state)
    adapter_actions, adapter_http_latency = query_adapter_chunk(obs, task_description)
    origin = time.perf_counter()
    control_dt = get_control_dt(env)

    if condition == "async_naive":
        # The deterministic artificial trigger is immediately before the
        # final committed chunk starts.  All eight actions remain.
        stale_observation = prepare_observation(obs)
        t_predict = time.perf_counter()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(oft_pipeline, stale_observation, task_description, origin)
            obs, done, adapter_step_durations = execute_adapter_chunk(env, obs, adapter_actions)
            t_switch = time.perf_counter()
            if done:
                return {"valid": False, "reason": "adapter_succeeded_before_handoff"}
            oft = future.result()
    elif condition == "sync":
        # T_predict is still logged at the same deterministic point, but Sync
        # intentionally performs no OFT-side dynamic work before T_switch.
        t_predict = time.perf_counter()
        obs, done, adapter_step_durations = execute_adapter_chunk(env, obs, adapter_actions)
        t_switch = time.perf_counter()
        if done:
            return {"valid": False, "reason": "adapter_succeeded_before_handoff"}
        oft = oft_pipeline(prepare_observation(obs), task_description, origin)
    else:
        raise ValueError(condition)

    t_ready_absolute = origin + oft["T_ready_s"]
    first_oft_action = oft.pop("first_oft_action")
    t_action = time.perf_counter()
    _, _, done, _ = env.step(first_oft_action.tolist())
    t_action_complete = time.perf_counter()

    bridge_budget = t_switch - t_predict
    critical_path = t_ready_absolute - t_predict
    slack = bridge_budget - critical_path
    handoff_latency = t_action - t_switch
    first_control_deadline = t_switch + control_dt
    missed_ticks = max(0, math.ceil(max(0.0, t_ready_absolute - first_control_deadline) / control_dt))

    return {
        "valid": True,
        "condition": condition,
        "timestamp_clock": "time.perf_counter",
        "T_predict_s": t_predict - origin,
        "T_switch_s": t_switch - origin,
        **oft,
        "T_action_s": t_action - origin,
        "T_action_complete_s": t_action_complete - origin,
        "N_remaining": CHUNK_SIZE,
        "B_s": bridge_budget,
        "C_async_s": critical_path if condition == "async_naive" else None,
        "S_s": slack if condition == "async_naive" else None,
        "L_handoff_s": handoff_latency,
        "ready_by_switch": bool(t_ready_absolute <= t_switch),
        "zero_stall": bool(t_ready_absolute <= first_control_deadline),
        "missed_control_ticks": missed_ticks,
        "nominal_control_dt_s": control_dt,
        "adapter_http_latency_s": adapter_http_latency,
        "adapter_step_durations_s": adapter_step_durations,
        "adapter_chunk_wall_s": sum(adapter_step_durations),
        "env_step_first_oft_action_s": t_action_complete - t_action,
        "first_oft_action_success": bool(done),
    }


def warm_up(env, task_description, initial_state, count):
    obs = initialize_episode(env, initial_state)
    observation = prepare_observation(obs)
    for i in range(count):
        _, adapter_latency = call_policy("adapter", observation, task_description)
        _, oft_latency = call_policy("oft", observation, task_description)
        print(
            f"[warmup {i + 1}/{count}] adapter={adapter_latency:.4f}s "
            f"oft={oft_latency:.4f}s",
            flush=True,
        )


def aggregate(records):
    by_condition = {}
    for condition in ("sync", "async_naive"):
        valid = [r for r in records if r.get("valid") and r["condition"] == condition]
        by_condition[condition] = {
            "valid_trials": len(valid),
            "L_handoff": summarize(r["L_handoff_s"] for r in valid),
            "oft_http_latency": summarize(r["oft_http_latency_s"] for r in valid),
            "missed_control_ticks": summarize(r["missed_control_ticks"] for r in valid),
            "zero_stall_rate": (
                sum(r["zero_stall"] for r in valid) / len(valid) if valid else None
            ),
        }
        if condition == "async_naive":
            by_condition[condition].update({
                "B": summarize(r["B_s"] for r in valid),
                "C_async": summarize(r["C_async_s"] for r in valid),
                "S": summarize(r["S_s"] for r in valid),
                "ready_by_switch_rate": (
                    sum(r["ready_by_switch"] for r in valid) / len(valid) if valid else None
                ),
            })

    sync = by_condition["sync"]
    async_naive = by_condition["async_naive"]
    sync_med = sync["L_handoff"]["median_s"]
    async_med = async_naive["L_handoff"]["median_s"]
    iso_oft = sync["oft_http_latency"]["median_s"]
    contended_oft = async_naive["oft_http_latency"]["median_s"]
    comparisons = {
        "hidden_fraction_median": (
            1.0 - async_med / sync_med if sync_med and async_med is not None else None
        ),
        "oft_contention_slowdown_median": (
            contended_oft / iso_oft if iso_oft and contended_oft is not None else None
        ),
    }
    return {"conditions": by_condition, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.environ.get("ASYNC_MIX_VLA_WORKDIR", str(Path(__file__).resolve().parent.parent / "runs")),
            "test0_live_switch", "async_handoff_latency.json",
        ),
    )
    args = parser.parse_args()

    env, task_description, initial_states = get_env_and_task()
    initial_state = initial_states[args.episode]
    warm_up(env, task_description, initial_state, args.warmup)

    records = []
    # Interleave and alternate ordering to reduce thermal/frequency bias.
    for rep in range(args.repetitions):
        order = ("sync", "async_naive") if rep % 2 == 0 else ("async_naive", "sync")
        for condition in order:
            result = run_trial(env, task_description, initial_state, condition)
            result.update({"repetition": rep, "episode": args.episode})
            records.append(result)
            if result["valid"]:
                print(
                    f"[rep={rep:02d} {condition}] L={1000 * result['L_handoff_s']:.3f}ms "
                    f"OFT={1000 * result['oft_http_latency_s']:.3f}ms "
                    f"zero_stall={result['zero_stall']} missed={result['missed_control_ticks']}",
                    flush=True,
                )
            else:
                print(f"[rep={rep:02d} {condition}] INVALID: {result['reason']}", flush=True)

    env.close()
    report = {
        "protocol": {
            "conditions": ["sync", "async_naive"],
            "async_future_included": False,
            "async_future_exclusion_reason": "No real future-state predictor exists in this repository.",
            "episode": args.episode,
            "warmup_queries_per_model": args.warmup,
            "repetitions_per_condition": args.repetitions,
            "chunk_size": CHUNK_SIZE,
        },
        "summary": aggregate(records),
        "records": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
