"""Seamless Adapter -> OFT handoff: does accurate future context change OFT's action?

Compares six conditions built from the SAME matched rollout (same trigger,
same fixed bridge length, same real prefix+bridge execution). Three are
deployable-ish (naive, vlash), one is the scoring target (sync_reference),
and three are OFFLINE DIAGNOSTIC / ACAUSAL oracle conditions that isolate
which half of "stale context" (vision vs. proprioception) actually drives
handoff mismatch -- none of these three are available to a real causal async
system, since all require the true T_switch frame or state, which only
exists after the bridge has already finished executing:

  - naive               : stale proprioception + stale image at T_predict
                           (the existing async-naive baseline).
  - vlash                : proprioception rolled forward using the remaining
                           committed Adapter actions. Position: additive, as
                           in arXiv:2512.01031, but each action is first run
                           through the REAL OSC_POSE controller's own
                           scale_action() (LIBERO's default config maps its
                           [-1,1] action space to +-0.05m / +-0.5rad per
                           step; the raw policy output is in that [-1,1]
                           space, not already physical units -- confirmed by
                           reading robosuite/controllers/osc.py and
                           single_arm.py's control(), not assumed). Rotation:
                           proper SO(3) composition
                           (q_new = axisangle2quat(scaled_delta) * q_old, via
                           robosuite.utils.transform_utils, the exact
                           function set_goal_orientation() uses internally)
                           rather than axis-angle addition -- verified from
                           control_utils.py's set_goal_orientation(), which
                           composes rotation matrices, not vectors. Gripper:
                           last-action-absolute rule, unchanged. Image stays
                           stale.
  - vlash_oracle_vision  : [ORACLE/ACAUSAL] same rolled-forward
                           proprioception, but the image is the TRUE T_switch
                           frame, obtained by actually executing Adapter's
                           remaining committed actions in the real env. This
                           is deliberately NOT F2F-AP (arXiv:2604.02408),
                           which has no released code and would require
                           training a separate flow predictor plus
                           fine-tuning OFT's vision encoder. The oracle
                           answers a prerequisite question -- would accurate
                           future vision even help -- before investing in
                           building the real thing.
  - oracle_proprio       : [ORACLE/ACAUSAL] stale T_predict image + TRUE
                           T_switch proprioception (ground truth, not
                           VLASH's estimate). Isolates the proprio-only
                           contribution to handoff mismatch.
  - oracle_vision        : [ORACLE/ACAUSAL] TRUE T_switch image + stale
                           T_predict proprioception. Isolates the
                           vision-only contribution to handoff mismatch.
  - sync_reference       : true proprioception + true image at T_switch. Not
                           a deployable condition -- the target the other
                           five are scored against.

This measures ACTION DIVERGENCE from sync_reference, not latency. Unlike
test2_async_bridge_length.py (imported from, never modified), there is no
ThreadPoolExecutor here: the oracle conditions are inherently acausal (they
need the real bridge to finish before they can exist), so all six conditions
are queried sequentially, after the real prefix+bridge execution completes.
Timing and action-quality are the two separate axes the async formulation
report calls out explicitly; this script is entirely about the latter.
"""
import argparse
import copy
import csv
import json
import os
import statistics
from pathlib import Path

import numpy as np

from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_env, quat2axisangle
from measure_async_handoff_latency import initialize_episode, percentile, query_adapter_chunk, warm_up
from prismatic.vla.constants import ROBOT_PLATFORM
from robosuite.utils.transform_utils import axisangle2quat, quat_multiply
from run_test0_switch_timing import CHUNK_SIZE, get_env_and_task, prepare_observation, process_action, call_policy


DEFAULT_OUTPUT_DIR = os.path.join(
    os.environ.get("ASYNC_MIX_VLA_WORKDIR", str(Path(__file__).resolve().parent.parent / "runs")),
    "seamless_handoff",
)
CONDITIONS = ("naive", "vlash", "vlash_oracle_vision", "oracle_proprio", "oracle_vision", "sync_reference")
SCORED_CONDITIONS = ("naive", "vlash", "vlash_oracle_vision", "oracle_proprio", "oracle_vision")
ORACLE_CONDITIONS = ("vlash_oracle_vision", "oracle_proprio", "oracle_vision")

# --mode generalize: LIBERO-10 is task_id 0-9; sample every task rather than
# just one, since a single deterministic episode (as the earlier --mode sweep
# used) cannot tell us whether stale-vision dominance is task-general or an
# artifact of one task/episode. Two episodes/task keeps this a single quick
# job (10 * 2 * CHUNK_SIZE = 160 single-run trials, no repetitions -- the
# pipeline is deterministic, so repeating a state buys nothing, per instruction).
GENERALIZATION_TASK_IDS = tuple(range(10))
GENERALIZATION_EPISODES_PER_TASK = 2


def get_env_and_task_for(task_id):
    """Same as run_test0_switch_timing.get_env_and_task(), but parameterized
    by task_id instead of that module's hardcoded TASK_ID constant -- needed
    here since --mode generalize sweeps across tasks, and that module is
    never modified (imported from throughout this script)."""
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_10"]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = get_libero_env(task, "openvla", resolution=256)
    return env, task_description, initial_states


def gripper_open_closed_bounds(env):
    """True open/closed qpos extremes for the two gripper finger joints, read
    directly from the compiled MuJoCo model (ground truth, since this is
    simulation) -- not hardcoded, since no numeric convention for
    ``robot0_gripper_qpos`` exists anywhere else in this repo.

    For each joint, one range endpoint is ~0 (fingers touching = closed) and
    the other is the max-magnitude extreme (fingers apart = open); picking by
    |value| avoids assuming which sign corresponds to open per joint.
    """
    robot = env.robots[0]
    joint_names = robot.gripper_joints  # same order used to build robot0_gripper_qpos
    ranges = np.array(
        [env.sim.model.jnt_range[env.sim.model.joint_name2id(name)] for name in joint_names],
        dtype=np.float64,
    )
    lo, hi = ranges[:, 0], ranges[:, 1]
    closed_bounds = np.where(np.abs(lo) < np.abs(hi), lo, hi)
    open_bounds = np.where(np.abs(lo) < np.abs(hi), hi, lo)
    return open_bounds, closed_bounds


def roll_forward_proprio(env, base_obs, remaining_actions, open_bounds, closed_bounds):
    """VLASH-style state rollforward, corrected to match this env's actual
    OSC_POSE controller math (verified against robosuite source, not
    assumed):

    - Each remaining raw action's first 6 dims are run through the REAL
      controller's own ``scale_action()`` (env.robots[0].controller) before
      use. LIBERO's OSC_POSE config is control_delta=True with
      input=[-1,1] -> output_max=[0.05,0.05,0.05,0.5,0.5,0.5]; the raw
      policy action is in that [-1,1] space, not already physical
      meters/radians (confirmed via single_arm.py's control(): arm_action =
      action[:6] is passed straight to controller.set_goal(), which calls
      scale_action() before using it).
    - Position (state dims 0-2): additive in world frame -- confirmed valid
      by control_utils.set_goal_position(): goal_position = current + delta.
    - Orientation (state dims 3-5): proper quaternion composition, matching
      control_utils.set_goal_orientation()'s
      goal_orientation = R(delta) @ R(current) exactly (via quat_multiply,
      not axis-angle vector addition, which control_utils.py never does).
    - Gripper (state dims 6-7): last remaining action's processed gripper
      command treated as "absolute" (clamped to the true open/closed qpos
      extremes), not accumulated -- gripper_qpos isn't a delta quantity the
      6D arm math applies to.
    """
    controller = env.robots[0].controller
    pos = np.asarray(base_obs["robot0_eef_pos"], dtype=np.float64).copy()
    quat = np.asarray(base_obs["robot0_eef_quat"], dtype=np.float64).copy()  # (x,y,z,w)
    for raw_action in remaining_actions:
        raw_action = np.asarray(raw_action, dtype=np.float64)
        scaled = np.asarray(controller.scale_action(raw_action[:6]), dtype=np.float64)
        pos += scaled[:3]
        delta_quat = axisangle2quat(scaled[3:6])
        quat = quat_multiply(delta_quat, quat)  # q_new = q_delta * q_old, world-frame composition
    axis_angle = quat2axisangle(quat).astype(np.float64)

    last_gripper_command = process_action(remaining_actions[-1])[6]
    predicted_gripper_qpos = open_bounds if last_gripper_command < 0 else closed_bounds
    predicted_gripper_qpos = np.asarray(predicted_gripper_qpos, dtype=np.float64)

    state = np.concatenate([pos, axis_angle, predicted_gripper_qpos]).astype(np.float32)
    return state, predicted_gripper_qpos


def build_condition_observation(image_source_obs, state_override=None):
    prepared = dict(prepare_observation(image_source_obs))
    if state_override is not None:
        prepared["state"] = np.asarray(state_override, dtype=np.float32)
    return prepared


def query_oft_chunk(observation, task_description):
    actions, latency = call_policy("oft", observation, task_description)
    actions = actions[:CHUNK_SIZE]
    if len(actions) != CHUNK_SIZE:
        raise ValueError(f"OFT returned {len(actions)} actions; expected {CHUNK_SIZE}")
    return actions, latency


def action_divergence(condition_actions, reference_actions):
    """condition_actions/reference_actions: (CHUNK_SIZE, 7) arrays, already
    passed through process_action (dims 0-5 = pose delta, dim 6 = binarized
    gripper in this env's -1=open/+1=close convention).
    """
    diffs = condition_actions[:, :6] - reference_actions[:, :6]
    per_step_l2 = np.linalg.norm(diffs, axis=1)
    gripper_agree = condition_actions[:, 6] == reference_actions[:, 6]
    return {
        "first_action_l2": float(per_step_l2[0]),
        "first_action_gripper_agree": bool(gripper_agree[0]),
        "chunk_mean_l2": float(per_step_l2.mean()),
        "chunk_gripper_agree_rate": float(gripper_agree.mean()),
    }


def run_matched_trial(env, task_description, initial_state, remaining_actions):
    if not 1 <= remaining_actions <= CHUNK_SIZE:
        raise ValueError(f"remaining_actions must be in [1,{CHUNK_SIZE}]")

    obs = initialize_episode(env, initial_state)
    open_bounds, closed_bounds = gripper_open_closed_bounds(env)
    adapter_actions, adapter_http_latency = query_adapter_chunk(obs, task_description)
    trigger_after = CHUNK_SIZE - remaining_actions

    # Execute the prefix of the already-committed Adapter chunk for real.
    for raw_action in adapter_actions[:trigger_after]:
        action = process_action(raw_action)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return {
                "valid": False,
                "reason": "adapter_succeeded_before_T_predict",
                "remaining_adapter_actions": remaining_actions,
            }

    # Artificial trigger: this is what a real async system would actually have.
    stale_obs = copy.deepcopy(obs)
    remaining = adapter_actions[trigger_after:]
    vlash_state, predicted_gripper_qpos = roll_forward_proprio(
        env, stale_obs, remaining, open_bounds, closed_bounds
    )

    # Execute the remaining committed Adapter actions for real -> true T_switch state.
    for raw_action in remaining:
        action = process_action(raw_action)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return {
                "valid": False,
                "reason": "adapter_succeeded_during_bridge",
                "remaining_adapter_actions": remaining_actions,
            }
    true_future_obs = copy.deepcopy(obs)

    # Real proprio states at both ends of the bridge, computed once and reused
    # for the oracle_proprio/oracle_vision cross-conditions below (avoids
    # recomputing prepare_observation's state formula twice for the same obs).
    stale_state = prepare_observation(stale_obs)["state"]
    true_state = prepare_observation(true_future_obs)["state"]

    condition_observations = {
        "naive": build_condition_observation(stale_obs),
        "vlash": build_condition_observation(stale_obs, state_override=vlash_state),
        "vlash_oracle_vision": build_condition_observation(true_future_obs, state_override=vlash_state),
        # [ORACLE/ACAUSAL] stale image, but ground-truth (not VLASH-estimated) T_switch proprio.
        "oracle_proprio": build_condition_observation(stale_obs, state_override=true_state),
        # [ORACLE/ACAUSAL] true T_switch image, but stale T_predict proprio.
        "oracle_vision": build_condition_observation(true_future_obs, state_override=stale_state),
        "sync_reference": build_condition_observation(true_future_obs),
    }

    predicted_actions = {}
    oft_http_latency_s = {}
    for name in CONDITIONS:
        actions, latency = query_oft_chunk(condition_observations[name], task_description)
        predicted_actions[name] = np.stack([process_action(a) for a in actions])
        oft_http_latency_s[name] = latency

    reference = predicted_actions["sync_reference"]
    divergence = {
        name: action_divergence(predicted_actions[name], reference) for name in SCORED_CONDITIONS
    }

    actual_gripper_qpos = np.asarray(true_future_obs["robot0_gripper_qpos"], dtype=np.float64)
    gripper_error = predicted_gripper_qpos - actual_gripper_qpos
    actual_pos = np.asarray(true_future_obs["robot0_eef_pos"], dtype=np.float64)
    actual_axis_angle = quat2axisangle(true_future_obs["robot0_eef_quat"]).astype(np.float64)
    predicted_pos = vlash_state[:3].astype(np.float64)
    predicted_axis_angle = vlash_state[3:6].astype(np.float64)

    # Explicit 3-way proprio trace: state at T_predict, VLASH's predicted
    # state at T_switch, and the true state at T_switch, component-by-
    # component. Reuses stale_state/true_state computed above.
    proprio_at_t_predict = np.asarray(stale_state, dtype=np.float64)
    true_proprio_at_t_switch = np.asarray(true_state, dtype=np.float64)
    predicted_proprio_at_t_switch = vlash_state.astype(np.float64)
    component_labels = ["eef_x", "eef_y", "eef_z", "aa_x", "aa_y", "aa_z", "gripper_j1", "gripper_j2"]
    proprio_trace = {
        "component": component_labels,
        "proprio_at_t_predict": proprio_at_t_predict.tolist(),
        "predicted_proprio_at_t_switch": predicted_proprio_at_t_switch.tolist(),
        "true_proprio_at_t_switch": true_proprio_at_t_switch.tolist(),
        "error_per_component": (predicted_proprio_at_t_switch - true_proprio_at_t_switch).tolist(),
    }

    return {
        "valid": True,
        "remaining_adapter_actions": remaining_actions,
        "trigger_after": trigger_after,
        "adapter_http_latency_s": adapter_http_latency,
        "oft_http_latency_s": oft_http_latency_s,
        "proprio_trace": proprio_trace,
        "rollforward_validation": {
            "predicted_gripper_qpos": predicted_gripper_qpos.tolist(),
            "actual_gripper_qpos": actual_gripper_qpos.tolist(),
            "gripper_qpos_error": gripper_error.tolist(),
            "gripper_qpos_error_l2": float(np.linalg.norm(gripper_error)),
            "predicted_eef_pos": predicted_pos.tolist(),
            "actual_eef_pos": actual_pos.tolist(),
            "eef_pos_error_l2": float(np.linalg.norm(predicted_pos - actual_pos)),
            "predicted_axis_angle": predicted_axis_angle.tolist(),
            "actual_axis_angle": actual_axis_angle.tolist(),
            "axis_angle_error_l2": float(np.linalg.norm(predicted_axis_angle - actual_axis_angle)),
        },
        "action_divergence_vs_sync_reference": divergence,
        "predicted_first_actions": {name: predicted_actions[name][0].tolist() for name in CONDITIONS},
    }


def print_trace(result):
    print("=" * 92)
    print(f"SEAMLESS HANDOFF TRACE: remaining Adapter actions = {result['remaining_adapter_actions']}")
    print("=" * 92)
    pt = result["proprio_trace"]
    print("Proprio trace (T_predict -> predicted T_switch (VLASH) -> true T_switch, per component):")
    print(f"  {'component':12s} {'t_predict':>12s} {'predicted':>12s} {'true':>12s} {'error':>12s}")
    for i, name in enumerate(pt["component"]):
        print(
            f"  {name:12s} {pt['proprio_at_t_predict'][i]:12.5f} "
            f"{pt['predicted_proprio_at_t_switch'][i]:12.5f} "
            f"{pt['true_proprio_at_t_switch'][i]:12.5f} "
            f"{pt['error_per_component'][i]:12.5f}"
        )
    rv = result["rollforward_validation"]
    print("Rollforward validation (VLASH proprioception estimate vs. true T_switch state):")
    print(
        f"  gripper_qpos  predicted={rv['predicted_gripper_qpos']} actual={rv['actual_gripper_qpos']} "
        f"error_L2={rv['gripper_qpos_error_l2']:.5f}"
    )
    print(f"  eef_pos       error_L2={rv['eef_pos_error_l2']:.5f}")
    print(f"  axis_angle    error_L2={rv['axis_angle_error_l2']:.5f}")
    print("Action divergence vs. sync_reference (post-process_action; dims 0-5 = pose delta, dim 6 = gripper):")
    for name, div in result["action_divergence_vs_sync_reference"].items():
        tag = "[oracle]" if name in ORACLE_CONDITIONS else "        "
        print(
            f"  {tag} {name:20s} first_action_L2={div['first_action_l2']:.5f} "
            f"first_gripper_agree={div['first_action_gripper_agree']} "
            f"chunk_mean_L2={div['chunk_mean_l2']:.5f} "
            f"chunk_gripper_agree_rate={div['chunk_gripper_agree_rate']:.2f}"
        )
    print(
        "Reminder: vlash_oracle_vision / oracle_proprio / oracle_vision all use the true T_switch "
        "frame and/or true T_switch proprio (obtained by actually executing the bridge) -- offline "
        "diagnostic upper bounds only, NOT F2F-AP, not available to a real causal async system."
    )


def aggregate_bucket(valid):
    """Summary stats for one bucket of already-filtered valid records (all
    sharing the same remaining_adapter_actions, or the whole pool)."""
    out = {"valid_trials": len(valid)}
    for name in SCORED_CONDITIONS:
        first_l2 = [r["action_divergence_vs_sync_reference"][name]["first_action_l2"] for r in valid]
        chunk_l2 = [r["action_divergence_vs_sync_reference"][name]["chunk_mean_l2"] for r in valid]
        gripper_agree = [r["action_divergence_vs_sync_reference"][name]["first_action_gripper_agree"] for r in valid]
        out[name] = {
            "first_action_l2_median": statistics.median(first_l2) if first_l2 else None,
            "first_action_l2_p95": percentile(first_l2, 0.95),
            "chunk_mean_l2_median": statistics.median(chunk_l2) if chunk_l2 else None,
            "first_action_gripper_agree_rate": (sum(gripper_agree) / len(gripper_agree)) if gripper_agree else None,
        }
    gripper_err = [r["rollforward_validation"]["gripper_qpos_error_l2"] for r in valid]
    pos_err = [r["rollforward_validation"]["eef_pos_error_l2"] for r in valid]
    aa_err = [r["rollforward_validation"]["axis_angle_error_l2"] for r in valid]
    out["rollforward_validation"] = {
        "gripper_qpos_error_l2_median": statistics.median(gripper_err) if gripper_err else None,
        "gripper_qpos_error_l2_p95": percentile(gripper_err, 0.95),
        "eef_pos_error_l2_median": statistics.median(pos_err) if pos_err else None,
        "axis_angle_error_l2_median": statistics.median(aa_err) if aa_err else None,
    }
    return out


def aggregate_by_remaining(records):
    """Pooled across every tested state (task, episode), bucketed by bridge
    length (1..CHUNK_SIZE) -- mirrors test2_async_bridge_length.py's own
    aggregate() bucketing. Answers: does mismatch grow with bridge length,
    now with real between-state variance instead of one repeated state."""
    out = {}
    for remaining in range(1, CHUNK_SIZE + 1):
        bucket = [r for r in records if r.get("valid") and r["remaining_adapter_actions"] == remaining]
        out[str(remaining)] = aggregate_bucket(bucket)
    return out


def aggregate_by_task(records):
    """Bucketed by task_id, pooled across episodes and bridge lengths.
    Answers: are there tasks where proprioception matters more."""
    out = {}
    task_ids = sorted({r["task_id"] for r in records if r.get("valid") and "task_id" in r})
    for task_id in task_ids:
        bucket = [r for r in records if r.get("valid") and r.get("task_id") == task_id]
        out[str(task_id)] = aggregate_bucket(bucket)
    return out


def aggregate_by_state(records):
    """Bucketed by (task_id, trial_index), pooled across bridge lengths --
    the per-task/episode report, keyed "{task_id}_{trial_index}" matching
    this project's established episode-key convention."""
    out = {}
    keys = sorted({(r["task_id"], r["trial_index"]) for r in records if r.get("valid") and "task_id" in r})
    for task_id, trial_index in keys:
        bucket = [
            r
            for r in records
            if r.get("valid") and r.get("task_id") == task_id and r.get("trial_index") == trial_index
        ]
        out[f"{task_id}_{trial_index}"] = aggregate_bucket(bucket)
    return out


RAW_CSV_FIELDS = (
    ["task_id", "trial_index", "repetition", "remaining_adapter_actions"]
    + [f"{name}_first_action_l2" for name in SCORED_CONDITIONS]
    + [f"{name}_chunk_mean_l2" for name in SCORED_CONDITIONS]
    + [f"{name}_first_gripper_agree" for name in SCORED_CONDITIONS]
    + ["gripper_qpos_error_l2", "eef_pos_error_l2", "axis_angle_error_l2"]
)


def write_raw_csv(records, json_output_path):
    csv_path = os.path.splitext(json_output_path)[0] + "_raw.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            if not record.get("valid"):
                continue
            div = record["action_divergence_vs_sync_reference"]
            rv = record["rollforward_validation"]
            row = {
                "task_id": record.get("task_id"),
                "trial_index": record.get("trial_index"),
                "repetition": record.get("repetition"),
                "remaining_adapter_actions": record["remaining_adapter_actions"],
                "gripper_qpos_error_l2": rv["gripper_qpos_error_l2"],
                "eef_pos_error_l2": rv["eef_pos_error_l2"],
                "axis_angle_error_l2": rv["axis_angle_error_l2"],
            }
            for name in SCORED_CONDITIONS:
                row[f"{name}_first_action_l2"] = div[name]["first_action_l2"]
                row[f"{name}_chunk_mean_l2"] = div[name]["chunk_mean_l2"]
                row[f"{name}_first_gripper_agree"] = div[name]["first_action_gripper_agree"]
            writer.writerow(row)
    return csv_path


def main():
    if ROBOT_PLATFORM != "LIBERO":
        raise RuntimeError(
            f"This script requires LIBERO constants, but auto-detection selected {ROBOT_PLATFORM}. "
            "Launch through the provided Slurm script."
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["trace", "sweep", "generalize"], required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--remaining", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = []

    if args.mode == "generalize":
        # Multiple tasks/episodes, single deterministic run per
        # (task, episode, remaining) -- no repetitions: the pipeline is
        # deterministic, so repeating a state produces identical duplicates
        # (confirmed empirically in --mode sweep), not real samples.
        warmed_up = False
        for task_id in GENERALIZATION_TASK_IDS:
            env, task_description, initial_states = get_env_and_task_for(task_id)
            if not warmed_up:
                warm_up(env, task_description, initial_states[0], args.warmup)
                warmed_up = True
            for trial_index in range(GENERALIZATION_EPISODES_PER_TASK):
                initial_state = initial_states[trial_index]
                for remaining in range(1, CHUNK_SIZE + 1):
                    result = run_matched_trial(env, task_description, initial_state, remaining)
                    result["task_id"] = task_id
                    result["trial_index"] = trial_index
                    records.append(result)
                    if result.get("valid"):
                        div = result["action_divergence_vs_sync_reference"]
                        print(
                            f"[task={task_id} ep={trial_index} remaining={remaining}] "
                            f"naive_L2={div['naive']['first_action_l2']:.4f} "
                            f"vlash_L2={div['vlash']['first_action_l2']:.4f} "
                            f"vlash_oracle_L2={div['vlash_oracle_vision']['first_action_l2']:.4f} "
                            f"oracle_proprio_L2={div['oracle_proprio']['first_action_l2']:.4f} "
                            f"oracle_vision_L2={div['oracle_vision']['first_action_l2']:.4f} "
                            f"gripper_err_L2={result['rollforward_validation']['gripper_qpos_error_l2']:.4f}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[task={task_id} ep={trial_index} remaining={remaining}] "
                            f"INVALID: {result['reason']}",
                            flush=True,
                        )
                    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
                    with open(args.output, "w") as f:
                        json.dump({"mode": "generalize", "records": records}, f, indent=2)
                    write_raw_csv(records, args.output)
            env.close()
        report = {
            "mode": "generalize",
            "task_ids": list(GENERALIZATION_TASK_IDS),
            "episodes_per_task": GENERALIZATION_EPISODES_PER_TASK,
            "bridge_length_range": [1, CHUNK_SIZE],
            "summary_by_remaining": aggregate_by_remaining(records),
            "summary_by_task": aggregate_by_task(records),
            "summary_by_state": aggregate_by_state(records),
            "records": records,
        }
        csv_path = write_raw_csv(records, args.output)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved {args.output}")
        print(f"Saved {csv_path}")
        return

    env, task_description, initial_states = get_env_and_task()
    initial_state = initial_states[args.episode]
    warm_up(env, task_description, initial_state, args.warmup)

    if args.mode == "trace":
        result = run_matched_trial(env, task_description, initial_state, args.remaining)
        records.append(result)
        if not result.get("valid"):
            raise RuntimeError(result["reason"])
        print_trace(result)
        report = {
            "mode": "trace",
            "episode": args.episode,
            "remaining_adapter_actions": args.remaining,
            "record": result,
        }
    else:
        # Sweep the full bridge-length range (1..CHUNK_SIZE), not just
        # --remaining -- rotating/reversing the order each repetition to
        # spread thermal/frequency bias, same pattern as
        # test2_async_bridge_length.py's own sweep mode.
        for repetition in range(args.repetitions):
            order = list(range(1, CHUNK_SIZE + 1))
            shift = repetition % CHUNK_SIZE
            order = order[shift:] + order[:shift]
            if repetition % 2:
                order.reverse()
            for remaining in order:
                result = run_matched_trial(env, task_description, initial_state, remaining)
                result["repetition"] = repetition
                records.append(result)
                if result.get("valid"):
                    div = result["action_divergence_vs_sync_reference"]
                    print(
                        f"[rep={repetition:02d} remaining={remaining}] "
                        f"naive_L2={div['naive']['first_action_l2']:.4f} "
                        f"vlash_L2={div['vlash']['first_action_l2']:.4f} "
                        f"vlash_oracle_L2={div['vlash_oracle_vision']['first_action_l2']:.4f} "
                        f"oracle_proprio_L2={div['oracle_proprio']['first_action_l2']:.4f} "
                        f"oracle_vision_L2={div['oracle_vision']['first_action_l2']:.4f} "
                        f"gripper_err_L2={result['rollforward_validation']['gripper_qpos_error_l2']:.4f}",
                        flush=True,
                    )
                else:
                    print(f"[rep={repetition:02d} remaining={remaining}] INVALID: {result['reason']}", flush=True)
                os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
                with open(args.output, "w") as f:
                    json.dump({"mode": "sweep", "records": records}, f, indent=2)
                write_raw_csv(records, args.output)
        report = {
            "mode": "sweep",
            "episode": args.episode,
            "bridge_length_range": [1, CHUNK_SIZE],
            "repetitions_per_bridge_length": args.repetitions,
            "summary": aggregate_by_remaining(records),
            "records": records,
        }

    env.close()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {args.output}")
    if args.mode == "sweep":
        csv_path = write_raw_csv(records, args.output)
        print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
