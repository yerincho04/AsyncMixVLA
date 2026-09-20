"""
Test 0 -- OFT recovery vs. mid-rollout switch timing (synchronous, true observation).

For a fixed episode, run VLA-Adapter from step 0 to a switch step t (true env
execution, perturbation firing on its normal schedule if applicable), snapshot
the exact MuJoCo sim state at t, then continue the SAME episode with OFT from
t to the horizon. No latency/async modeling, no prediction -- t is chosen only
by the sweep grid.

Generalizes run_stage3_branching.py's single-snapshot (chunk-boundary-only)
branch pattern to an arbitrary target step t (mid-chunk cutoff allowed), run
once per (episode, t) pair -- i.e. this reruns the Adapter prefix fresh for
every t rather than reusing one rollout's snapshots across all t's. Simpler
and directly reuses the already-validated single-snapshot restore mechanism;
the cost tradeoff is discussed in the accompanying report, not optimized here
per instruction to keep this test clean.

Per explicit design decision (2026-08-05): LIBERO's `done` flag is ONLY ever
True on success (see bddl_base_domain.py step() override) -- there is no
terminal-failure predicate in the engine. The "unrecoverable-at-switch" bucket
from the original spec is therefore empty by construction: the only exclusion
category that exists is "Adapter already succeeded before reaching t."

Three modes:
  --mode trace_one  : run ONE episode across a small set of t's, print full trace.
  --mode validate    : determinism (2x same sweep) + t=0 matches standalone OFT +
                        t=horizon matches standalone Adapter, on a small episode set.
  --mode sweep       : the full 50-episode x ~13-t x {clean,seed101,seed102} sweep.
"""
import argparse
import copy
import hashlib
import json
import pickle
import time
from collections import deque
from pathlib import Path

import msgpack
import msgpack_numpy as m
import numpy as np
import requests

m.patch()

from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    get_libero_dummy_action,
    quat2axisangle,
)
from experiments.robot.robot_utils import normalize_gripper_action, invert_gripper_action

from run_mixed_libero10 import (
    TASK_MAX_STEPS,
    get_source_initial_z_map,
    schedule_from_manifest_event,
    source_displacement_allowed,
    translate_body_xy,
    get_body_position,
)

TASK_ID = 6
NUM_STEPS_WAIT = 10
CHUNK_SIZE = 8
MAX_STEPS = TASK_MAX_STEPS["libero_10"]  # 520
SOURCE_BODY = "chocolate_pudding_1_main"

import os as _os  # module-level, just for ENDPOINTS's port env-var lookup below

# Ports are env-var-overridable (ADAPTER_PORT/OFT_PORT), defaulting to the
# original fixed 8002/8001 -- added so parallel sweep shards can each claim a
# unique port pair and never collide regardless of node co-location (a fixed-
# port collision crashed a live_validate shard mid-run on 2026-08-06: two
# shards landed on the same physical node and both tried to bind 127.0.0.1:
# 8002). Zero behavior change for anything that doesn't set these.
ENDPOINTS = {
    "adapter": f"http://127.0.0.1:{_os.environ.get('ADAPTER_PORT', '8002')}/act",
    "oft": f"http://127.0.0.1:{_os.environ.get('OFT_PORT', '8001')}/act",
    # F2F-AP baseline/system-component track: only reachable if the OFT server
    # was started with --f2f_ap_checkpoint (see serve_oft_libero10.py). /act
    # itself and the "oft" endpoint above are untouched.
    "oft_f2f_ap": f"http://127.0.0.1:{_os.environ.get('OFT_PORT', '8001')}/act_with_f2f_ap",
    # AsyncMixVLA visual-handoff mode vision_alignment="action_consistency":
    # only reachable if the OFT server was started with
    # --action_consistency_checkpoint. /act, /act_with_f2f_ap and "oft" above
    # are untouched.
    "oft_action_consistency": f"http://127.0.0.1:{_os.environ.get('OFT_PORT', '8001')}/act_with_action_consistency",
}

SESSION = requests.Session()


class GateArgs:
    source_displace_require_on_table = True
    source_displace_table_z_tol = 0.025
    source_displace_min_gripper_dist = 0.08
    obj_displace_zero_velocity = True
    # Added 2026-08-25: same fix already applied to feasibility_scan.py's
    # GATE_ARGS for the same reason -- harmless for task 6 (its source maps
    # to a semantic BDDL region, so source_is_already_on_mapped_target
    # short-circuits before source_is_inside_or_near_target ever reads
    # these), but task 4's source maps to a REAL body (plate_2_main), which
    # does reach that code path and crashed with AttributeError until now.
    # Values copied verbatim from feasibility_scan.py's GATE_ARGS -- not
    # newly chosen, so the underlying validated gate logic is unchanged.
    target_displace_carry_sources = True
    target_displace_carry_xy_tol = 0.12
    target_displace_carry_z_tol = 0.12


def prepare_observation(obs):
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)
    state = np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    ).astype(np.float32)
    return {"full_image": img, "wrist_image": wrist_img, "state": state}


def process_action(action):
    action = normalize_gripper_action(action, binarize=True)
    action = invert_gripper_action(action)
    return action


def call_policy(model, observation, task_description, timeout=120):
    payload = {
        "full_image": observation["full_image"],
        "wrist_image": observation["wrist_image"],
        "state": observation["state"],
        "task_description": task_description,
    }
    packed = msgpack.packb(payload, default=m.encode, use_bin_type=True)
    t0 = time.perf_counter()
    r = SESSION.post(ENDPOINTS[model], data=packed, headers={"Content-Type": "application/msgpack"}, timeout=timeout)
    r.raise_for_status()
    latency = time.perf_counter() - t0
    out = msgpack.unpackb(r.content, object_hook=m.decode, raw=False)
    actions = np.asarray(out["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] != 7:
        raise ValueError(f"Expected [T,7], got {actions.shape}")
    return actions, latency


def call_policy_f2f_ap(observation, task_description, bridge_actions, timeout=120):
    """F2F-AP baseline/system-component track: like call_policy("oft", ...)
    but posts to /act_with_f2f_ap, which also needs the committed
    bridge_actions (the F2F-AP predictor's only causal input besides the
    current image) to predict the T_switch visual latent server-side.
    Timed identically to call_policy so its cost is comparable and can be
    charged to the async compute budget the same way."""
    payload = {
        "full_image": observation["full_image"],
        "wrist_image": observation["wrist_image"],
        "state": observation["state"],
        "task_description": task_description,
        "bridge_actions": np.asarray(bridge_actions, dtype=np.float64),
    }
    packed = msgpack.packb(payload, default=m.encode, use_bin_type=True)
    t0 = time.perf_counter()
    r = SESSION.post(ENDPOINTS["oft_f2f_ap"], data=packed, headers={"Content-Type": "application/msgpack"}, timeout=timeout)
    r.raise_for_status()
    latency = time.perf_counter() - t0
    out = msgpack.unpackb(r.content, object_hook=m.decode, raw=False)
    actions = np.asarray(out["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] != 7:
        raise ValueError(f"Expected [T,7], got {actions.shape}")
    return actions, latency


def call_policy_action_consistency(observation, task_description, bridge_actions, timeout=120):
    """AsyncMixVLA visual-handoff mode vision_alignment="action_consistency":
    like call_policy_f2f_ap but posts to /act_with_action_consistency (needs
    the OFT server started with --action_consistency_checkpoint). Same
    payload shape and same identical timing as call_policy_f2f_ap so the
    frozen residual's own inference cost is charged to the async compute
    budget the same way."""
    payload = {
        "full_image": observation["full_image"],
        "wrist_image": observation["wrist_image"],
        "state": observation["state"],
        "task_description": task_description,
        "bridge_actions": np.asarray(bridge_actions, dtype=np.float64),
    }
    packed = msgpack.packb(payload, default=m.encode, use_bin_type=True)
    t0 = time.perf_counter()
    r = SESSION.post(ENDPOINTS["oft_action_consistency"], data=packed,
                     headers={"Content-Type": "application/msgpack"}, timeout=timeout)
    r.raise_for_status()
    latency = time.perf_counter() - t0
    out = msgpack.unpackb(r.content, object_hook=m.decode, raw=False)
    actions = np.asarray(out["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] != 7:
        raise ValueError(f"Expected [T,7], got {actions.shape}")
    return actions, latency


def apply_displacement_step(env, episode_step, displacement_schedule, active_displacements,
                             source_initial_z_map, current_obs, diagnostics, log_events=None,
                             task_id=TASK_ID):
    """Same per-step displacement handling as run_mixed_libero10.run_episode() /
    run_stage3_branching.apply_displacement_step(), restricted to group=='source'
    (the only group this test's manifests use). current_obs is passed explicitly
    (not a module-level cell) since this function is called from two different
    stepping loops (prefix + continuation) in this file.

    `task_id` defaults to the module-level TASK_ID (6) so every existing
    caller that predates this parameter (run_live_switch,
    run_sync_handoff_latency, perturbation_replay.py's replay loop) is
    completely unaffected. Generalized (2026-08-25) for task-4 live use:
    those callers were previously silently always gating the displacement
    against task 6's own object/target/region map regardless of which task
    was actually running, which happened to be harmless everywhere it was
    used so far (task 6 only) but would have been a real bug the moment a
    task-4 live rollout called this with a real manifest event.
    """
    if episode_step in displacement_schedule:
        for event in displacement_schedule[episode_step]:
            body_name = event["body_name"]
            if body_name in active_displacements:
                continue
            if not source_displacement_allowed(
                env=env, obs=current_obs, task_id=task_id, body_name=body_name,
                source_initial_z_map=source_initial_z_map, args=GateArgs,
            ):
                if log_events is not None:
                    log_events.append((episode_step, "skip_gate_blocked", body_name))
                continue
            active_displacements[body_name] = {
                "group": event["group"], "remaining": event["duration"],
                "delta_xy_per_step": event["delta_xy_per_step"],
                "total_delta_xy": event["total_delta_xy"],
                "displacement_magnitude": event["displacement_magnitude"],
            }
            if diagnostics is not None and body_name == diagnostics["body_name"]:
                diagnostics["fired"] = True
                diagnostics["never_reached"] = False
                diagnostics["position_before"] = get_body_position(env, body_name).tolist()
            if log_events is not None:
                log_events.append((episode_step, "fired", body_name))

    for body_name in list(active_displacements.keys()):
        if not source_displacement_allowed(
            env=env, obs=current_obs, task_id=task_id, body_name=body_name,
            source_initial_z_map=source_initial_z_map, args=GateArgs,
        ):
            if diagnostics is not None and body_name == diagnostics["body_name"]:
                diagnostics["cancelled_early"] = True
                diagnostics["position_after"] = get_body_position(env, body_name).tolist()
            del active_displacements[body_name]
            continue
        delta_xy = active_displacements[body_name]["delta_xy_per_step"]
        moved = translate_body_xy(env, body_name, delta_xy, zero_velocity=GateArgs.obj_displace_zero_velocity)
        if not moved:
            if diagnostics is not None and body_name == diagnostics["body_name"]:
                diagnostics["cancelled_early"] = True
                diagnostics["position_after"] = get_body_position(env, body_name).tolist()
            del active_displacements[body_name]


def run_prefix_to_step(env, task_description, initial_state, manifest_event, target_step, log_prefix="adapter"):
    """Runs `log_prefix` model from step 0 (post-warmup) to target_step (or until
    success, whichever comes first). target_step may fall anywhere, including
    mid-chunk -- any unconsumed queued actions from a partially-used chunk are
    simply discarded once target_step is reached.

    Returns a dict:
      succeeded_by_target=True  -> {"succeeded_by_target": True, "success_step": int, "action_log": [...]}
      succeeded_by_target=False -> {"succeeded_by_target": False, "snapshot": ..., "reached_step": target_step,
                                     "displacement_schedule": {...}, "active_displacements": {...},
                                     "source_initial_z_map": {...}, "action_log": [...], "obs_at_snapshot": {...}}
    """
    env.reset()
    obs = env.set_init_state(initial_state)

    for _ in range(NUM_STEPS_WAIT):
        obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))

    displacement_schedule = schedule_from_manifest_event(manifest_event) if manifest_event else {}
    active_displacements = {}
    source_initial_z_map = get_source_initial_z_map(env, TASK_ID) if manifest_event else {}
    diagnostics = None
    if manifest_event is not None:
        diagnostics = {
            "body_name": manifest_event["body_name"], "fired": False,
            "cancelled_early": False, "never_reached": True,
            "position_before": None, "position_after": None,
        }

    action_queue = deque()
    action_log = []
    t = 0

    while t < target_step:
        if len(action_queue) == 0:
            observation = prepare_observation(obs)
            actions, _ = call_policy(log_prefix, observation, task_description)
            for a in actions[:CHUNK_SIZE]:
                action_queue.append(a)

        action = process_action(action_queue.popleft())

        apply_displacement_step(env, t, displacement_schedule, active_displacements,
                                 source_initial_z_map, obs, diagnostics)

        obs, reward, done, info = env.step(action.tolist())
        action_log.append((t, log_prefix, action.tolist()))

        for body_name in list(active_displacements.keys()):
            active_displacements[body_name]["remaining"] -= 1
            if active_displacements[body_name]["remaining"] <= 0:
                del active_displacements[body_name]

        t += 1

        if done:
            return {"succeeded_by_target": True, "success_step": t, "action_log": action_log,
                    "diagnostics": diagnostics}

    snapshot = env.get_sim_state()
    # env.get_sim_state()/set_state_from_flattened() only serialize (time, qpos, qvel)
    # -- see robosuite's binding_utils.py MjSimState -- so qacc_warmstart/ctrl are lost
    # across a restore unless captured and reapplied separately (root-cause confirmed
    # via diag_obscompare: qpos/qvel restore bit-exact but the derived observation does
    # not, in 0/14 cases, unless these two fields are also restored).
    extra_state = {
        "qacc_warmstart": np.copy(env.sim.data.qacc_warmstart),
        "ctrl": np.copy(env.sim.data.ctrl),
    }
    return {
        "succeeded_by_target": False,
        "snapshot": snapshot,
        "extra_state": extra_state,
        "reached_step": t,
        "displacement_schedule": displacement_schedule,
        "active_displacements": copy.deepcopy(active_displacements),
        "source_initial_z_map": source_initial_z_map,
        "action_log": action_log,
        "obs_at_snapshot": prepare_observation(obs),
        "diagnostics": diagnostics,
    }


def restore_full_state(env, snapshot, extra_state):
    """Restores snapshot into env, INCLUDING qacc_warmstart/ctrl, which
    env.set_init_state()/regenerate_obs_from_state() does not capture or restore
    (those only round-trip time/qpos/qvel). Mirrors LIBERO's ControlEnv.
    regenerate_obs_from_state() exactly, with one deliberate addition. Callers
    must have already called env.reset() -- required when reusing the same env
    object, since robosuite's self.timestep/self.done live outside sim.get_state().

    qacc_warmstart is injected TWICE, not once: once before sim.forward() (mostly
    inert -- forward() unconditionally overwrites qacc_warmstart with its own
    freshly-solved value at the end of the constraint solve, confirmed empirically
    via debug_restore4/5.py, so an injection that only happens before forward()
    never survives to matter) and once AGAIN immediately after, so the value the
    NEXT real env.step()/mj_step() actually reads as its solver seed is the one
    continuous execution would have carried, not forward()'s own recomputed one.
    This matters in practice, not just in principle: outcome_stability testing
    (diag_outcome_stability, job 1973315) showed the pre-this-fix version flipped
    2/4 tested cases from baseline SUCCESS to complete restore FAILURE (full
    520-step horizon, no success) -- a real, sweep-invalidating divergence, not a
    benign rounding artifact."""
    env.set_state(snapshot)
    if extra_state is not None:
        env.sim.data.qacc_warmstart[:] = extra_state["qacc_warmstart"]
        env.sim.data.ctrl[:] = extra_state["ctrl"]
    env.sim.forward()
    if extra_state is not None:
        env.sim.data.qacc_warmstart[:] = extra_state["qacc_warmstart"]
    # CONFIRMED root cause of the outcome-flip rate (2026-08-06, debug_controller.py):
    # each robosuite controller (self.robots[i].controller) caches ee_pos/ee_ori_mat/
    # goal_pos/etc. as plain Python attributes, refreshed only when self.new_update is
    # True. After env.reset() + set_state() above, new_update is already False (something
    # during reset's own internal sim.forward() consumes it against the PRE-restore
    # default pose) -- so the controller's cached ee_pos stays frozen at the wrong
    # (default reset) position even though the true live sim.data.site_xpos is correctly
    # restored (confirmed bit-close, diff ~5e-5, matching the established noise floor).
    # This is NOT an MjData/physics bug -- it's a stale Python-level cache the restore
    # path never invalidated. The very first post-restore action's goal_pos is computed
    # as this STALE ee_pos + delta, producing an immediate ~0.026 positional error that
    # matches the magnitude of the outcome-flip-causing jump measured in
    # diag_trajectory_diff (job 1973508: 0.00028 -> 0.02706 in the single step after
    # restore). Force every robot's controller to refresh from the now-correct live sim
    # state before any action can be computed against the stale cache.
    for robot in env.env.robots:
        robot.controller.update(force=True)
    env.check_success()
    env._post_process()
    env._update_observables(force=True)
    # NOTE: _get_observations() is defined on the underlying robosuite env
    # (env.env), not on LIBERO's ControlEnv wrapper itself -- ControlEnv.
    # regenerate_obs_from_state() calls self.env._get_observations() for the
    # same reason. Calling env._get_observations() directly raises
    # AttributeError: 'OffScreenRenderEnv' object has no attribute '_get_observations'.
    return env.env._get_observations()


def run_continuation(env, task_description, snapshot, start_step, model,
                      displacement_schedule, active_displacements_init, source_initial_z_map,
                      diagnostics=None, max_steps=MAX_STEPS, extra_state=None, obs_at_switch=None):
    """Restores snapshot (env.reset() + restore_full_state(), including
    qacc_warmstart/ctrl -- required when reusing the same env object, since
    robosuite's self.timestep/self.done live outside sim.get_state()), then runs
    `model` from start_step to done/max_steps.

    obs_at_switch: the prepared observation continuous execution actually exposed
    at the switch instant (prefix["obs_at_snapshot"]), used ONLY for the first
    policy query. Root-cause finding (debug_restore4/5.py, confirmed via repeated
    forward()-call convergence tests): MuJoCo's mj_step() leaves site_xpos/derived
    observables synced to kinematics from BEFORE that step's own final position
    integration, while qpos is already fully integrated -- i.e., env.step()'s
    returned observation is inherently one physics-substep "behind" its own qpos.
    restore_full_state()'s explicit sim.forward() call correctly (and reproducibly
    -- confirmed idempotent after the first call) syncs to the truly-current qpos,
    which is why it can NEVER bit-match what env.step() itself would have shown at
    that instant, regardless of what auxiliary MjData fields are captured/restored
    (qacc_warmstart/ctrl restoration, tested empirically, changes nothing here).
    Reusing the exact observation continuous execution already produced sidesteps
    this mismatch for the one query where it matters; every later query in this
    loop comes from a real env.step() and is self-consistent the normal way."""
    env.reset()
    obs = restore_full_state(env, snapshot, extra_state)

    active_displacements = copy.deepcopy(active_displacements_init)
    action_queue = deque()
    action_log = []
    t = start_step
    success = False
    first_query = True

    while t < max_steps:
        if len(action_queue) == 0:
            if first_query and obs_at_switch is not None:
                observation = obs_at_switch
            else:
                observation = prepare_observation(obs)
            first_query = False
            actions, _ = call_policy(model, observation, task_description)
            for a in actions[:CHUNK_SIZE]:
                action_queue.append(a)

        action = process_action(action_queue.popleft())

        apply_displacement_step(env, t, displacement_schedule, active_displacements,
                                 source_initial_z_map, obs, diagnostics)

        obs, reward, done, info = env.step(action.tolist())
        action_log.append((t, model, action.tolist()))

        for body_name in list(active_displacements.keys()):
            active_displacements[body_name]["remaining"] -= 1
            if active_displacements[body_name]["remaining"] <= 0:
                del active_displacements[body_name]

        t += 1

        if done:
            success = True
            break

    return {"success": success, "final_step": t, "action_log": action_log}


def run_one_switch(env, task_description, initial_state, manifest_event, switch_step):
    """Full Test-0 procedure for one (episode, t): Adapter 0->t, snapshot, OFT t->horizon.
    Returns a result dict with a 'category' field:
      'adapter_already_succeeded' | 'valid_candidate'
    (no 'unrecoverable_at_switch' category -- see module docstring)."""
    if switch_step >= MAX_STEPS:
        # t = horizon anchor: Adapter the whole way, OFT never called.
        prefix = run_prefix_to_step(env, task_description, initial_state, manifest_event,
                                     target_step=MAX_STEPS, log_prefix="adapter")
        if prefix["succeeded_by_target"]:
            return {"category": "adapter_only_result", "switch_step": switch_step,
                    "adapter_success_step": prefix["success_step"], "final_success": True,
                    "final_step": prefix["success_step"], "prefix_action_log": prefix["action_log"],
                    "oft_action_log": []}
        return {"category": "adapter_only_result", "switch_step": switch_step,
                "adapter_success_step": None, "final_success": False,
                "final_step": MAX_STEPS, "prefix_action_log": prefix["action_log"], "oft_action_log": []}

    prefix = run_prefix_to_step(env, task_description, initial_state, manifest_event,
                                 target_step=switch_step, log_prefix="adapter")

    if prefix["succeeded_by_target"]:
        return {"category": "adapter_already_succeeded", "switch_step": switch_step,
                "adapter_success_step": prefix["success_step"], "final_success": True,
                "final_step": prefix["success_step"], "prefix_action_log": prefix["action_log"],
                "oft_action_log": []}

    cont = run_continuation(
        env, task_description, prefix["snapshot"], prefix["reached_step"], "oft",
        prefix["displacement_schedule"], prefix["active_displacements"], prefix["source_initial_z_map"],
        diagnostics=prefix["diagnostics"], max_steps=MAX_STEPS, extra_state=prefix["extra_state"],
        obs_at_switch=prefix["obs_at_snapshot"],
    )

    return {
        "category": "valid_candidate", "switch_step": switch_step,
        "adapter_success_step": None, "final_success": cont["success"],
        "final_step": cont["final_step"],
        "prefix_action_log": prefix["action_log"], "oft_action_log": cont["action_log"],
        "diagnostics": prefix["diagnostics"],
    }


def get_env_and_task(task_id=TASK_ID):
    """task_id defaults to the module-level TASK_ID (6) so every existing
    caller keeps working unchanged; pass an explicit task_id to use this for
    other LIBERO-Long tasks (added for the full-benchmark sweep)."""
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_10"]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = get_libero_env(task, "openvla", resolution=256)
    return env, task_description, initial_states


def switch_grid(manifest_event, max_steps=MAX_STEPS, n_decile_points=11):
    """~10% deciles (always incl. 0 and max_steps) + perturbation-anchored points."""
    deciles = sorted(set(int(round(i * max_steps / (n_decile_points - 1))) for i in range(n_decile_points)))
    grid = {t: "decile" for t in deciles}
    if manifest_event is not None:
        trig = int(manifest_event["trigger_step"])
        dur = int(manifest_event["duration"])
        anchors = {
            "just_before_push": trig - 1,
            "just_after_push": trig + dur,
            "well_after_push": trig + dur + 40,
        }
        for label, t in anchors.items():
            t = max(0, min(max_steps, t))
            grid[t] = grid.get(t, label) if t in grid else label
    return sorted(grid.items())


_WORKDIR = Path(_os.environ.get(
    "ASYNC_MIX_VLA_WORKDIR", Path(__file__).resolve().parent.parent / "runs"
))
RESULTS_DIR = str(_WORKDIR / "test0_switch_timing")
MANIFEST_50TRIAL_DIR = str(_WORKDIR / "stage2_source_perturbation_50trial")
CLEAN_LABELS_PATH = str(_WORKDIR / "stage1_clean_control/clean_categories_full50.json")
PERTURBED_LABELS_CSV = f"{MANIFEST_50TRIAL_DIR}/stage2_50trial_per_episode.csv"


def load_manifest_event(condition, episode):
    if condition == "clean":
        return None
    with open(f"{MANIFEST_50TRIAL_DIR}/manifest_{condition}.json") as f:
        manifest = json.load(f)
    return manifest[f"{TASK_ID}_{episode}"]


def load_standalone_clean_labels():
    with open(CLEAN_LABELS_PATH) as f:
        d = json.load(f)
    return {int(k): v for k, v in d["adapter_success"].items()}, {int(k): v for k, v in d["oft_success"].items()}


def load_standalone_perturbed_labels():
    """Returns {(condition, episode): {"adapter": bool, "oft": bool}}."""
    import csv
    out = {}
    with open(PERTURBED_LABELS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["seed"], int(row["trial_index"]))
            out[key] = {"adapter": row["adapter_success"] == "True", "oft": row["oft_success"] == "True"}
    return out


def print_trace(result, episode, switch_step, condition):
    print("=" * 100)
    print(f"TRACE: episode={episode} condition={condition} switch_step={switch_step}")
    print("=" * 100)
    print(f"category = {result['category']}")
    prefix_log = result["prefix_action_log"]
    print(f"\n--- Adapter prefix: {len(prefix_log)} steps ---")
    for (t, src, a) in prefix_log:
        print(f"  t={t:>4} [{src:>7}] action={['%.3f' % x for x in a]}")
    if result["category"] == "adapter_already_succeeded":
        print(f"\n>>> Adapter SUCCEEDED at step {result['adapter_success_step']} "
              f"(before reaching switch_step={switch_step}) -- EXCLUDED from denominator, no OFT run.")
    elif result["category"] == "adapter_only_result":
        print(f"\n>>> switch_step >= horizon: Adapter-only result, OFT never called.")
        print(f">>> final_success={result['final_success']} final_step={result['final_step']}")
    else:
        print(f"\n>>> Snapshot taken at step {switch_step} (full sim state via env.get_sim_state()).")
        oft_log = result["oft_action_log"]
        print(f"\n--- OFT continuation: {len(oft_log)} steps ---")
        for (t, src, a) in oft_log:
            print(f"  t={t:>4} [{src:>7}] action={['%.3f' % x for x in a]}")
        print(f"\n>>> OUTCOME: final_success={result['final_success']} final_step={result['final_step']}")
    print("=" * 100)


def mode_trace_one(episode, switch_step, condition):
    env, task_description, initial_states = get_env_and_task()
    initial_state = initial_states[episode]
    manifest_event = load_manifest_event(condition, episode)
    if manifest_event is not None:
        print(f"Manifest event for episode {episode} / {condition}: {manifest_event}")
    if switch_step is None:
        switch_step = MAX_STEPS // 2
    t0 = time.perf_counter()
    result = run_one_switch(env, task_description, initial_state, manifest_event, switch_step)
    elapsed = time.perf_counter() - t0
    print_trace(result, episode, switch_step, condition)
    print(f"\nWall time: {elapsed:.1f}s")
    env.close()
    return result


def mode_validate(episodes, conditions=("clean", "seed101", "seed102")):
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    env, task_description, initial_states = get_env_and_task()
    clean_adapter, clean_oft = load_standalone_clean_labels()
    perturbed_labels = load_standalone_perturbed_labels()

    all_pass = True

    print("### CHECK 1: determinism (same episode/t/condition run twice) ###")
    det_episode, det_condition = episodes[0], conditions[-1]  # a perturbed condition exercises more code paths
    manifest_event = load_manifest_event(det_condition, det_episode)
    t_mid = MAX_STEPS // 2
    r1 = run_one_switch(env, task_description, initial_states[det_episode], manifest_event, t_mid)
    r2 = run_one_switch(env, task_description, initial_states[det_episode], manifest_event, t_mid)
    det_match = (
        r1["category"] == r2["category"]
        and r1["final_success"] == r2["final_success"]
        and r1["final_step"] == r2["final_step"]
        and [a for (_, _, a) in r1["prefix_action_log"]] == [a for (_, _, a) in r2["prefix_action_log"]]
        and [a for (_, _, a) in r1["oft_action_log"]] == [a for (_, _, a) in r2["oft_action_log"]]
    )
    print(f"  episode={det_episode} condition={det_condition} t={t_mid}: "
          f"run1 success={r1['final_success']} step={r1['final_step']} | "
          f"run2 success={r2['final_success']} step={r2['final_step']} -> "
          f"{'PASS (bit-identical)' if det_match else 'FAIL'}")
    all_pass &= det_match

    print(f"\n### CHECK 2: t=0 must reproduce standalone OFT-from-start numbers "
          f"(per-episode identity logged, {len(episodes)} episodes) ###")
    t0_identity = {}  # {condition: {episode_id(str): {"success": bool, "expected": bool, "match": bool}}}
    for condition in conditions:
        t0_identity[condition] = {}
        n_match, n_total = 0, 0
        for ep in episodes:
            manifest_event = load_manifest_event(condition, ep)
            r = run_one_switch(env, task_description, initial_states[ep], manifest_event, 0)
            expected = clean_oft[ep] if condition == "clean" else perturbed_labels[(condition, ep)]["oft"]
            match = (r["final_success"] == expected)
            n_total += 1
            n_match += int(match)
            t0_identity[condition][str(ep)] = {
                "success": r["final_success"], "expected": expected, "match": match,
                "final_step": r["final_step"],
            }
            flag = "" if match else "  <-- MISMATCH"
            print(f"  [{condition}] episode {ep:>2}: OFT-from-scratch success={r['final_success']!s:>5} "
                  f"(standalone={expected!s:>5}){flag}")
        ok = (n_match == n_total)
        all_pass &= ok
        solved = sorted(ep for ep, v in t0_identity[condition].items() if v["success"])
        print(f"  [{condition}] t=0 vs standalone OFT: {n_match}/{n_total} match -> {'PASS' if ok else 'FAIL'}")
        print(f"  [{condition}] OFT-from-scratch solves {len(solved)}/{n_total} episodes: {solved}")

    t0_path = f"{RESULTS_DIR}/t0_oft_from_scratch_identity.json"
    with open(t0_path, "w") as f:
        json.dump(t0_identity, f, indent=1)
    print(f"  Saved per-episode t=0 identity to {t0_path}")

    print(f"\n### CHECK 3: t=horizon must reproduce standalone Adapter numbers "
          f"(per-episode identity logged, {len(episodes)} episodes) ###")
    thorizon_identity = {}
    for condition in conditions:
        thorizon_identity[condition] = {}
        n_match, n_total = 0, 0
        for ep in episodes:
            manifest_event = load_manifest_event(condition, ep)
            r = run_one_switch(env, task_description, initial_states[ep], manifest_event, MAX_STEPS)
            expected = clean_adapter[ep] if condition == "clean" else perturbed_labels[(condition, ep)]["adapter"]
            match = (r["final_success"] == expected)
            n_total += 1
            n_match += int(match)
            thorizon_identity[condition][str(ep)] = {
                "success": r["final_success"], "expected": expected, "match": match,
                "final_step": r["final_step"],
            }
            if not match:
                print(f"  MISMATCH: condition={condition} ep={ep} t=horizon: "
                      f"test0={r['final_success']} standalone_adapter={expected}")
        ok = (n_match == n_total)
        all_pass &= ok
        print(f"  [{condition}] t=horizon vs standalone Adapter: {n_match}/{n_total} match -> {'PASS' if ok else 'FAIL'}")

    thorizon_path = f"{RESULTS_DIR}/thorizon_adapter_alone_identity.json"
    with open(thorizon_path, "w") as f:
        json.dump(thorizon_identity, f, indent=1)
    print(f"  Saved per-episode t=horizon identity to {thorizon_path}")

    env.close()
    print(f"\n{'=' * 60}\nVALIDATION {'PASSED' if all_pass else 'FAILED'}\n{'=' * 60}")
    return all_pass


def build_sweep_units(episodes, conditions=("clean", "seed101", "seed102")):
    """Returns [(condition, episode, grid), ...] -- one unit per (condition, episode),
    grid = switch_grid(manifest_event) for that specific episode/condition (perturbed
    conditions have per-episode-specific anchor points, so grid size varies)."""
    units = []
    for condition in conditions:
        for ep in episodes:
            manifest_event = load_manifest_event(condition, ep)
            grid = switch_grid(manifest_event)
            units.append((condition, ep, grid))
    return units


def mode_sweep(episodes=None, conditions=("clean", "seed101", "seed102"), pairs=None, out_suffix=""):
    """episodes/conditions: default dense sweep (all combos).
    pairs: optional explicit [[condition, episode], ...] list (for parallel sharding) --
    overrides episodes/conditions when given.
    out_suffix: appended to the output filename so parallel jobs don't clobber each other."""
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    env, task_description, initial_states = get_env_and_task()

    if pairs is not None:
        units = [(condition, ep, switch_grid(load_manifest_event(condition, ep))) for condition, ep in pairs]
    else:
        units = build_sweep_units(episodes if episodes is not None else list(range(50)), conditions)

    n_total_points = sum(len(grid) for (_, _, grid) in units)
    print(f"Sweep shard: {len(units)} (condition, episode) units, {n_total_points} total switch-points.", flush=True)

    all_records = []
    out_path = f"{RESULTS_DIR}/sweep_records{out_suffix}.pkl"
    done_points = 0
    t_start = time.perf_counter()

    for condition, ep, grid in units:
        manifest_event = load_manifest_event(condition, ep)
        for switch_step, point_label in grid:
            t0 = time.perf_counter()
            result = run_one_switch(env, task_description, initial_states[ep], manifest_event, switch_step)
            elapsed = time.perf_counter() - t0
            record = {
                "condition": condition, "episode": ep, "switch_step": switch_step,
                "point_label": point_label, "category": result["category"],
                "final_success": result["final_success"], "final_step": result["final_step"],
                "adapter_success_step": result.get("adapter_success_step"),
                "wall_time_s": elapsed,
            }
            all_records.append(record)
            done_points += 1
            total_elapsed = time.perf_counter() - t_start
            print(f"[{condition}] ep={ep:>2} t={switch_step:>4} ({point_label:>16}) "
                  f"cat={result['category']:>24} success={result['final_success']} "
                  f"step={result['final_step']:>4} ({elapsed:.1f}s) "
                  f"[{done_points}/{n_total_points}, {total_elapsed/60:.1f}min elapsed]", flush=True)
            with open(out_path, "wb") as f:
                pickle.dump(all_records, f)

    env.close()
    print(f"\nSaved {len(all_records)} records to {out_path}", flush=True)
    return all_records


def mode_diag_selfhandoff(episodes, k_values):
    """Diagnostic B: is snapshot/restore lossless, independent of cross-model
    switching? For each episode, within ONE server session: run OFT-from-scratch
    as a baseline (full trajectory), then for each k, run OFT-to-k -> snapshot ->
    restore -> continue-with-OFT, and compare the continuation's actions against
    the baseline's own actions from step k onward, EXACTLY (not just outcome).
    Since both sides use the same model in the same session, Gate 1 already
    established within-session determinism -- so if restore is lossless, the two
    should match bit-for-bit. Any divergence here is restore-specific, not
    cross-session-nondeterminism-specific (that's what diag_A / the OFT repro
    check on the side is for)."""
    env, task_description, initial_states = get_env_and_task()

    results = []
    for ep in episodes:
        print(f"\n=== episode {ep}: baseline OFT-from-scratch ===", flush=True)
        baseline = run_prefix_to_step(env, task_description, initial_states[ep], None,
                                       target_step=MAX_STEPS, log_prefix="oft")
        baseline_actions = [a for (_, _, a) in baseline["action_log"]]
        baseline_len = len(baseline_actions)
        baseline_success = baseline["succeeded_by_target"]
        print(f"  baseline: success={baseline_success} len={baseline_len}", flush=True)

        for k in k_values:
            if k >= baseline_len:
                print(f"  [ep={ep} k={k}] SKIP: baseline episode only {baseline_len} steps long", flush=True)
                results.append({"episode": ep, "k": k, "skipped": True, "reason": "k >= baseline_len"})
                continue

            prefix = run_prefix_to_step(env, task_description, initial_states[ep], None,
                                         target_step=k, log_prefix="oft")
            if prefix["succeeded_by_target"]:
                print(f"  [ep={ep} k={k}] SKIP: OFT alone already succeeded by step {k} "
                      f"(success_step={prefix['success_step']})", flush=True)
                results.append({"episode": ep, "k": k, "skipped": True, "reason": "succeeded_before_k"})
                continue

            cont = run_continuation(
                env, task_description, prefix["snapshot"], k, "oft",
                displacement_schedule={}, active_displacements_init={}, source_initial_z_map={},
                diagnostics=None, max_steps=MAX_STEPS, extra_state=prefix["extra_state"],
                obs_at_switch=prefix["obs_at_snapshot"],
            )
            cont_actions = [a for (_, _, a) in cont["action_log"]]
            baseline_tail = baseline_actions[k:]

            len_match = (len(cont_actions) == len(baseline_tail))
            first_divergence = None
            n_compare = min(len(cont_actions), len(baseline_tail))
            for i in range(n_compare):
                if cont_actions[i] != baseline_tail[i]:
                    first_divergence = k + i
                    break
            exact_match = len_match and (first_divergence is None)
            outcome_match = (cont["success"] == baseline_success)

            status = "MATCH (restore lossless)" if exact_match else "MISMATCH (restore lost something)"
            print(f"  [ep={ep} k={k}] continuation_len={len(cont_actions)} baseline_tail_len={len(baseline_tail)} "
                  f"first_divergence_step={first_divergence} outcome_match={outcome_match} -> {status}", flush=True)

            results.append({
                "episode": ep, "k": k, "skipped": False,
                "exact_match": exact_match, "len_match": len_match,
                "first_divergence_step": first_divergence, "outcome_match": outcome_match,
                "baseline_success": baseline_success, "continuation_success": cont["success"],
                "continuation_len": len(cont_actions), "baseline_tail_len": len(baseline_tail),
            })

    env.close()

    n_tested = sum(1 for r in results if not r["skipped"])
    n_match = sum(1 for r in results if not r["skipped"] and r["exact_match"])
    print(f"\n{'=' * 90}\nDIAGNOSTIC B SUMMARY: {n_match}/{n_tested} self-handoffs exactly matched baseline "
          f"({len(results) - n_tested} skipped)\n{'=' * 90}", flush=True)

    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = f"{RESULTS_DIR}/diag_selfhandoff_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved {out_path}", flush=True)
    return results


def mode_diag_obscompare(episodes, k_values):
    """Root-cause probe: at the exact moment of restore (before any new action is
    taken), compare (a) the observation (image/wrist_image/proprio state) OFT would
    receive, and (b) raw MuJoCo data fields not captured by env.get_sim_state()
    (qacc_warmstart, ctrl, qfrc_applied, xfrc_applied) between continuous execution
    and a restore of the exact same point. env.get_sim_state()/set_state_from_flattened()
    in this robosuite build only serialize (time, qpos, qvel) -- see
    binding_utils.py's MjSimState -- so qacc_warmstart/ctrl are NOT preserved across
    a restore. This directly tests whether that gap actually changes the observation
    (pointing at observation regen) and/or leaves the raw solver-adjacent fields
    different (pointing at physics/contact-solver state), rather than theorizing."""
    env, task_description, initial_states = get_env_and_task()

    results = []
    for ep in episodes:
        for k in k_values:
            print(f"\n=== episode {ep} k={k} ===", flush=True)
            prefix = run_prefix_to_step(env, task_description, initial_states[ep], None,
                                         target_step=k, log_prefix="oft")
            if prefix["succeeded_by_target"]:
                print(f"  SKIP: OFT alone already succeeded by step {k}", flush=True)
                results.append({"episode": ep, "k": k, "skipped": True})
                continue

            # Continuous-execution ground truth, captured WITHOUT any reset/restore.
            continuous_obs = prefix["obs_at_snapshot"]  # prepare_observation() right after k steps
            snapshot = prefix["snapshot"]  # env.get_sim_state().flatten() at that same instant

            # NOTE: env.sim is a property that returns a NEW MjSim object after every
            # hard reset (robosuite's ControlEnv.reset() -> _destroy_sim() calls
            # sim.free(); self.sim = None, then rebuilds it). Must re-fetch env.sim
            # fresh on each side of the reset rather than caching one reference --
            # caching produced a freed/stale MjSim (AttributeError: no attribute 'data').
            sim_before = env.sim
            continuous_qacc_warmstart = np.copy(sim_before.data.qacc_warmstart) if hasattr(sim_before.data, "qacc_warmstart") else None
            continuous_ctrl = np.copy(sim_before.data.ctrl) if hasattr(sim_before.data, "ctrl") else None
            continuous_qfrc_applied = np.copy(sim_before.data.qfrc_applied) if hasattr(sim_before.data, "qfrc_applied") else None
            continuous_xfrc_applied = np.copy(sim_before.data.xfrc_applied) if hasattr(sim_before.data, "xfrc_applied") else None
            continuous_qvel = np.copy(sim_before.data.qvel)
            continuous_qpos = np.copy(sim_before.data.qpos)

            # Now restore into the SAME env object (matching the real Test-0/Diagnostic-B path).
            # Uses restore_full_state() (qpos/qvel/time + qacc_warmstart/ctrl), not the
            # bare env.set_init_state() -- this mode now doubles as the fix-verification
            # probe: if qacc_warmstart/ctrl were the whole story, all fields below should
            # now match.
            env.reset()
            restored_obs_raw = restore_full_state(env, snapshot, prefix["extra_state"])
            restored_obs = prepare_observation(restored_obs_raw)

            sim_after = env.sim
            restored_qpos = np.copy(sim_after.data.qpos)
            restored_qvel = np.copy(sim_after.data.qvel)
            restored_qacc_warmstart = np.copy(sim_after.data.qacc_warmstart) if hasattr(sim_after.data, "qacc_warmstart") else None
            restored_ctrl = np.copy(sim_after.data.ctrl) if hasattr(sim_after.data, "ctrl") else None
            restored_qfrc_applied = np.copy(sim_after.data.qfrc_applied) if hasattr(sim_after.data, "qfrc_applied") else None
            restored_xfrc_applied = np.copy(sim_after.data.xfrc_applied) if hasattr(sim_after.data, "xfrc_applied") else None

            qpos_match = np.array_equal(continuous_qpos, restored_qpos)
            qvel_match = np.array_equal(continuous_qvel, restored_qvel)
            image_match = np.array_equal(continuous_obs["full_image"], restored_obs["full_image"])
            wrist_match = np.array_equal(continuous_obs["wrist_image"], restored_obs["wrist_image"])
            state_match = np.array_equal(continuous_obs["state"], restored_obs["state"])
            qacc_ws_match = (np.array_equal(continuous_qacc_warmstart, restored_qacc_warmstart)
                              if continuous_qacc_warmstart is not None else None)
            ctrl_match = (np.array_equal(continuous_ctrl, restored_ctrl)
                          if continuous_ctrl is not None else None)
            qfrc_applied_match = (np.array_equal(continuous_qfrc_applied, restored_qfrc_applied)
                                   if continuous_qfrc_applied is not None else None)
            xfrc_applied_match = (np.array_equal(continuous_xfrc_applied, restored_xfrc_applied)
                                   if continuous_xfrc_applied is not None else None)

            print(f"  qpos_match={qpos_match} qvel_match={qvel_match}", flush=True)
            print(f"  OBSERVATION: image_match={image_match} wrist_match={wrist_match} state_match={state_match}", flush=True)
            print(f"  RAW MJDATA (not in get_state()): qacc_warmstart_match={qacc_ws_match} "
                  f"ctrl_match={ctrl_match} qfrc_applied_match={qfrc_applied_match} "
                  f"xfrc_applied_match={xfrc_applied_match}", flush=True)
            if not state_match:
                diff = np.abs(np.asarray(continuous_obs["state"]) - np.asarray(restored_obs["state"]))
                print(f"  state vectors: continuous={continuous_obs['state']} restored={restored_obs['state']} "
                      f"max_abs_diff={diff.max():.8f}", flush=True)

            results.append({
                "episode": ep, "k": k, "skipped": False,
                "qpos_match": bool(qpos_match), "qvel_match": bool(qvel_match),
                "image_match": bool(image_match), "wrist_match": bool(wrist_match), "state_match": bool(state_match),
                "qacc_warmstart_match": qacc_ws_match if qacc_ws_match is None else bool(qacc_ws_match),
                "ctrl_match": ctrl_match if ctrl_match is None else bool(ctrl_match),
                "qfrc_applied_match": qfrc_applied_match if qfrc_applied_match is None else bool(qfrc_applied_match),
                "xfrc_applied_match": xfrc_applied_match if xfrc_applied_match is None else bool(xfrc_applied_match),
            })

    env.close()

    tested = [r for r in results if not r["skipped"]]
    n = len(tested)
    print(f"\n{'=' * 90}\nDIAGNOSTIC OBS-COMPARE SUMMARY (n={n})\n{'=' * 90}", flush=True)
    for field in ["qpos_match", "qvel_match", "image_match", "wrist_match", "state_match",
                  "qacc_warmstart_match", "ctrl_match", "qfrc_applied_match", "xfrc_applied_match"]:
        vals = [r[field] for r in tested if r[field] is not None]
        if vals:
            print(f"  {field}: {sum(vals)}/{len(vals)} matched", flush=True)
        else:
            print(f"  {field}: field not present on this mjData build", flush=True)

    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = f"{RESULTS_DIR}/diag_obscompare_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved {out_path}", flush=True)
    return results


def mode_diag_outcome_stability(episode_k_pairs):
    """Outcome-stability check (the bar that actually matters for the real sweep,
    per user decision after Diagnostic B: bit-exact self-handoff trajectory
    matching is the wrong bar, since the sweep only records success/failure
    outcomes, not trajectories -- and Diagnostic B's own non-chunk-aligned k
    failures were shown to be a test-design artifact, not a restore defect).

    For each (episode, k) with k a CHUNK_SIZE-aligned switch point: run OFT
    continuously to completion as a baseline (never restored), then run TWO
    INDEPENDENT restored continuations from the same (episode, k) snapshot
    (each: prefix 0->k, restore, continue k->horizon with the obs_at_switch
    fix) to completion. Compare final success/failure only across all three
    (baseline, restore #1, restore #2) -- not action-by-action trajectories.
    If outcomes are always stable despite the known small step-208-style
    numerical divergence, that divergence is benign for the sweep's purposes.
    If any outcome flips, the divergence is large enough to matter and the
    qacc_warmstart lead needs to be chased before running the sweep."""
    env, task_description, initial_states = get_env_and_task()

    for _, k in episode_k_pairs:
        assert k % CHUNK_SIZE == 0, f"k={k} must be chunk-aligned (multiple of {CHUNK_SIZE})"

    # Group by episode so the (expensive, up-to-520-step) baseline rollout is
    # computed ONCE per episode and reused across all its k's, instead of being
    # redundantly recomputed per (episode, k) pair -- OFT is deterministic
    # (Diagnostic A), so re-running the same episode's baseline would just
    # waste compute reproducing the identical result.
    from collections import OrderedDict
    pairs_by_episode = OrderedDict()
    for ep, k in episode_k_pairs:
        pairs_by_episode.setdefault(ep, []).append(k)

    results = []
    for ep, k_values in pairs_by_episode.items():
        print(f"\n=== episode {ep}: baseline OFT-from-scratch (continuous, never restored) ===", flush=True)
        baseline = run_prefix_to_step(env, task_description, initial_states[ep], None,
                                       target_step=MAX_STEPS, log_prefix="oft")
        baseline_success = baseline["succeeded_by_target"]
        baseline_step = baseline.get("success_step")
        print(f"  baseline: success={baseline_success} step={baseline_step}", flush=True)

        for k in k_values:
            prefix = run_prefix_to_step(env, task_description, initial_states[ep], None,
                                         target_step=k, log_prefix="oft")
            if prefix["succeeded_by_target"]:
                print(f"  [ep={ep} k={k}] SKIP: OFT alone already succeeded by step {k} "
                      f"(success_step={prefix['success_step']})", flush=True)
                results.append({"episode": ep, "k": k, "skipped": True, "reason": "succeeded_before_k"})
                continue

            run_outcomes = []
            for run_idx in (1, 2):
                cont = run_continuation(
                    env, task_description, prefix["snapshot"], k, "oft",
                    displacement_schedule={}, active_displacements_init={}, source_initial_z_map={},
                    diagnostics=None, max_steps=MAX_STEPS, extra_state=prefix["extra_state"],
                    obs_at_switch=prefix["obs_at_snapshot"],
                )
                print(f"  [ep={ep} k={k}] restore #{run_idx}: success={cont['success']} "
                      f"final_step={cont['final_step']}", flush=True)
                run_outcomes.append({"success": cont["success"], "final_step": cont["final_step"]})

            outcomes_agree = (
                baseline_success == run_outcomes[0]["success"] == run_outcomes[1]["success"]
            )
            status = "STABLE" if outcomes_agree else "FLIPPED (outcome-level divergence)"
            print(f"  [ep={ep} k={k}] baseline={baseline_success} restore#1={run_outcomes[0]['success']} "
                  f"restore#2={run_outcomes[1]['success']} -> {status}", flush=True)

            results.append({
                "episode": ep, "k": k, "skipped": False,
                "baseline_success": baseline_success, "baseline_step": baseline_step,
                "restore1_success": run_outcomes[0]["success"], "restore1_step": run_outcomes[0]["final_step"],
                "restore2_success": run_outcomes[1]["success"], "restore2_step": run_outcomes[1]["final_step"],
                "outcomes_agree": outcomes_agree,
            })

    env.close()

    tested = [r for r in results if not r["skipped"]]
    n_stable = sum(1 for r in tested if r["outcomes_agree"])
    print(f"\n{'=' * 90}\nOUTCOME-STABILITY SUMMARY: {n_stable}/{len(tested)} cases had stable outcomes "
          f"({len(results) - len(tested)} skipped)\n{'=' * 90}", flush=True)

    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = f"{RESULTS_DIR}/diag_outcome_stability_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved {out_path}", flush=True)
    return results


def mode_diag_trajectory_diff(episode, k, n_steps):
    """Per user pushback on warmup-replay (2026-08-06): an 89% outcome flip
    rate is too large to be explained by a ~0.003-magnitude numerical
    perturbation from cold-solver noise -- that scale of collapse (turning 8
    of 9 winnable episodes into failures, including one that flipped only 8
    steps before its own natural success) looks like systematic corruption,
    not noise. Before touching anything, check directly: does the state
    divergence between the restored continuation and continuous execution
    stay bounded (~0.003, consistent with noise) or GROW over time
    (consistent with something like lost controller-internal state -- e.g.
    OSC_POSE's goal/interpolation buffer living in the Python controller
    object, not in MjData, so restore silently starts it fresh)? And do
    OFT's own predicted actions ever diverge, or only the resulting states
    (which would follow trivially once actions differ)?

    Logs (state, action-about-to-be-executed) at every real step from k to
    k+n_steps-1, on BOTH a continuous OFT-from-scratch baseline and the
    restored continuation (prefix 0->k, restore with the obs_at_switch fix,
    continue), then diffs them step by step. Only run on ONE episode/k at a
    time -- this is a targeted trace, not a scaled sweep."""
    assert k % CHUNK_SIZE == 0, f"k={k} must be chunk-aligned (multiple of {CHUNK_SIZE})"
    env, task_description, initial_states = get_env_and_task()

    def rollout_logging(log_from_step, log_until_step):
        env.reset()
        obs = env.set_init_state(initial_states[episode])
        for _ in range(NUM_STEPS_WAIT):
            obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
        action_queue = deque()
        log = []
        t = 0
        success, success_step = False, None
        while t < log_until_step:
            if len(action_queue) == 0:
                observation = prepare_observation(obs)
                actions, _ = call_policy("oft", observation, task_description)
                for a in actions[:CHUNK_SIZE]:
                    action_queue.append(a)
            current_state = np.asarray(prepare_observation(obs)["state"]).tolist()
            action = process_action(action_queue.popleft())
            if t >= log_from_step:
                log.append({"t": t, "state": current_state, "action": action.tolist()})
            obs, reward, done, info = env.step(action.tolist())
            t += 1
            if done:
                success, success_step = True, t
                break
        return {"log": log, "success": success, "success_step": success_step, "final_step": t}

    print(f"=== Continuous baseline: episode {episode}, logging steps {k}..{k + n_steps - 1} ===", flush=True)
    baseline = rollout_logging(log_from_step=k, log_until_step=min(k + n_steps, MAX_STEPS))
    print(f"  baseline: success={baseline['success']} success_step={baseline['success_step']} "
          f"final_step={baseline['final_step']}", flush=True)

    print(f"\n=== Restored continuation: episode {episode}, prefix 0->{k}, then logging from {k} ===", flush=True)
    prefix = run_prefix_to_step(env, task_description, initial_states[episode], None,
                                 target_step=k, log_prefix="oft")
    if prefix["succeeded_by_target"]:
        print("  Prefix alone already succeeded before k -- nothing to compare.", flush=True)
        env.close()
        return None

    env.reset()
    obs = restore_full_state(env, prefix["snapshot"], prefix["extra_state"])
    action_queue = deque()
    restored_log = []
    t = k
    first_query = True
    success, success_step = False, None
    end_t = min(k + n_steps, MAX_STEPS)
    while t < end_t:
        if len(action_queue) == 0:
            if first_query and prefix["obs_at_snapshot"] is not None:
                observation = prefix["obs_at_snapshot"]
            else:
                observation = prepare_observation(obs)
            first_query = False
            actions, _ = call_policy("oft", observation, task_description)
            for a in actions[:CHUNK_SIZE]:
                action_queue.append(a)
        current_state = np.asarray(prepare_observation(obs)["state"]).tolist()
        action = process_action(action_queue.popleft())
        restored_log.append({"t": t, "state": current_state, "action": action.tolist()})
        obs, reward, done, info = env.step(action.tolist())
        t += 1
        if done:
            success, success_step = True, t
            break
    print(f"  restored: success={success} success_step={success_step} final_step={t}", flush=True)

    env.close()

    baseline_by_t = {e["t"]: e for e in baseline["log"]}
    restored_by_t = {e["t"]: e for e in restored_log}
    common_ts = sorted(set(baseline_by_t) & set(restored_by_t))

    print(f"\n{'=' * 90}\nSTEP-BY-STEP DIFF (episode={episode}, k={k})\n{'=' * 90}", flush=True)
    print(f"{'step':>6} {'action_match':>13} {'state_max_diff':>16}", flush=True)
    first_action_divergence = None
    diff_trace = []
    for t in common_ts:
        b, r = baseline_by_t[t], restored_by_t[t]
        action_match = (b["action"] == r["action"])
        state_diff = max(abs(x - y) for x, y in zip(b["state"], r["state"]))
        diff_trace.append({"t": t, "action_match": action_match, "state_max_diff": state_diff})
        print(f"{t:>6} {str(action_match):>13} {state_diff:>16.8f}", flush=True)
        if not action_match and first_action_divergence is None:
            first_action_divergence = t

    growth = None
    if len(diff_trace) >= 2:
        early = np.mean([d["state_max_diff"] for d in diff_trace[:max(1, len(diff_trace) // 4)]])
        late = np.mean([d["state_max_diff"] for d in diff_trace[-max(1, len(diff_trace) // 4):]])
        growth = late / early if early > 0 else float("inf")

    print(f"\n{'=' * 90}\nSUMMARY: first_action_divergence_step={first_action_divergence}  "
          f"early_vs_late_state_diff_growth={growth}\n{'=' * 90}", flush=True)

    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = f"{RESULTS_DIR}/diag_trajectory_diff_ep{episode}_k{k}.json"
    with open(out_path, "w") as f:
        json.dump({
            "episode": episode, "k": k,
            "baseline_success": baseline["success"], "baseline_success_step": baseline["success_step"],
            "restored_success": success, "restored_success_step": success_step,
            "first_action_divergence_step": first_action_divergence,
            "early_vs_late_state_diff_growth": growth,
            "diff_trace": diff_trace,
        }, f, indent=1)
    print(f"Saved {out_path}", flush=True)
    return diff_trace


def _obs_fingerprint(observation):
    """Content hash of the exact observation dict fed to call_policy(), so two
    observations can be compared for exact equality without dumping full image
    arrays into logs/JSON."""
    full_image = np.asarray(observation["full_image"])
    wrist_image = np.asarray(observation["wrist_image"])
    state = np.asarray(observation["state"])
    return {
        "full_image_hash": hashlib.md5(full_image.tobytes()).hexdigest(),
        "wrist_image_hash": hashlib.md5(wrist_image.tobytes()).hexdigest(),
        "full_image_shape": list(full_image.shape),
        "state": state.tolist(),
    }


def mode_diag_query_input_diff(episode, k, n_queries):
    """Per user directive (2026-08-06): the controller-cache fix closed the
    FIRST-chunk divergence (episode 1/k=200 now exact-matches) but the
    dominant divergence still appears at exactly the query-2 boundary
    (step k+CHUNK_SIZE) across most cases. Rather than guess at a second
    mechanism, diff the INPUTS to query 2 directly: the exact observation
    (image/wrist_image/proprio) fed to call_policy(), on both continuous and
    restored paths, at every query boundary (not just every step). Also
    confirms/refutes, by direct code inspection already done and reported
    separately: the OFT server (serve_oft_libero10.py) is stateless per
    request (POLICY dict loaded once, get_vla_action() is a pure function of
    (obs, task_label), normalization stats are static/precomputed, no
    KV-cache or temporal-ensembling buffer anywhere in the path) -- so any
    difference found here is a difference in the ACTUAL observation content,
    not a hidden history/context buffer on the model/server side.

    Logs, at EVERY query boundary (query_index 0, 1, 2, ...) on both the
    continuous baseline and the restored continuation: the query's step
    index t, an md5 fingerprint of full_image/wrist_image, and the exact
    proprio state vector. Diffs query-by-query. n_queries counts queries
    from the switch point k onward (query 0 is the first post-switch query
    at step k; query 1 is the next one, at step k+CHUNK_SIZE, assuming no
    success in between)."""
    assert k % CHUNK_SIZE == 0, f"k={k} must be chunk-aligned (multiple of {CHUNK_SIZE})"
    env, task_description, initial_states = get_env_and_task()

    def rollout_logging_queries(log_from_step, max_queries_after_log_from_step):
        env.reset()
        obs = env.set_init_state(initial_states[episode])
        for _ in range(NUM_STEPS_WAIT):
            obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
        action_queue = deque()
        query_log = []
        t = 0
        success, success_step = False, None
        n_logged_queries = 0
        while True:
            if len(action_queue) == 0:
                observation = prepare_observation(obs)
                if t >= log_from_step:
                    fp = _obs_fingerprint(observation)
                    fp["query_index"] = n_logged_queries
                    fp["t"] = t
                    query_log.append(fp)
                    n_logged_queries += 1
                    if n_logged_queries >= max_queries_after_log_from_step:
                        return {"query_log": query_log, "success": success, "success_step": success_step}
                actions, _ = call_policy("oft", observation, task_description)
                for a in actions[:CHUNK_SIZE]:
                    action_queue.append(a)
            action = process_action(action_queue.popleft())
            obs, reward, done, info = env.step(action.tolist())
            t += 1
            if done or t >= MAX_STEPS:
                success, success_step = done, (t if done else None)
                return {"query_log": query_log, "success": success, "success_step": success_step}

    print(f"=== Continuous baseline: episode {episode}, logging {n_queries} queries from step {k} ===", flush=True)
    baseline = rollout_logging_queries(log_from_step=k, max_queries_after_log_from_step=n_queries)
    print(f"  baseline queries logged: {len(baseline['query_log'])}  "
          f"success={baseline['success']} success_step={baseline['success_step']}", flush=True)
    for e in baseline["query_log"]:
        print(f"  [continuous] query={e['query_index']} t={e['t']} "
              f"full_image_hash={e['full_image_hash'][:12]} wrist_image_hash={e['wrist_image_hash'][:12]} "
              f"state={['%.5f' % x for x in e['state']]}", flush=True)

    print(f"\n=== Restored continuation: episode {episode}, prefix 0->{k}, logging {n_queries} queries ===", flush=True)
    prefix = run_prefix_to_step(env, task_description, initial_states[episode], None,
                                 target_step=k, log_prefix="oft")
    if prefix["succeeded_by_target"]:
        print("  Prefix alone already succeeded before k -- nothing to compare.", flush=True)
        env.close()
        return None

    env.reset()
    obs = restore_full_state(env, prefix["snapshot"], prefix["extra_state"])
    action_queue = deque()
    restored_query_log = []
    t = k
    first_query = True
    success, success_step = False, None
    n_logged_queries = 0
    while n_logged_queries < n_queries and t < MAX_STEPS:
        if len(action_queue) == 0:
            if first_query and prefix["obs_at_snapshot"] is not None:
                observation = prefix["obs_at_snapshot"]
            else:
                observation = prepare_observation(obs)
            first_query = False
            fp = _obs_fingerprint(observation)
            fp["query_index"] = n_logged_queries
            fp["t"] = t
            restored_query_log.append(fp)
            n_logged_queries += 1
            actions, _ = call_policy("oft", observation, task_description)
            for a in actions[:CHUNK_SIZE]:
                action_queue.append(a)
        action = process_action(action_queue.popleft())
        obs, reward, done, info = env.step(action.tolist())
        t += 1
        if done:
            success, success_step = True, t
            break
    print(f"  restored: success={success} success_step={success_step} final_step={t}", flush=True)
    for e in restored_query_log:
        print(f"  [restored]   query={e['query_index']} t={e['t']} "
              f"full_image_hash={e['full_image_hash'][:12]} wrist_image_hash={e['wrist_image_hash'][:12]} "
              f"state={['%.5f' % x for x in e['state']]}", flush=True)

    env.close()

    print(f"\n{'=' * 90}\nQUERY-BY-QUERY INPUT DIFF (episode={episode}, k={k})\n{'=' * 90}", flush=True)
    n_common = min(len(baseline["query_log"]), len(restored_query_log))
    diffs = []
    for i in range(n_common):
        b, r = baseline["query_log"][i], restored_query_log[i]
        t_match = (b["t"] == r["t"])
        image_match = (b["full_image_hash"] == r["full_image_hash"])
        wrist_match = (b["wrist_image_hash"] == r["wrist_image_hash"])
        state_match = (b["state"] == r["state"])
        state_max_diff = max(abs(x - y) for x, y in zip(b["state"], r["state"]))
        print(f"  query={i} baseline_t={b['t']} restored_t={r['t']} t_match={t_match} "
              f"image_match={image_match} wrist_match={wrist_match} state_match={state_match} "
              f"state_max_diff={state_max_diff:.8f}", flush=True)
        diffs.append({
            "query_index": i, "baseline_t": b["t"], "restored_t": r["t"], "t_match": t_match,
            "image_match": image_match, "wrist_match": wrist_match, "state_match": state_match,
            "state_max_diff": state_max_diff,
        })

    first_mismatch = next((d["query_index"] for d in diffs if not (d["image_match"] and d["wrist_match"] and d["state_match"])), None)
    print(f"\nFirst query with ANY input mismatch: {first_mismatch}", flush=True)

    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = f"{RESULTS_DIR}/diag_query_input_diff_ep{episode}_k{k}.json"
    with open(out_path, "w") as f:
        json.dump({"episode": episode, "k": k, "diffs": diffs, "first_mismatch_query": first_mismatch}, f, indent=1)
    print(f"Saved {out_path}", flush=True)
    return diffs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["trace_one", "validate", "sweep", "diag_selfhandoff", "diag_obscompare", "diag_outcome_stability", "diag_trajectory_diff", "diag_query_input_diff"], required=True)
    parser.add_argument("--diag_n_steps", type=int, default=280,
                         help="For diag_trajectory_diff: number of real steps from k to log/compare.")
    parser.add_argument("--diag_n_queries", type=int, default=3,
                         help="For diag_query_input_diff: number of policy queries from k to log/compare.")
    parser.add_argument("--diag_episodes", type=int, nargs="+", default=[10, 14, 23, 27, 35, 0, 1])
    parser.add_argument("--diag_k_values", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--diag_ek_pairs", type=str, default=None,
                         help="JSON string [[episode, k], ...] for diag_outcome_stability, e.g. "
                              "'[[10,80],[14,160],[23,240],[27,320],[35,400]]'.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--switch_step", type=int, default=None)
    parser.add_argument("--condition", choices=["clean", "seed101", "seed102"], default="clean")
    parser.add_argument("--validate_episodes", type=int, nargs="+", default=list(range(50)))
    parser.add_argument("--sweep_episodes", type=int, nargs="+", default=list(range(50)))
    parser.add_argument("--sweep_pairs_file", type=str, default=None,
                         help="JSON file with [[condition, episode], ...] -- for parallel sharding, "
                              "overrides --sweep_episodes/dense conditions.")
    parser.add_argument("--sweep_out_suffix", type=str, default="",
                         help="Appended to sweep_records<suffix>.pkl so parallel shards don't collide.")
    args = parser.parse_args()

    print(f"run_test0_switch_timing.py mode={args.mode}. "
          f"MAX_STEPS={MAX_STEPS}, CHUNK_SIZE={CHUNK_SIZE}, TASK_ID={TASK_ID}.", flush=True)

    if args.mode == "trace_one":
        mode_trace_one(args.episode, args.switch_step, args.condition)
    elif args.mode == "validate":
        ok = mode_validate(args.validate_episodes)
        if not ok:
            raise SystemExit("Validation FAILED -- stopping, not proceeding to sweep.")
    elif args.mode == "sweep":
        if args.sweep_pairs_file:
            with open(args.sweep_pairs_file) as f:
                pairs = json.load(f)
            mode_sweep(pairs=pairs, out_suffix=args.sweep_out_suffix)
        else:
            mode_sweep(episodes=args.sweep_episodes, out_suffix=args.sweep_out_suffix)
    elif args.mode == "diag_selfhandoff":
        mode_diag_selfhandoff(args.diag_episodes, args.diag_k_values)
    elif args.mode == "diag_obscompare":
        mode_diag_obscompare(args.diag_episodes, args.diag_k_values)
    elif args.mode == "diag_outcome_stability":
        pairs = json.loads(args.diag_ek_pairs) if args.diag_ek_pairs else \
            [[10, 80], [14, 160], [23, 240], [27, 320], [35, 400]]
        mode_diag_outcome_stability([tuple(p) for p in pairs])
    elif args.mode == "diag_trajectory_diff":
        switch_step = args.switch_step if args.switch_step is not None else 240
        mode_diag_trajectory_diff(args.episode, switch_step, args.diag_n_steps)
    elif args.mode == "diag_query_input_diff":
        switch_step = args.switch_step if args.switch_step is not None else 200
        mode_diag_query_input_diff(args.episode, switch_step, args.diag_n_queries)
