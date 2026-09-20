"""Canonical live AsyncMixVLA runner -- a genuine end-to-end state machine,
not an evaluation script with pre-known episode outcomes.

    ADAPTER -> ASYNC_OFT_PENDING -> OFT -> DONE

Adapter controls the robot normally, generating action chunks and executing
them one action at a time. At every valid control step (chunk_position > 0,
i.e. once at least one action has actually executed, matching the
already-validated trigger-evaluation convention), the configured trigger is
evaluated. Until it fires, Adapter behavior is completely unchanged from a
plain Adapter-only rollout. The first time the trigger returns
should_switch=True, the decision is frozen (see FrozenTrigger below): the
remaining actions of the CURRENT chunk that were already generated but not
yet executed become the bridge, executed for real while the async Full OFT
request runs in parallel (or, for sync_full_oft, after the chunk finishes).
Once OFT's first action is issued, control passes to OFT permanently -- no
Adapter<->OFT oscillation, no re-triggering.

Modes: adapter_only, sync_full_oft, naive_async_full_oft, vlash_async_full_oft.
Trigger backends: never, oracle_onset (debug), learned (not implemented yet).

Does NOT modify results/table1_complementarity/ or results/table2b_outcome_oracle/
-- those are frozen evaluation baselines, read from (oracle_manifest.json)
but never written to.
"""
import argparse
import json
import time
from pathlib import Path

from experiments.robot.libero.env_perturbations import get_control_dt
from experiments.robot.libero.libero_utils import get_libero_dummy_action
from measure_async_handoff_latency import warm_up
from asyncmixvla.runtime_support import continue_with_oft_until_done, make_displacement_stepper
from run_mixed_libero10 import get_source_initial_z_map, schedule_from_manifest_event
from run_test0_switch_timing import (
    CHUNK_SIZE, MAX_STEPS, NUM_STEPS_WAIT, call_policy, get_env_and_task, prepare_observation, process_action,
)

from asyncmixvla.bridge import run_async_bridge, run_sync_handoff
from asyncmixvla.trigger import (
    ForcedStepTrigger,
    NeverTrigger,
    OracleOnsetTrigger,
    TriggerResult,
)

MECHANISM_FOR_MODE = {
    "sync_full_oft": "sync",
    "naive_async_full_oft": "naive_async",
    "vlash_async_full_oft": "vlash_async",
}


class FrozenTrigger:
    """Wraps any Trigger and enforces "exactly one Adapter->OFT escalation":
    once the wrapped trigger has returned should_switch=True once, every
    subsequent call short-circuits to should_switch=False without even
    calling the wrapped trigger again. This is the mechanism-independent
    guarantee that no backend (oracle, learned, or a future one) can cause
    Adapter<->OFT oscillation, enforced at the state-machine boundary
    rather than trusted to each backend's own internal bookkeeping."""

    def __init__(self, inner):
        self.inner = inner
        self.fired = False

    def update(self, **kwargs) -> TriggerResult:
        if self.fired:
            return TriggerResult(should_switch=False, trigger_metadata={"frozen": True})
        result = self.inner.update(**kwargs)
        if result.should_switch:
            self.fired = True
        return result


def build_trigger(args, env=None, manifest_event=None):
    if args.trigger == "never":
        return FrozenTrigger(NeverTrigger())
    if args.trigger == "forced_step":
        if args.forced_trigger_step is None:
            raise ValueError("--trigger forced_step requires --forced_trigger_step")
        return FrozenTrigger(ForcedStepTrigger(args.forced_trigger_step))
    if args.trigger == "oracle_onset":
        return FrozenTrigger(OracleOnsetTrigger(args.oracle_manifest, args.task_id, args.trial_id, args.condition))
    if args.trigger == "observable_cascade":
        if not getattr(args, "observable_cascade_checkpoint", None):
            raise ValueError("--trigger observable_cascade requires --observable_cascade_checkpoint")
        from observable_cascade_v1.trigger import ObservableCascadeTrigger
        return FrozenTrigger(ObservableCascadeTrigger(args.observable_cascade_checkpoint))
    if args.trigger == "observable_cascade_continuous":
        # Re-arming variant: V1 offers a fresh candidate after each decline instead
        # of latching after its first (only) candidate. Reuses the fitted V2 model
        # as-is; only the candidate-arming policy differs from "observable_cascade".
        if not getattr(args, "observable_cascade_checkpoint", None):
            raise ValueError("--trigger observable_cascade_continuous requires --observable_cascade_checkpoint")
        from observable_cascade_v1.continuous_trigger import ContinuousObservableCascadeTrigger
        return FrozenTrigger(ContinuousObservableCascadeTrigger(args.observable_cascade_checkpoint))
    if args.trigger == "visual_gate":
        # Non-privileged camera gate. Same observation boundary as "deployable":
        # no env, no task id, no manifest -- only the prepared camera/proprio dict.
        if not getattr(args, "visual_gate_checkpoint", None):
            raise ValueError("--trigger visual_gate requires --visual_gate_checkpoint")
        from visual_trigger_v1.trigger import VisualGateTrigger
        return FrozenTrigger(VisualGateTrigger(args.visual_gate_checkpoint))
    if args.trigger == "deployable":
        # The deployable detector receives its checkpoint only. In particular,
        # do not pass the simulator, task ID, or perturbation manifest into its
        # constructor; its observation boundary filters decision inputs.
        if not getattr(args, "deployable_checkpoint", None):
            raise ValueError("--trigger deployable requires --deployable_checkpoint")
        from deployable_trigger_v1.trigger import DeployableTrigger
        return FrozenTrigger(DeployableTrigger(args.deployable_checkpoint))
    raise ValueError(f"unknown trigger backend: {args.trigger}")


def setup_perturbation(task_id, trial_id, condition, manifest_path, env, obs):
    """Only called for condition="perturbed" (see run_episode). Returns
    (displacement_hook, diagnostics, obs_holder) -- displacement_hook(episode_step)
    must be called before every env.step(), matching the pre_step_hook
    contract asyncmixvla/bridge.py's bridge functions expect."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    entry = manifest[str(trial_id)]
    if not entry["feasible"]:
        raise ValueError(f"task{task_id} trial{trial_id}: manifest entry not feasible for perturbed condition")
    event = entry["event"]
    displacement_schedule = schedule_from_manifest_event(event)
    active_displacements = {}
    source_initial_z_map = get_source_initial_z_map(env, task_id)
    diagnostics = {"body_name": event["body_name"], "fired": False, "cancelled_early": False,
                    "never_reached": True, "position_before": None, "position_after": None}
    obs_holder = {"obs": obs}
    hook = make_displacement_stepper(env, displacement_schedule, active_displacements,
                                      source_initial_z_map, diagnostics, obs_holder, task_id)
    return hook, diagnostics, obs_holder, event


class _EpisodeTimer:
    """Wraps env.step to capture true end-to-end wall-clock completion time
    with a monotonic high-resolution timer (perf_counter_ns -- monotonic,
    unaffected by system clock adjustments, ns resolution on Linux; the
    stdlib-recommended choice for interval timing, equivalent to
    monotonic_ns() for this purpose). Always on (no --debug_trace_out
    dependency): episode_start_ns is stamped once, right before the real
    task loop begins (same instant as `origin` below); episode_end_ns is
    stamped the FIRST time any env.step() call -- from the Adapter loop, the
    bridge/handoff functions in asyncmixvla/bridge.py, the OFT continuation
    loop, or the oft_only loop -- returns done=True (LIBERO's success
    signal in this codebase; there is no separate timeout-done). Because
    this wraps the single shared env object, every caller's env.step() goes
    through it, so the elapsed wall time between the two timestamps
    necessarily includes ALL policy inference, trigger computation, OFT
    async compute, handoff waiting, and environment stepping in between --
    nothing needs to be separately instrumented."""

    def __init__(self, env):
        self.env = env
        self.orig_step = env.step
        self.start_ns = None
        self.end_ns = None

    def start(self):
        self.start_ns = time.perf_counter_ns()

    def wrapped(self, action):
        obs, rew, done, info = self.orig_step(action)
        if done and self.end_ns is None:
            self.end_ns = time.perf_counter_ns()
        return obs, rew, done, info

    def install(self):
        self.env.step = self.wrapped


class _DebugStepLogger:
    """TEMPORARY debug instrumentation (opt-in via --debug_trace_out), for
    diffing this runtime's exact step-by-step trajectory against the
    frozen Table 2B reference's (see debug_trace_compare.py). No effect on
    default behavior; not part of the production AsyncMixVLA path."""

    def __init__(self, env, body_id, extra_bodies=None):
        self.env = env
        self.body_id = body_id
        self.orig_step = env.step
        self.records = []
        # Optional: also log task-relevant object positions each step (used by
        # characterize_stuck_handoffs.py to tell "OFT redoes a finished subtask"
        # apart from "the scene is in a state OFT can't handle").
        self.extra_bodies = list(extra_bodies or [])

    def _body_pos(self):
        if self.body_id is None:
            return None
        return self.env.sim.data.body_xpos[self.body_id].copy().tolist()

    def wrapped(self, action):
        pre_pos = self._body_pos()
        obs, rew, done, info = self.orig_step(action)
        post_pos = self._body_pos()
        self.records.append({
            "step": len(self.records), "action": [float(x) for x in action],
            "perturbed_body_pos_pre": pre_pos, "perturbed_body_pos_post": post_pos,
            "eef_pos": [float(x) for x in obs["robot0_eef_pos"]],
            "eef_quat": [float(x) for x in obs["robot0_eef_quat"]],
            "gripper_qpos": [float(x) for x in obs["robot0_gripper_qpos"]],
            "bodies": self._bodies(),
            "done": bool(done),
        })
        return obs, rew, done, info

    def _bodies(self):
        out = {}
        for b in self.extra_bodies:
            try:
                bid = self.env.sim.model.body_name2id(b)
                out[b] = self.env.sim.data.body_xpos[bid].copy().tolist()
            except Exception:
                pass
        return out

    def install(self):
        self.env.step = self.wrapped


def run_episode(args):
    recorder = getattr(args, "observable_recorder", None)
    if recorder is not None and (args.mode != "adapter_only" or args.trigger != "never"):
        raise ValueError("observable_recorder requires mode=adapter_only and trigger=never")
    env, task_description, initial_states = get_env_and_task(task_id=args.task_id)
    initial_state = initial_states[args.trial_id]

    debug_logger = None
    if getattr(args, "debug_trace_out", None):
        body_id = None
        if args.condition == "perturbed":
            with open(args.manifest_path) as f:
                _m = json.load(f)
            try:
                body_id = env.sim.model.body_name2id(_m[str(args.trial_id)]["event"]["body_name"])
            except Exception:
                body_id = None
        from run_mixed_libero10 import LIBERO_10_SOURCE_TARGET as _ST
        _info = _ST.get(args.task_id, {})
        debug_logger = _DebugStepLogger(
            env, body_id,
            extra_bodies=list(_info.get("source_objects", [])) + list(_info.get("target_objects", [])))
        debug_logger.install()

    if args.warmup > 0:
        # Every timing-sensitive script in this project warms up both model
        # servers before the real timed run (measure_async_handoff_latency.py,
        # test2_async_bridge_length.py, run_seamless_handoff.py, ...) -- the
        # first-ever inference call to a freshly-started server pays a real
        # CUDA/JIT cold-start cost that has nothing to do with steady-state
        # async latency. Skipping this here caused Smoke B's first run to
        # show C_async=2.03s (vs. the ~0.15-0.3s seen everywhere else in this
        # project) and a false final_success=False that didn't match Table 2B's
        # known result for that exact episode -- a real, now-fixed bug, not a
        # cosmetic one. warm_up() does its own reset/set_init_state; the real
        # episode below re-initializes from scratch afterward.
        warm_up(env, task_description, initial_state, args.warmup)

    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(NUM_STEPS_WAIT):
        obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))

    displacement_hook, diagnostics, obs_holder, manifest_event = (None, None, None, None)
    zeroshot_family = getattr(args, "zeroshot_family", None)
    if zeroshot_family:
        # Novel perturbation family (rotation / target displacement / post-grasp
        # slip). Only the disturbance changes -- trigger, bridge and handoff run
        # exactly as frozen. manifest_event stays None on purpose (see
        # setup_zeroshot_perturbation docstring).
        from asyncmixvla.zeroshot_perturbation import setup_zeroshot_perturbation
        displacement_hook, diagnostics, obs_holder, manifest_event = setup_zeroshot_perturbation(
            args.zeroshot_manifest_path, args.trial_id, env, obs)
    elif args.condition == "perturbed":
        displacement_hook, diagnostics, obs_holder, manifest_event = setup_perturbation(
            args.task_id, args.trial_id, args.condition, args.manifest_path, env, obs)

    if recorder is not None:
        # Recording starts only after both initialization waits. Its step 0
        # observation is the input to the first real policy action, never an
        # idle/warm-up state. Privileged diagnostics below are labels only.
        from run_mixed_libero10 import get_body_position
        from copy import deepcopy as recording_copy
        recording_body = diagnostics.get("body_name") if diagnostics is not None else None
        recorder.start(task_description)

    def pre_step_hook(episode_step, current_obs):
        """Contract matches asyncmixvla/bridge.py's bridge functions:
        called with the observation as of immediately before this step's
        env.step(), so the displacement gate (which reads obs_holder["obs"])
        always sees the live pre-action state -- during the Adapter phase
        AND during the bridge, regardless of which module is driving
        env.step() at that moment."""
        if displacement_hook is not None:
            obs_holder["obs"] = current_obs
            displacement_hook(episode_step)

    trigger = build_trigger(args, env=env, manifest_event=manifest_event)
    episode_timer = _EpisodeTimer(env)
    episode_timer.install()
    origin = time.perf_counter()
    episode_timer.start()

    state = "ADAPTER"
    episode_step = 0
    done = False
    trigger_fired = False
    trigger_step = None
    trigger_score = None
    trigger_metadata = None
    bridge_result = None
    adapter_steps = 0
    oft_steps = 0
    n_prefetch_executed = 0
    state_reached_oft = False

    mechanism = MECHANISM_FOR_MODE.get(args.mode)  # None for adapter_only

    async_stats = None
    if args.mode == "oft_only":
        # Full OFT baseline: OFT drives from step 0. The perturbation hook is
        # applied every step so novel families animate identically to the other
        # baselines. No trigger, no bridge.
        while not done and episode_step < args.max_steps:
            observation = prepare_observation(obs)
            oft_actions, _ = call_policy("oft", observation, task_description)
            for raw_action in oft_actions[:CHUNK_SIZE]:
                if episode_step >= args.max_steps:
                    break
                action = process_action(raw_action)
                pre_step_hook(episode_step, obs)
                obs, _, done, _ = env.step(action.tolist())
                oft_steps += 1
                episode_step += 1
                if done:
                    break
        state = "DONE"

    if args.mode == "oft_async_continuous":
        # OFT asynchronous at EVERY chunk boundary with a step-based delay -- the
        # regime VLASH targets (see asyncmixvla/continuous_async.py). No Adapter,
        # no trigger; state/vision alignment select Naive, VLASH or Ours.
        if getattr(args, "async_delay", None) is None:
            raise ValueError("--mode oft_async_continuous requires --async_delay")
        from asyncmixvla.continuous_async import run_oft_continuous_async
        res = run_oft_continuous_async(
            env, task_description, obs, delay=int(args.async_delay),
            state_alignment=args.state_alignment, vision_alignment=args.vision_alignment,
            pre_step_hook=pre_step_hook, max_steps=args.max_steps, start_step=episode_step)
        oft_steps += res["final_step"] - episode_step
        done, episode_step, obs = res["done"], res["final_step"], res["obs"]
        async_stats = res["stats"]
        state = "DONE"

    while not done and episode_step < args.max_steps and state == "ADAPTER":
        observation = prepare_observation(obs)
        adapter_actions, _ = call_policy("adapter", observation, task_description)
        adapter_actions = adapter_actions[:CHUNK_SIZE]

        for chunk_position, raw_action in enumerate(adapter_actions):
            if recorder is not None and episode_step >= args.max_steps:
                # Exact real-action horizon for the new collection protocol.
                # Preserve the historical runtime's default chunk behavior.
                break
            fired_here = False
            if args.trigger != "never" and mechanism is not None:
                if args.trigger in ("deployable", "visual_gate", "observable_cascade", "observable_cascade_continuous"):
                    # Structural boundary: pass only camera/proprio inputs and
                    # causal policy/control history, never the raw simulator
                    # observation dictionary, environment or task metadata.
                    result = trigger.update(
                        observation=prepare_observation(obs),
                        adapter_action=process_action(raw_action).tolist(),
                        chunk_position=chunk_position, episode_step=episode_step,
                    )
                else:
                    robot_state = observation["state"] if chunk_position == 0 else prepare_observation(obs)["state"]
                    result = trigger.update(
                        env=env, observation=obs, robot_state=robot_state,
                        adapter_action=process_action(raw_action).tolist(),
                        chunk_position=chunk_position, remaining_actions=CHUNK_SIZE - chunk_position,
                        task_id=args.task_id, episode_step=episode_step,
                    )
                if result.should_switch:
                    fired_here = True
                    trigger_fired = True
                    trigger_step = episode_step
                    trigger_score = result.trigger_score
                    trigger_metadata = result.trigger_metadata

            if fired_here:
                bridge_actions_raw = adapter_actions[chunk_position:]
                state = "ASYNC_OFT_PENDING"
                if mechanism == "sync":
                    bridge_result = run_sync_handoff(env, task_description, obs, bridge_actions_raw, origin,
                                                      action_number_offset=episode_step, pre_step_hook=pre_step_hook)
                else:
                    bridge_result = run_async_bridge(env, task_description, obs, bridge_actions_raw, origin,
                                                      mechanism=mechanism, action_number_offset=episode_step,
                                                      pre_step_hook=pre_step_hook, state_alignment=args.state_alignment,
                                                      vision_alignment=args.vision_alignment,
                                                      adaptive_bridge=bool(getattr(args, "adaptive_bridge", False)))
                adapter_steps += chunk_position
                # With an adaptive bridge the Adapter may have driven past the end of
                # its original chunk, so count what actually executed, not what was
                # planned. Identical to len(bridge_actions_raw) when adaptive is off.
                n_bridge_executed = len(bridge_result.get("action_timeline") or bridge_actions_raw)
                adapter_steps += max(0, n_bridge_executed - len(bridge_actions_raw))
                episode_step += n_bridge_executed
                obs = bridge_result["obs"]
                done = bridge_result["done"]
                if not bridge_result["valid"]:
                    state = "DONE"
                else:
                    state = "OFT"
                    # The bridge function already executed one extra env.step()
                    # beyond bridge_actions_raw -- the first OFT action itself
                    # (obs/done above already reflect it) -- account for that
                    # step here so episode_step and oft_steps both include it.
                    oft_steps += 1
                    episode_step += 1
                break  # leave the adapter_actions for-loop; state machine drives from here

            # Trigger did not fire (or not evaluated): execute this Adapter action normally.
            if recorder is not None:
                recorded_raw_action = raw_action.copy()
            action = process_action(raw_action)
            if recorder is not None:
                recorder.observe(episode_step, chunk_position, prepare_observation(obs),
                                 recorded_raw_action, action.copy())
                hook_body_before = (get_body_position(env, recording_body).copy().tolist()
                                    if recording_body is not None else None)
            pre_step_hook(episode_step, obs)
            if recorder is not None:
                hook_body_after = (get_body_position(env, recording_body).copy().tolist()
                                   if recording_body is not None else None)
                recorder.event(episode_step, recording_copy(diagnostics), hook_body_before, hook_body_after)
            obs, _, done, _ = env.step(action.tolist())
            if recorder is not None:
                recorder.executed(episode_step, action.copy(), bool(done), prepare_observation(obs))
            adapter_steps += 1
            episode_step += 1
            if done:
                state = "DONE"
                break

    if state == "OFT":
        state_reached_oft = True
        # DEVIATION FROM THE ORIGINAL ARCHITECTURE SPEC, MATCHING TABLE 2B'S
        # ACTUAL REFERENCE BEHAVIOR INSTEAD: the spec's section 9 says the
        # first async-fetched OFT chunk "must be consumed" before requesting
        # the next one. The existing, validated Table 2B reference
        # (run_condition_vlash / run_condition_async) does NOT do this -- it
        # uses only the single first OFT action, then immediately issues a
        # brand-new synchronous OFT query via continue_with_oft_until_done,
        # discarding the other 7 already-fetched actions entirely. A
        # trace-diff against the reference (debug_trace_compare.py /
        # debug_trace_diff.py) found this to be the FIRST point of
        # divergence for task6/trial8 (step 107, right where consuming the
        # extra 7 actions had pushed this runtime 7 steps further than the
        # reference) -- consuming them, though arguably the more sensible
        # production design (an already-fetched valid chunk shouldn't go to
        # waste), is NOT what Table 2B's numbers were validated against. For
        # this milestone's reproduce-Table-2B goal, match the reference
        # exactly; bridge_result["remaining_oft_chunk_actions"] is now
        # unused here on purpose. Revisit for the real production runtime.
        # --- takeover horizon (Experiment A: --takeover_k) ------------------
        # k=1 reproduces the frozen/Table-2B behavior EXACTLY: the bridge has
        # already executed action 1 of the fetched OFT chunk and the other 7
        # are discarded. k>1 additionally executes the next k-1 ALREADY-FETCHED
        # OFT actions before returning to fresh-observation replanning. This is
        # what gives any predicted-handoff-context alignment (proprio roll-
        # forward / action-consistency latent / oracle context) a causal
        # execution horizon: at k=1 its influence is a single action out of a
        # 250-500 step rollout. Applies identically to async and sync handoffs
        # (both return remaining_oft_chunk_actions), so the k sweep is matched.
        takeover_k = int(getattr(args, "takeover_k", 1) or 1)
        prefetched = bridge_result.get("remaining_oft_chunk_actions") or []
        n_prefetch_executed = 0
        for extra_action in prefetched[:max(0, takeover_k - 1)]:
            if done or episode_step >= args.max_steps:
                break
            pre_step_hook(episode_step, obs)
            obs, _, done, _ = env.step(list(extra_action))
            oft_steps += 1
            episode_step += 1
            n_prefetch_executed += 1

        if not done and episode_step < args.max_steps:
            # NOTE: continue_with_oft_until_done (reused unmodified from
            # run_hierarchical_live_validation.py) has no pre_step_hook
            # parameter, so any still-in-progress perturbation displacement
            # stops animating once execution reaches this point. In every
            # smoke episode used for milestone 5 the perturbation's fixed
            # duration (5 steps) has already fully elapsed by here, but this
            # is a real, documented limitation, not silently glossed over --
            # see the milestone-5 report.
            steps_before_continuation = episode_step
            cont = continue_with_oft_until_done(env, task_description, obs, done, episode_step, args.max_steps)
            done = cont["success"]
            episode_step = cont["final_step"]
            oft_steps += episode_step - steps_before_continuation
        state = "DONE"

    if debug_logger is not None:
        with open(args.debug_trace_out, "w") as f:
            json.dump({"final_success": bool(done), "steps": debug_logger.records}, f, indent=2)
        print(f"Saved debug trace to {args.debug_trace_out}: {len(debug_logger.records)} steps", flush=True)

    if recorder is not None:
        recorder.finish(bool(done), episode_step)
    env.close()

    trace = {
        "task_id": args.task_id, "trial_id": args.trial_id, "condition": args.condition,
        "seed": 7, "manifest_id": args.manifest_path if args.condition == "perturbed" else None,
        "mode": args.mode, "state_alignment": args.state_alignment, "vision_alignment": args.vision_alignment,
        "final_success": bool(done),
        "trigger_backend": args.trigger, "trigger_fired": trigger_fired, "trigger_step": trigger_step,
        "trigger_score": trigger_score, "trigger_metadata": trigger_metadata,
        "T_predict": bridge_result["T_predict_s"] if bridge_result and bridge_result.get("valid") else None,
        "T_switch": bridge_result["T_switch_s"] if bridge_result and bridge_result.get("valid") else None,
        "T_ready": bridge_result["T_ready_s"] if bridge_result and bridge_result.get("valid") else None,
        "T_action": bridge_result["T_action_s"] if bridge_result and bridge_result.get("valid") else None,
        "bridge_action_count": len(bridge_result["action_timeline"]) if bridge_result else 0,
        "bridge_actions": [a["episode_step"] for a in bridge_result["action_timeline"]] if bridge_result else [],
        "B": bridge_result.get("B_s") if bridge_result and bridge_result.get("valid") else None,
        "C_async": bridge_result.get("C_async_s") if bridge_result and bridge_result.get("valid") else None,
        "S": bridge_result.get("S_s") if bridge_result and bridge_result.get("valid") else None,
        "L_handoff": bridge_result.get("L_handoff_s") if bridge_result and bridge_result.get("valid") else None,
        "missed_ticks": bridge_result.get("missed_control_ticks") if bridge_result and bridge_result.get("valid") else None,
        "no_stall": bridge_result.get("no_stall") if bridge_result and bridge_result.get("valid") else None,
        "first_oft_action": bridge_result.get("first_oft_action") if bridge_result and bridge_result.get("valid") else None,
        "oft_chunk_actions": bridge_result.get("oft_chunk_actions") if bridge_result and bridge_result.get("valid") else None,
        "adapter_steps": adapter_steps, "oft_steps": oft_steps, "final_step": episode_step,
        "vlash_prediction_bridge_actions": bridge_result.get("vlash_prediction_bridge_actions") if bridge_result else None,
        "bridge_executed_raw_actions": (
            [a["raw_action"] for a in bridge_result["action_timeline"]] if bridge_result else None
        ),
        "vlash_actions_match_bridge_actions": (
            bridge_result.get("vlash_prediction_bridge_actions") ==
            [a["raw_action"] for a in bridge_result["action_timeline"]]
        ) if bridge_result and bridge_result.get("mechanism") == "vlash_async" else None,
        "bridge_result_valid": bridge_result.get("valid") if bridge_result else None,
        "bridge_result_reason": bridge_result.get("reason") if bridge_result else None,
        "trigger_debug_trace": getattr(trigger.inner, "trace", None),
        "episode_start_ns": episode_timer.start_ns, "episode_end_ns": episode_timer.end_ns,
        "completion_time_s": ((episode_timer.end_ns - episode_timer.start_ns) / 1e9
                              if episode_timer.end_ns is not None else None),
        "adaptive_bridge": bool(getattr(args, "adaptive_bridge", False)),
        "adaptive_extra_bridge_actions": (bridge_result.get("adaptive_extra_bridge_actions")
                                           if bridge_result else None),
        "takeover_k": int(getattr(args, "takeover_k", 1) or 1),
        "n_prefetch_executed": n_prefetch_executed if state_reached_oft else 0,
        "perturbation_diagnostics": diagnostics,
        "zeroshot_family": zeroshot_family,
        "async_delay": getattr(args, "async_delay", None),
        "async_stats": async_stats,
    }
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--trial_id", type=int, required=True)
    parser.add_argument("--condition", required=True, choices=["clean", "perturbed"])
    parser.add_argument("--mode", required=True,
                         choices=["adapter_only", "oft_only", "sync_full_oft", "naive_async_full_oft",
                                  "vlash_async_full_oft", "oft_async_continuous"])
    parser.add_argument("--async_delay", type=int, default=None,
                         help="oft_async_continuous only: actions executed between capturing the "
                              "observation and the new OFT chunk arriving (0 = synchronous).")
    parser.add_argument("--zeroshot_family", default=None,
                         help="If set, apply a novel zero-shot perturbation family from --zeroshot_manifest_path "
                              "instead of the frozen translation perturbation. The AsyncMixVLA pipeline is unchanged.")
    parser.add_argument("--zeroshot_manifest_path", default=None)
    parser.add_argument("--trigger_persistence_k", type=int, default=1,
                         help="learned_continuous: fire after this many CONSECUTIVE steps at/above threshold.")
    parser.add_argument("--trigger_no_stage_a_gate", action="store_true",
                         help="learned_continuous: drop the Stage-A motion-precursor gate entirely.")
    parser.add_argument("--trigger_dry_run", action="store_true",
                         help="learned_continuous: score and log every step but NEVER fire (calibration pass).")
    parser.add_argument("--adaptive_bridge", action="store_true",
                         help="Keep the Adapter driving (fetching further chunks) until OFT is actually ready, "
                              "instead of stalling the robot when the current chunk runs out. Off by default so "
                              "every previously recorded result stays bit-reproducible.")
    parser.add_argument("--takeover_k", type=int, default=1,
                         help="How many actions of the already-prefetched OFT chunk to execute after the bridge "
                              "before returning to fresh-observation replanning. k=1 is the frozen behavior.")
    parser.add_argument("--trigger", default="oracle_onset",
                         choices=["never", "oracle_onset", "forced_step", "deployable",
                                  "visual_gate", "observable_cascade", "observable_cascade_continuous"],
                         help="Deployable observable triggers plus oracle/forced-step debug baselines.")
    parser.add_argument("--observable_cascade_checkpoint", default=None,
                         help="Observable V1/V2 artifact. Use corrected observable_cascade_v1 runtime for evaluation.")
    parser.add_argument("--visual_gate_checkpoint", default=None,
                         help="Required for --trigger visual_gate: calibrated camera-gate artifact JSON.")
    parser.add_argument("--deployable_checkpoint", default=None,
                         help="Required for --trigger deployable: calibrated camera/proprio/action-history detector.")
    parser.add_argument("--forced_trigger_step", type=int, default=None,
                         help="Required with --trigger forced_step: hand off the first time episode_step >= this. "
                              "Used by the handoff-only live DEV benchmark so every method switches at the same step.")
    release_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--oracle_manifest", default=str(
        release_root / "results/manifests/oracle_trigger/oracle_manifest.json"))
    parser.add_argument("--manifest_path", default=None,
                         help="Perturbation manifest; required for --condition perturbed.")
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--warmup", type=int, default=3,
                         help="Dummy Adapter+OFT queries before the real timed run (0 to disable).")
    parser.add_argument("--debug_trace_out", default=None,
                         help="TEMPORARY debug instrumentation: if set, dumps a per-step action/state trace "
                              "(see debug_trace_compare.py) to this path. No effect on default behavior.")
    parser.add_argument("--state_alignment", default="vlash_additive",
                         choices=["stale", "vlash_additive", "vlash_gain_corrected"],
                         help="Proprio alignment mode for mechanism=vlash_async (see asyncmixvla/state_alignment.py). "
                              "Default 'vlash_additive' reproduces the already-validated AsyncMixVLA path unchanged. "
                              "Ignored for naive_async/sync mechanisms.")
    parser.add_argument("--vision_alignment", default="stale",
                         choices=["stale", "oracle_future", "oracle_context", "f2f_ap", "action_consistency"],
                         help="Vision alignment mode for run_async_bridge (see asyncmixvla/vision_alignment.py). "
                              "Default 'stale' reproduces the already-validated AsyncMixVLA path unchanged. "
                              "'f2f_ap' requires the OFT server to have been started with --f2f_ap_checkpoint. "
                              "'action_consistency' is the frozen gate-passed AsyncMixVLA visual-handoff method "
                              "(job 2168888) and requires the OFT server started with "
                              "--action_consistency_checkpoint. "
                              "'oracle_future' is acausal/comparison-only (see vision_alignment.py) and disables "
                              "async overlap for that call (bridge runs to completion before OFT is queried) -- "
                              "never a real-deployment option, an upper-bound reference only.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.mode in ("adapter_only", "oft_only"):
        args.trigger = "never"
    if args.condition == "perturbed" and not args.manifest_path:
        args.manifest_path = str(
            release_root / f"results/manifests/full10_perturbations/manifest_task{args.task_id}.json")

    print(f"=== run_asyncmixvla: task{args.task_id} trial{args.trial_id} {args.condition} "
          f"mode={args.mode} trigger={args.trigger} ===", flush=True)
    trace = run_episode(args)
    print(f"final_success={trace['final_success']} trigger_fired={trace['trigger_fired']} "
          f"trigger_step={trace['trigger_step']} T_predict={trace['T_predict']} T_switch={trace['T_switch']} "
          f"B={trace['B']} C_async={trace['C_async']} S={trace['S']} L_handoff={trace['L_handoff']} "
          f"no_stall={trace['no_stall']}", flush=True)

    with open(args.out, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
