"""Corrected-runtime TEST driver for the observable_cascade_v1 trigger family
(v1_direct / v1_v2_cascade / v1_v2_cascade_continuous).

WHY THIS FILE EXISTS (incident, 2026-09-17): run_final_switching_tables.py calls
run_asyncmixvla.run_episode, whose OFT-continuation phase (continue_with_oft_
until_done) does NOT keep animating an in-progress perturbation past the switch
-- see run_asyncmixvla.py's own inline comment at that call site. But V1/V2 were
fit, calibrated, and DEV-validated entirely under a DIFFERENT runtime --
research_recovery_v1.runtime.EpisodeEnv -- whose hook is applied on every single
env.step(), including OFT continuation (research_recovery_v1/PROTOCOL.md: "hook
applied before EVERY action including OFT continuation"; "Historical DEV numbers
are not interchangeable with this corrected benchmark"). The first TEST attempt
(jobs 2267936, 2268413) used the mismatched old runtime and was cancelled before
being reported; those traces stay on disk under results/final_switching/traces/
as *_corrected-less names, exploratory only, never to be used for Table 2. This
file re-runs TEST under the SAME runtime as training/DEV.

Reuses research_recovery_v1.runtime.EpisodeEnv and run_asyncmixvla.build_trigger
unmodified (both frozen). Does NOT call research_recovery_v1.runtime.run()
directly: that function's own docstring declares itself frozen/untouched, and
its internal trigger-args SimpleNamespace does not plumb through
--observable_cascade_checkpoint. The control loop below is a close copy of that
run() function, extended only to pass the checkpoint path to build_trigger() and
to capture the actual TriggerResult.trigger_metadata at the switching decision
instead of fabricating one.

Each output record also carries sha256 of this driver file and of the trigger
checkpoint used, addressing a real gap flagged in the same audit: the original
driver recorded no source/artifact hashes at all, so provenance across an array
job spanning hours of file edits could not be proven after the fact.
"""
import argparse
import hashlib
import json
import os
import time
import types
from pathlib import Path

from experiments.robot.libero.libero_utils import get_libero_dummy_action
from experiments.robot.robot_utils import set_seed_everywhere
from measure_async_handoff_latency import warm_up
from research_recovery_v1.runtime import EpisodeEnv
from run_asyncmixvla import build_trigger, setup_perturbation
from run_test0_switch_timing import MAX_STEPS, call_policy, get_env_and_task, prepare_observation, process_action

from asyncmixvla.bridge import run_async_bridge, run_sync_handoff

OPENVLA_ROOT = Path(__file__).resolve().parent
REPO_ROOT = OPENVLA_ROOT.parent
DEFAULT_TEST_MANIFEST = str(REPO_ROOT / "results/manifests/final_test/test_episode_manifest_portable.json")
DEFAULT_ORACLE_MANIFEST = str(REPO_ROOT / "results/manifests/oracle_trigger/oracle_manifest.json")

TRIGGER_MARK = {"v1_v2_cascade": "v1v2_corrected",
                "v1_v2_cascade_continuous": "v1v2cont_corrected", "oracle_onset": "oracle_corrected"}
OBSERVABLE_CASCADE_CHECKPOINT = {
    "v1_v2_cascade": str(OPENVLA_ROOT / "observable_cascade_v1/models/v2_candidate.json"),
    "v1_v2_cascade_continuous": str(OPENVLA_ROOT / "observable_cascade_v1/models/v2_candidate.json"),
    # oracle_onset needs no observable_cascade checkpoint; build_trigger reads
    # args.oracle_manifest/task_id/trial_id/condition instead. Kept as None here
    # so this dict stays a single source of truth for "does this trigger need a
    # checkpoint file" without a second parallel mapping.
    "oracle_onset": None,
}
RUNTIME_TRIGGER = {"v1_v2_cascade": "observable_cascade",
                    "v1_v2_cascade_continuous": "observable_cascade_continuous", "oracle_onset": "oracle_onset"}

METHOD_CONFIG = {
    "sync":  dict(mechanism="sync",  alignment="stale"),
    "naive": dict(mechanism="async", alignment="stale"),
    "ours":  dict(mechanism="async", alignment="ac"),
}

_THIS_FILE_SHA256 = hashlib.sha256(open(__file__, "rb").read()).hexdigest()


def run_corrected_episode(point, method, trigger_name, checkpoint, max_steps=MAX_STEPS,
                          oracle_manifest=DEFAULT_ORACLE_MANIFEST, takeover_k=1):
    """Close copy of research_recovery_v1.runtime.run()'s control loop (frozen,
    untouched by its own docstring), extended only to plumb an observable_cascade
    checkpoint (or oracle_manifest, for --trigger oracle_onset) through to
    build_trigger() and to record the real switching decision's trigger_metadata."""
    config = METHOD_CONFIG[method]
    backend = RUNTIME_TRIGGER[trigger_name]
    set_seed_everywhere(7)
    base, language, initial = get_env_and_task(task_id=point["task_id"])
    try:
        warm_up(base, language, initial[point["trial_index"]], 3)
        base.reset()
        obs = base.set_init_state(initial[point["trial_index"]])
        for _ in range(10):
            obs, _, _, _ = base.step(get_libero_dummy_action("openvla"))
        hook, diag, holder, event = None, None, None, None
        if point["condition"] == "perturbed":
            hook, diag, holder, event = setup_perturbation(
                point["task_id"], point["trial_index"], point["condition"],
                point["perturbation_manifest"], base, obs)
        env = EpisodeEnv(base, obs, max_steps, hook, holder)
        # oracle_onset reads args.oracle_manifest/task_id/trial_id/condition (read-only,
        # frozen manifest -- see OracleOnsetTrigger); observable_cascade[_continuous]
        # reads args.observable_cascade_checkpoint instead. Both fields are always set;
        # build_trigger only consults the one its backend actually needs.
        args = types.SimpleNamespace(
            trigger=backend, forced_trigger_step=point.get("episode_step"), task_id=point["task_id"],
            observable_cascade_checkpoint=checkpoint, oracle_manifest=oracle_manifest,
            trial_id=point["trial_index"], condition=point["condition"])
        trigger = build_trigger(args, env=base, manifest_event=event)
        origin = time.perf_counter()
        if not 1 <= int(takeover_k) <= 8:
            raise ValueError("takeover_k must be in [1,8]")
        fired, switch_step, handoff = False, None, None
        n_prefetch_executed = 0
        switch_metadata, prefix_hash = None, None
        active = "adapter"
        while not env.done and env.steps < max_steps:
            env.policy = active
            env.oft_starts_at = None
            chunk, _ = call_policy(active, prepare_observation(env.obs), language)
            chunk = chunk[:8]
            if len(chunk) != 8:
                raise ValueError(f"expected 8 actions, got {len(chunk)}")
            for pos, raw in enumerate(chunk):
                if env.done or env.steps >= max_steps:
                    break
                # process_action is non-mutating (normalize_/invert_gripper_action both
                # .copy()), but compute it ONCE and reuse for both the trigger and the
                # env step, matching observable_cascade_v1/runtime.py -- the runtime V1/V2
                # were actually validated under -- so the two can be diffed bit-for-bit.
                action = process_action(raw.copy())
                decision = None
                if active == "adapter" and not fired:
                    # observable_cascade[_continuous] enforces a narrower, non-privileged
                    # update() signature (camera/proprio only -- see run_asyncmixvla.py's
                    # own branch at the equivalent call site, which this mirrors). Passing
                    # env/robot_state/remaining_actions/task_id here, as the privileged
                    # backends require, raises TypeError -- caught by smoke_corrected_
                    # driver.py before any TEST compute was spent.
                    if backend in ("observable_cascade", "observable_cascade_continuous"):
                        decision = trigger.update(
                            observation=prepare_observation(env.obs), adapter_action=action.tolist(),
                            chunk_position=pos, episode_step=env.steps)
                    else:
                        decision = trigger.update(
                            env=base, observation=env.obs, robot_state=prepare_observation(env.obs)["state"],
                            adapter_action=action.tolist(), chunk_position=pos,
                            remaining_actions=8 - pos, task_id=point["task_id"], episode_step=env.steps)
                if decision is not None and decision.should_switch and env.steps < max_steps - 1:
                    fired, switch_step = True, env.steps
                    switch_metadata = decision.trigger_metadata
                    prefix_hash = env.digest.hexdigest()
                    bridge = chunk[pos:]
                    bridge = bridge[:min(8, max_steps - env.steps - 1)]
                    env.oft_starts_at = env.steps + len(bridge)
                    if config["mechanism"] == "sync":
                        handoff = run_sync_handoff(env, language, env.obs, bridge, origin,
                                                   action_number_offset=env.steps)
                    else:
                        ours = config["alignment"] == "ac"
                        handoff = run_async_bridge(
                            env, language, env.obs, bridge, origin,
                            mechanism="vlash_async" if ours else "naive_async",
                            action_number_offset=env.steps,
                            state_alignment="vlash_gain_corrected" if ours else "stale",
                            vision_alignment="action_consistency" if ours else "stale")
                    env.oft_starts_at = None
                    if handoff["valid"]:
                        active = "oft"
                        # run_*_handoff has already executed the first action of
                        # the fetched OFT chunk.  Execute the same additional
                        # k-1 actions for sync, naive, and ours before the next
                        # fresh-observation query.  k=1 preserves the previous
                        # Table-2 behavior exactly.
                        env.policy = "oft"
                        prefetched = handoff.get("remaining_oft_chunk_actions") or []
                        for extra_action in prefetched[:max(0, int(takeover_k) - 1)]:
                            if env.done or env.steps >= max_steps:
                                break
                            env.step(list(extra_action))
                            n_prefetch_executed += 1
                    elif not env.done:
                        raise RuntimeError(f'Invalid handoff without terminal success: {handoff.get("reason")}')
                    break
                env.step(action.tolist())
        if sum(env.counts.values()) != env.steps or env.steps > max_steps:
            raise AssertionError("Invalid action accounting")
        return {
            "final_success": bool(env.done), "final_step": env.steps, "policy_steps": env.counts,
            "trigger_fired": fired, "trigger_step": switch_step, "trigger_metadata": switch_metadata,
            "prefix_action_sha256": prefix_hash, "all_actions_sha256": env.digest.hexdigest(),
            "perturbation_diagnostics": diag,
            "trigger_debug_trace": getattr(getattr(trigger, "inner", None), "trace", None),
            "bridge_result_valid": handoff.get("valid") if handoff else None,
            "bridge_action_count": len(handoff.get("action_timeline") or []) if handoff else None,
            "L_handoff": handoff.get("L_handoff_s") if handoff else None,
            "C_async": handoff.get("C_async_s") if handoff else None,
            "B": handoff.get("B_s") if handoff else None,
            "no_stall": handoff.get("no_stall") if handoff else None,
            "first_oft_action": handoff.get("first_oft_action") if handoff else None,
            "oft_chunk_actions": handoff.get("oft_chunk_actions") if handoff else None,
            "takeover_k": int(takeover_k),
            "n_prefetch_executed": n_prefetch_executed,
            "wall_s": time.perf_counter() - origin,
        }
    finally:
        base.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", type=int, required=True)
    ap.add_argument("--trigger", required=True, choices=list(TRIGGER_MARK))
    ap.add_argument("--methods", default="sync,naive,ours")
    ap.add_argument("--manifest", default=DEFAULT_TEST_MANIFEST)
    ap.add_argument("--oracle_manifest", default=DEFAULT_ORACLE_MANIFEST,
                    help="only consulted for --trigger oracle_onset; read-only, frozen")
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "runs/final_switching_corrected/traces"))
    ap.add_argument("--max_steps", type=int, default=MAX_STEPS)
    ap.add_argument("--limit", type=int, default=None, help="smoke: only the first N episodes for this task")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--split", default=None,
                    help="Optional manifest split filter (for TRAIN/DEV development runs).")
    ap.add_argument("--observable_cascade_checkpoint", default=None,
                    help="Override the trigger artifact for a development run.")
    ap.add_argument("--mark", default=None, help="Override output filename prefix.")
    ap.add_argument("--takeover_k", type=int, default=1,
                    help="Number of actions to consume from the fetched OFT chunk; k=1 is legacy behavior.")
    ap.add_argument("--action_consistency_checkpoint", default=None,
                    help="Checkpoint loaded by the OFT server; recorded and hashed for provenance.")
    args = ap.parse_args()

    if not 1 <= args.takeover_k <= 8:
        raise ValueError("takeover_k must be in [1,8]")

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.load(open(manifest_path))
    for point in manifest:
        perturbation_manifest = point.get("perturbation_manifest")
        if perturbation_manifest and not Path(perturbation_manifest).is_absolute():
            point["perturbation_manifest"] = str(
                (manifest_path.parent / perturbation_manifest).resolve()
            )
    points = [p for p in manifest if p["task_id"] == args.task_id]
    if args.split is not None:
        points = [p for p in points if str(p.get("split")) == args.split]
        if not all(int(p["trial_index"]) >= 12 for p in points):
            raise AssertionError("split-filtered development runs may not include TEST")
    if args.limit is not None:
        points = points[:args.limit]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    mark = args.mark or TRIGGER_MARK[args.trigger]
    checkpoint = args.observable_cascade_checkpoint or OBSERVABLE_CASCADE_CHECKPOINT[args.trigger]
    checkpoint_sha256 = (hashlib.sha256(open(checkpoint, "rb").read()).hexdigest()
                         if checkpoint else hashlib.sha256(open(args.oracle_manifest, "rb").read()).hexdigest())
    ac_checkpoint = (os.path.abspath(args.action_consistency_checkpoint)
                     if args.action_consistency_checkpoint else None)
    ac_checkpoint_sha256 = (hashlib.sha256(open(ac_checkpoint, "rb").read()).hexdigest()
                            if ac_checkpoint else None)
    print(f"task {args.task_id} trigger={args.trigger} (corrected runtime): {len(points)} episodes x "
          f"{len(methods)} methods = {len(points) * len(methods)} runs; takeover_k={args.takeover_k}",
          flush=True)

    done = failed = skipped = 0
    for pi, point in enumerate(points):
        for m in methods:
            tag = f"{mark}_{m}__task{point['task_id']}_trial{point['trial_index']}_{point['condition']}"
            out_path = os.path.join(args.out_dir, tag + ".json")
            if os.path.exists(out_path) and not args.force:
                skipped += 1
                continue
            t0 = time.perf_counter()
            rec = {"tag": tag, "method": m, "trigger": args.trigger, "bench_point": point,
                   "runtime": "research_recovery_v1.EpisodeEnv (corrected)",
                   "driver_sha256": _THIS_FILE_SHA256, "checkpoint_sha256": checkpoint_sha256,
                   "takeover_k": args.takeover_k,
                   "action_consistency_checkpoint": ac_checkpoint,
                   "action_consistency_sha256": ac_checkpoint_sha256}
            try:
                rec["trace"] = run_corrected_episode(point, m, args.trigger, checkpoint, args.max_steps,
                                                     oracle_manifest=args.oracle_manifest,
                                                     takeover_k=args.takeover_k)
                rec["ok"] = True
            except Exception as e:
                import traceback
                rec["ok"] = False
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["traceback"] = traceback.format_exc()
                failed += 1
            rec["wall_s"] = time.perf_counter() - t0
            json.dump(rec, open(out_path, "w"), indent=2)
            done += 1
            tr = rec.get("trace", {})
            print(f"  [{pi + 1}/{len(points)}] {tag}  ok={rec['ok']} success={tr.get('final_success')} "
                  f"fired={tr.get('trigger_fired')} step={tr.get('trigger_step')} "
                  f"L_handoff={tr.get('L_handoff')} ({rec['wall_s']:.1f}s)", flush=True)
    print(f"\ntask {args.task_id} trigger={args.trigger}: {done} run, {failed} errored, {skipped} skipped",
          flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
