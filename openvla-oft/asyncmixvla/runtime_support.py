"""Small runtime helpers shared by the deployable AsyncMixVLA runners."""

from run_test0_switch_timing import CHUNK_SIZE, call_policy, prepare_observation, process_action
from run_test0_switch_timing import apply_displacement_step


def make_displacement_stepper(env, displacement_schedule, active_displacements,
                              source_initial_z_map, diagnostics, obs_holder, task_id):
    """Apply the recorded disturbance before every real environment step."""
    state = {"last_step": None}

    def hook(step_idx):
        if state["last_step"] is not None and step_idx > state["last_step"]:
            for body_name in list(active_displacements):
                active_displacements[body_name]["remaining"] -= 1
                if active_displacements[body_name]["remaining"] <= 0:
                    del active_displacements[body_name]
        apply_displacement_step(
            env, step_idx, displacement_schedule, active_displacements,
            source_initial_z_map, obs_holder["obs"], diagnostics, task_id=task_id,
        )
        state["last_step"] = step_idx

    return hook


def continue_with_oft_until_done(env, task_description, obs, done, t_start, max_steps):
    """Continue with fresh-observation OFT queries until success or timeout."""
    action_log = []
    t = t_start
    success = bool(done)
    success_step = t_start if done else None
    while not success and t < max_steps:
        actions, _ = call_policy("oft", prepare_observation(obs), task_description)
        for raw_action in actions[:CHUNK_SIZE]:
            if t >= max_steps:
                break
            obs, _, done, _ = env.step(process_action(raw_action).tolist())
            action_log.append({"t": t, "model": "oft"})
            t += 1
            if done:
                success, success_step = True, t
                break
    return {"success": success, "success_step": success_step, "final_step": t,
            "action_log": action_log}
