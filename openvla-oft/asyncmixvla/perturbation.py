"""Manifest-driven LIBERO-10 disturbance runtime used by evaluation deployment."""

import logging
from collections import defaultdict

import numpy as np
from experiments.robot.libero.env_perturbations import get_body_position, get_body_z, translate_body_xy

logger = logging.getLogger(__name__)

LIBERO_10_SOURCE_TARGET = {
    0: {"source_objects": ["alphabet_soup_1_main", "tomato_sauce_1_main"], "target_objects": ["basket_1_main"]},
    1: {"source_objects": ["cream_cheese_1_main", "butter_1_main"], "target_objects": ["basket_1_main"]},
    2: {"source_objects": ["moka_pot_1_main"], "target_objects": ["flat_stove_1_main"]},
    3: {"source_objects": ["akita_black_bowl_1_main"], "target_objects": ["white_cabinet_1_cabinet_bottom"]},
    4: {"source_objects": ["porcelain_mug_1_main", "white_yellow_mug_1_main"], "target_objects": ["plate_1_main", "plate_2_main"], "source_to_target": {"porcelain_mug_1_main": "plate_1_main", "white_yellow_mug_1_main": "plate_2_main"}},
    5: {"source_objects": ["black_book_1_main"], "target_objects": ["desk_caddy_1_main"]},
    6: {"source_objects": ["porcelain_mug_1_main", "chocolate_pudding_1_main"], "target_objects": ["plate_1_main"], "source_to_target": {"porcelain_mug_1_main": "plate_1_main", "chocolate_pudding_1_main": "right_of_plate_1_main"}},
    7: {"source_objects": ["alphabet_soup_1_main", "cream_cheese_1_main"], "target_objects": ["basket_1_main"]},
    8: {"source_objects": ["moka_pot_1_main", "moka_pot_2_main"], "target_objects": ["flat_stove_1_main"]},
    9: {"source_objects": ["white_yellow_mug_1_main"], "target_objects": ["microwave_1_main"]},
}


class GateArgs:
    source_displace_require_on_table = True
    source_displace_table_z_tol = 0.025
    source_displace_min_gripper_dist = 0.08
    obj_displace_zero_velocity = True
    target_displace_carry_xy_tol = 0.12
    target_displace_carry_z_tol = 0.12


def _source_to_target(task_id):
    info = LIBERO_10_SOURCE_TARGET.get(task_id, {})
    if "source_to_target" in info:
        return dict(info["source_to_target"])
    targets = info.get("target_objects", [])
    return {source: targets[0] for source in info.get("source_objects", [])} if len(targets) == 1 else {}


def _body_available(env, body_name):
    try:
        get_body_position(env, body_name)
        return True
    except Exception:
        return False


def _already_on_target(env, task_id, source_body):
    target = _source_to_target(task_id).get(source_body)
    if target is None or not _body_available(env, target):
        return False
    source_pos, target_pos = get_body_position(env, source_body), get_body_position(env, target)
    return (
        np.linalg.norm(source_pos[:2] - target_pos[:2]) <= GateArgs.target_displace_carry_xy_tol
        and abs(get_body_z(env, source_body) - get_body_z(env, target)) <= GateArgs.target_displace_carry_z_tol
    )


def get_source_initial_z_map(env, task_id):
    result = {}
    for body_name in LIBERO_10_SOURCE_TARGET.get(task_id, {}).get("source_objects", []):
        try:
            result[body_name] = get_body_z(env, body_name)
        except Exception:
            pass
    return result


def source_displacement_allowed(env, obs, task_id, body_name, source_initial_z_map):
    if _already_on_target(env, task_id, body_name):
        return False
    if body_name not in source_initial_z_map:
        return False
    if get_body_z(env, body_name) > source_initial_z_map[body_name] + GateArgs.source_displace_table_z_tol:
        return False
    distance = np.linalg.norm(get_body_position(env, body_name) - np.asarray(obs["robot0_eef_pos"]))
    return bool(distance >= GateArgs.source_displace_min_gripper_dist)


def schedule_from_manifest_event(manifest_event):
    duration = int(manifest_event["duration"])
    delta_xy = np.asarray(manifest_event["delta_xy"], dtype=np.float64)
    event = {
        "group": manifest_event.get("group", "target"),
        "body_name": manifest_event["body_name"],
        "total_delta_xy": delta_xy,
        "delta_xy_per_step": delta_xy / duration,
        "displacement_magnitude": float(manifest_event["magnitude"]),
        "duration": duration,
    }
    schedule = defaultdict(list)
    schedule[int(manifest_event["trigger_step"])].append(event)
    return schedule


def apply_displacement_step(env, episode_step, schedule, active, source_z, obs, diagnostics, task_id):
    for event in schedule.get(episode_step, []):
        body = event["body_name"]
        if body in active or not source_displacement_allowed(env, obs, task_id, body, source_z):
            continue
        active[body] = {
            "group": event["group"], "remaining": event["duration"],
            "delta_xy_per_step": event["delta_xy_per_step"],
            "total_delta_xy": event["total_delta_xy"],
            "displacement_magnitude": event["displacement_magnitude"],
        }
        if diagnostics is not None and body == diagnostics["body_name"]:
            diagnostics.update(fired=True, never_reached=False, position_before=get_body_position(env, body).tolist())
    for body in list(active):
        if not source_displacement_allowed(env, obs, task_id, body, source_z):
            if diagnostics is not None and body == diagnostics["body_name"]:
                diagnostics.update(cancelled_early=True, position_after=get_body_position(env, body).tolist())
            del active[body]
            continue
        if not translate_body_xy(env, body, active[body]["delta_xy_per_step"], zero_velocity=True):
            if diagnostics is not None and body == diagnostics["body_name"]:
                diagnostics.update(cancelled_early=True, position_after=get_body_position(env, body).tolist())
            del active[body]
