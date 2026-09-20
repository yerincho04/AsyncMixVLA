import argparse
import gc
import json
import logging
import os
import pickle
import time
from collections import deque, defaultdict
from typing import Dict, Any, Tuple, List, Set

import msgpack
import msgpack_numpy as m
import numpy as np
import requests
import torch
import tqdm

m.patch()

from libero.libero import benchmark

from experiments.robot.libero.run_libero_eval import GenerateConfig, check_unnorm_key
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)

from experiments.robot.libero.action_noise import apply_action_noise
from experiments.robot.libero.visual_noise import apply_visual_noise, generate_fixed_noise_params
from experiments.robot.libero.env_perturbations import (
    body_has_free_joint,
    get_body_position,
    get_body_z,
    list_body_names,
    sample_random_xy_displacement,
    translate_body_xy,
)

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)

from experiments.robot.robot_utils import (
    DATE_TIME,
    normalize_gripper_action,
    invert_gripper_action,
    set_seed_everywhere,
    get_action,
    get_image_resize_size,
    get_model,
)


TASK_MAX_STEPS = {
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


LIBERO_10_SOURCE_TARGET = {
    0: {
        "source_objects": ["alphabet_soup_1_main", "tomato_sauce_1_main"],
        "target_objects": ["basket_1_main"],
    },
    1: {
        "source_objects": ["cream_cheese_1_main", "butter_1_main"],
        "target_objects": ["basket_1_main"],
    },
    2: {
        "source_objects": ["moka_pot_1_main"],
        "target_objects": ["flat_stove_1_main"],
    },
    3: {
        "source_objects": ["akita_black_bowl_1_main"],
        "target_objects": ["white_cabinet_1_cabinet_bottom"],
    },
    4: {
        "source_objects": ["porcelain_mug_1_main", "white_yellow_mug_1_main"],
        "target_objects": ["plate_1_main", "plate_2_main"],
        "source_to_target": {
            "porcelain_mug_1_main": "plate_1_main",
            "white_yellow_mug_1_main": "plate_2_main",
        },
    },
    5: {
        "source_objects": ["black_book_1_main"],
        "target_objects": ["desk_caddy_1_main"],
        "target_region": "desk_caddy_1_back_contain_region",
    },
    6: {
        "source_objects": ["porcelain_mug_1_main", "chocolate_pudding_1_main"],
        "target_objects": ["plate_1_main"],
        "source_to_target": {
            "porcelain_mug_1_main": "plate_1_main",
            "chocolate_pudding_1_main": "right_of_plate_1_main",
        },
    },
    7: {
        "source_objects": ["alphabet_soup_1_main", "cream_cheese_1_main"],
        "target_objects": ["basket_1_main"],
    },
    8: {
        "source_objects": ["moka_pot_1_main", "moka_pot_2_main"],
        "target_objects": ["flat_stove_1_main"],
    },
    9: {
        "source_objects": ["white_yellow_mug_1_main"],
        "target_objects": ["microwave_1_main"],
        "target_region": "microwave_1_heating_region",
    },
}


SESSION = requests.Session()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


FIXED_VISUAL_NOISE_TYPES = {
    "blur",
    "image_shift",
    "image_rotation",
    "enhanced_color_jitter",
}


def setup_logging(args):
    run_id = f"EVAL-{args.task_suite_name}-{args.policy}"

    if args.policy == "mixed":
        run_id += f"-first_{args.mixed_first}"

    run_id += f"-chunk{args.chunk_size}-seed{args.seed}-{DATE_TIME}"

    if args.run_id_note is not None:
        run_id += f"--{args.run_id_note}"

    os.makedirs(args.local_log_dir, exist_ok=True)

    local_log_filepath = os.path.join(args.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")

    logger.info(f"Logging to local log file: {local_log_filepath}")

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None, debug=False):
    if debug:
        logger.debug(message)
    else:
        logger.info(message)

    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def call_policy_server(
    endpoint: str,
    observation: Dict[str, Any],
    task_description: str,
    timeout: int = 120,
) -> np.ndarray:
    payload = {
        "full_image": observation["full_image"],
        "wrist_image": observation["wrist_image"],
        "state": observation["state"],
        "task_description": task_description,
    }

    packed = msgpack.packb(payload, default=m.encode, use_bin_type=True)

    r = SESSION.post(
        endpoint,
        data=packed,
        headers={"Content-Type": "application/msgpack"},
        timeout=timeout,
    )
    r.raise_for_status()

    out = msgpack.unpackb(r.content, object_hook=m.decode, raw=False)
    actions = np.asarray(out["actions"], dtype=np.float32)

    if actions.ndim != 2 or actions.shape[-1] != 7:
        raise ValueError(f"Expected action chunk shape [T, 7], got {actions.shape}")

    return actions


def load_direct_oft_bundle(checkpoint: str, chunk_size: int, seed: int) -> Dict[str, Any]:
    """Load an OFT policy in-process (no HTTP), for the direct-vs-server diagnostic.

    Mirrors serve_oft_libero10.py's make_cfg()/model-loading exactly, so the only
    thing that differs from the server path is the transport (no HTTP/msgpack hop).
    """
    cfg = GenerateConfig(
        pretrained_checkpoint=checkpoint,
        task_suite_name="libero_10",
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=chunk_size,
        seed=seed,
    )

    logger.info(f"[Direct OFT] Loading model: {cfg.pretrained_checkpoint}")

    model = get_model(cfg)
    check_unnorm_key(cfg, model)

    processor = get_processor(cfg)
    action_head = get_action_head(cfg, model.llm_dim)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    resize_size = get_image_resize_size(cfg)

    model.eval()
    action_head.eval()
    proprio_projector.eval()

    logger.info(f"[Direct OFT] Ready. unnorm_key={cfg.unnorm_key} resize_size={resize_size}")

    return {
        "cfg": cfg,
        "model": model,
        "processor": processor,
        "action_head": action_head,
        "proprio_projector": proprio_projector,
        "resize_size": resize_size,
    }


def call_policy_direct(
    bundle: Dict[str, Any],
    observation: Dict[str, Any],
    task_description: str,
) -> np.ndarray:
    """In-process equivalent of call_policy_server for the direct-vs-server diagnostic.

    Applies the exact same resize_image_for_policy step the server applies, then
    calls get_action directly -- no msgpack/HTTP round trip.
    """
    cfg = bundle["cfg"]
    resize_size = bundle["resize_size"]

    obs = {
        "full_image": resize_image_for_policy(observation["full_image"], resize_size),
        "wrist_image": resize_image_for_policy(observation["wrist_image"], resize_size),
        "state": np.asarray(observation["state"], dtype=np.float32),
    }

    with torch.inference_mode():
        actions = get_action(
            cfg,
            bundle["model"],
            obs,
            task_description,
            processor=bundle["processor"],
            action_head=bundle["action_head"],
            proprio_projector=bundle["proprio_projector"],
            noisy_action_projector=None,
            use_film=cfg.use_film,
        )

    actions = np.asarray(actions, dtype=np.float32)

    if actions.ndim != 2 or actions.shape[-1] != 7:
        raise ValueError(f"Expected action chunk shape [T, 7], got {actions.shape}")

    return actions


def prepare_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)

    return {
        "full_image": img,
        "wrist_image": wrist_img,
        "state": state,
    }


def process_action_for_libero(action: np.ndarray) -> np.ndarray:
    action = normalize_gripper_action(action, binarize=True)
    action = invert_gripper_action(action)
    return action


def visual_noise_enabled(args) -> bool:
    return args.obs_noise_type != "none"


def action_noise_enabled(args) -> bool:
    return args.action_noise_type != "none" and args.action_noise_magnitude > 0.0


def object_displacement_enabled(args) -> bool:
    source_enabled = (
        args.source_obj_displace_num_triggers > 0
        and args.source_obj_displace_magnitude > 0.0
    )

    target_enabled = (
        args.target_obj_displace_num_triggers > 0
        and args.target_obj_displace_magnitude > 0.0
    )

    return source_enabled or target_enabled


def make_visual_noise_kwargs(args) -> Dict[str, Any]:
    return {
        "salt_pepper_probability": args.obs_salt_pepper_probability,
        "kernel_size": args.obs_blur_kernel_size,
        "sigma": args.obs_blur_sigma,
        "max_shift_ratio": args.obs_image_shift_ratio,
        "max_angle": args.obs_image_rotation_angle,
        "max_factor": args.obs_enhanced_color_jitter_factor,
    }


def make_fixed_visual_noise_params(args) -> Dict[str, Any]:
    if not visual_noise_enabled(args):
        return {}

    if args.obs_noise_type not in FIXED_VISUAL_NOISE_TYPES:
        return {}

    image_shape = (args.env_img_res, args.env_img_res, 3)

    return generate_fixed_noise_params(
        args.obs_noise_type,
        image_shape,
        max_shift_ratio=args.obs_image_shift_ratio,
        max_angle=args.obs_image_rotation_angle,
        max_factor=args.obs_enhanced_color_jitter_factor,
        kernel_size=args.obs_blur_kernel_size,
        sigma=args.obs_blur_sigma,
    )


def apply_observation_noise(
    observation: Dict[str, Any],
    args,
    fixed_noise_params: Dict[str, Any],
) -> Dict[str, Any]:
    if not visual_noise_enabled(args):
        return observation

    noisy = dict(observation)
    kwargs = make_visual_noise_kwargs(args)

    if args.obs_noise_apply_to in ("both", "full"):
        noisy["full_image"] = apply_visual_noise(
            noisy["full_image"],
            args.obs_noise_type,
            args.obs_noise,
            fixed_params=fixed_noise_params,
            **kwargs,
        )

    if args.obs_noise_apply_to in ("both", "wrist"):
        noisy["wrist_image"] = apply_visual_noise(
            noisy["wrist_image"],
            args.obs_noise_type,
            args.obs_noise,
            fixed_params=fixed_noise_params,
            **kwargs,
        )

    return noisy


def apply_action_noise_for_libero(action: np.ndarray, args) -> np.ndarray:
    if not action_noise_enabled(args):
        return action

    noisy_action = action.copy()

    if args.action_noise_robot_dims_only:
        noisy_action[:6] = apply_action_noise(
            noisy_action[:6],
            args.action_noise_type,
            args.action_noise_magnitude,
            salt_pepper_probability=args.action_salt_pepper_probability,
            impulse_probability=args.action_impulse_probability,
            impulse_magnitude=args.action_noise_magnitude,
        )
    else:
        noisy_action = apply_action_noise(
            noisy_action,
            args.action_noise_type,
            args.action_noise_magnitude,
            salt_pepper_probability=args.action_salt_pepper_probability,
            impulse_probability=args.action_impulse_probability,
            impulse_magnitude=args.action_noise_magnitude,
        )

    return np.clip(noisy_action, -1.0, 1.0)


def choose_policy_source(policy: str, chunk_id: int, mixed_first: str) -> str:
    if policy == "oft":
        return "oft"

    if policy == "adapter":
        return "adapter"

    if policy == "mixed":
        if mixed_first == "oft":
            return "oft" if chunk_id % 2 == 0 else "adapter"
        if mixed_first == "adapter":
            return "adapter" if chunk_id % 2 == 0 else "oft"

    raise ValueError(f"Unknown policy={policy}, mixed_first={mixed_first}")


def get_endpoint_for_source(args, source: str) -> str:
    if source == "oft":
        return args.oft_endpoint
    if source == "adapter":
        return args.adapter_endpoint

    raise ValueError(f"Unknown source: {source}")


def get_task_displacement_objects(task_id: int) -> Dict[str, List[str]]:
    if task_id not in LIBERO_10_SOURCE_TARGET:
        return {
            "source_objects": [],
            "target_objects": [],
        }

    task_info = LIBERO_10_SOURCE_TARGET[task_id]

    return {
        "source_objects": list(task_info.get("source_objects", [])),
        "target_objects": list(task_info.get("target_objects", [])),
    }


def get_source_to_target_map(task_id: int) -> Dict[str, str]:
    """Return source->target mapping for target/carry/source-stop behavior.

    If explicit source_to_target exists, use it.
    Otherwise, if there is exactly one target object, map all source objects to it.
    """
    if task_id not in LIBERO_10_SOURCE_TARGET:
        return {}

    task_info = LIBERO_10_SOURCE_TARGET[task_id]
    source_objects = list(task_info.get("source_objects", []))
    target_objects = list(task_info.get("target_objects", []))

    if "source_to_target" in task_info:
        return dict(task_info["source_to_target"])

    if len(target_objects) == 1:
        return {source: target_objects[0] for source in source_objects}

    return {}


def is_plate_body(body_name: str) -> bool:
    return "plate" in body_name.lower()


def is_mug_body(body_name: str) -> bool:
    return "mug" in body_name.lower()


def body_position_available(env, body_name: str) -> bool:
    try:
        _ = get_body_position(env, body_name)
        return True
    except Exception:
        return False


def xy_distance_between_bodies(env, body_a: str, body_b: str) -> float:
    pos_a = get_body_position(env, body_a)
    pos_b = get_body_position(env, body_b)

    return float(np.linalg.norm(pos_a[:2] - pos_b[:2]))


def z_distance_between_bodies(env, body_a: str, body_b: str) -> float:
    z_a = get_body_z(env, body_a)
    z_b = get_body_z(env, body_b)

    return float(abs(z_a - z_b))


def source_is_inside_or_near_target(
    env,
    source_body: str,
    target_body: str,
    args,
) -> bool:
    """Heuristic check for whether a source should be treated as on/inside a target."""
    try:
        xy_dist = xy_distance_between_bodies(env, source_body, target_body)
        z_dist = z_distance_between_bodies(env, source_body, target_body)
    except Exception:
        return False

    return (
        xy_dist <= args.target_displace_carry_xy_tol
        and z_dist <= args.target_displace_carry_z_tol
    )


def source_is_already_on_mapped_target(
    env,
    task_id: int,
    source_body: str,
    args,
) -> bool:
    """Return True if a source object is already on/near its mapped target.

    This prevents mugs/source objects from continuing to move after they have
    been successfully placed on plates/targets.
    """
    source_to_target = get_source_to_target_map(task_id)

    if source_body not in source_to_target:
        return False

    target_body = source_to_target[source_body]

    # Some mappings can be semantic regions, e.g. right_of_plate_1_main.
    # If there is no actual MuJoCo body with that name, skip this check.
    if not body_position_available(env, target_body):
        return False

    return source_is_inside_or_near_target(
        env=env,
        source_body=source_body,
        target_body=target_body,
        args=args,
    )


def plate_target_has_mug_on_it(
    env,
    task_id: int,
    target_body: str,
    args,
) -> bool:
    """Return True when a plate target already has its mapped mug on/near it.

    This prevents plate displacement after the mug has been placed.
    Uses the same XY/Z thresholds as target carry logic, so no .sh change is needed.
    """
    if not is_plate_body(target_body):
        return False

    if task_id not in LIBERO_10_SOURCE_TARGET:
        return False

    source_to_target = get_source_to_target_map(task_id)

    for source_body, mapped_target in source_to_target.items():
        if mapped_target != target_body:
            continue

        if not is_mug_body(source_body):
            continue

        if not body_position_available(env, source_body):
            continue

        if source_is_inside_or_near_target(env, source_body, target_body, args):
            return True

    return False


def get_sources_to_carry_with_target(
    env,
    task_id: int,
    target_body: str,
    active_displacements: Dict[str, Any],
    args,
) -> List[str]:
    """Find source objects that should move together with a displaced target."""
    if not args.target_displace_carry_sources:
        return []

    if task_id not in LIBERO_10_SOURCE_TARGET:
        return []

    source_to_target = get_source_to_target_map(task_id)
    carry_sources = []

    for source_body, mapped_target in source_to_target.items():
        if mapped_target != target_body:
            continue

        if source_body in active_displacements:
            continue

        if not body_has_free_joint(env, source_body):
            continue

        if source_is_inside_or_near_target(env, source_body, target_body, args):
            carry_sources.append(source_body)

    return carry_sources


def get_source_initial_z_map(env, task_id: int) -> Dict[str, float]:
    task_objects = get_task_displacement_objects(task_id)
    source_objects = task_objects["source_objects"]

    z_map = {}

    for body_name in source_objects:
        try:
            z_map[body_name] = get_body_z(env, body_name)
        except Exception:
            pass

    return z_map


def source_object_is_still_on_table(
    env,
    body_name: str,
    source_initial_z_map: Dict[str, float],
    args,
) -> bool:
    if not args.source_displace_require_on_table:
        return True

    if body_name not in source_initial_z_map:
        return False

    initial_z = source_initial_z_map[body_name]
    current_z = get_body_z(env, body_name)

    return current_z <= initial_z + args.source_displace_table_z_tol


def source_object_is_far_from_gripper(
    env,
    obs: Dict[str, Any],
    body_name: str,
    args,
) -> bool:
    object_pos = get_body_position(env, body_name)
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)

    dist = float(np.linalg.norm(object_pos - eef_pos))

    return dist >= args.source_displace_min_gripper_dist


def source_displacement_allowed(
    env,
    obs: Dict[str, Any],
    task_id: int,
    body_name: str,
    source_initial_z_map: Dict[str, float],
    args,
    log_file=None,
) -> bool:
    if source_is_already_on_mapped_target(
        env=env,
        task_id=task_id,
        source_body=body_name,
        args=args,
    ):
        log_message(
            f"[DISPLACE] Skip source {body_name}: already on/near mapped target.",
            log_file,
            debug=True,
        )
        return False

    if not source_object_is_still_on_table(env, body_name, source_initial_z_map, args):
        log_message(
            f"[DISPLACE] Skip source {body_name}: object appears lifted/off table.",
            log_file,
            debug=True,
        )
        return False

    if not source_object_is_far_from_gripper(env, obs, body_name, args):
        log_message(
            f"[DISPLACE] Skip source {body_name}: gripper/eef is too close.",
            log_file,
            debug=True,
        )
        return False

    return True


def sample_non_overlapping_trigger_steps(
    num_triggers: int,
    min_step: int,
    max_step: int,
    duration: int,
    max_episode_steps: int,
) -> List[int]:
    if num_triggers <= 0:
        return []

    if duration <= 0:
        raise ValueError("Displacement duration must be positive.")

    lo = max(0, int(min_step))

    if max_step < 0:
        hi = max_episode_steps - 1
    else:
        hi = min(max_episode_steps - 1, int(max_step))

    if lo > hi:
        return []

    candidates = np.arange(lo, hi + 1)
    np.random.shuffle(candidates)

    chosen = []

    for step in candidates:
        step = int(step)

        overlaps_existing = any(abs(step - prev_step) < duration for prev_step in chosen)

        if overlaps_existing:
            continue

        chosen.append(step)

        if len(chosen) >= num_triggers:
            break

    return sorted(chosen)


def add_displacement_events_for_group(
    env,
    group_name: str,
    object_names: List[str],
    num_triggers_per_object: int,
    min_trigger_step: int,
    max_trigger_step: int,
    displacement_magnitude: float,
    duration: int,
    max_steps: int,
    available_body_names: Set[str],
    displacement_schedule: Dict[int, List[Dict[str, Any]]],
    log_file=None,
):
    if num_triggers_per_object <= 0:
        log_message(
            f"[DISPLACE] {group_name}: disabled because num_triggers=0",
            log_file,
            debug=True,
        )
        return

    if displacement_magnitude <= 0.0:
        log_message(
            f"[DISPLACE] {group_name}: disabled because displacement magnitude=0",
            log_file,
            debug=True,
        )
        return

    if len(object_names) == 0:
        log_message(
            f"[DISPLACE] {group_name}: no objects found for this task",
            log_file,
            debug=True,
        )
        return

    for body_name in object_names:
        if body_name not in available_body_names:
            log_message(
                f"[DISPLACE] {group_name}: skipping missing MuJoCo body: {body_name}",
                log_file,
                debug=True,
            )
            continue

        if not body_has_free_joint(env, body_name):
            log_message(
                f"[DISPLACE] {group_name}: skipping static/non-free body: {body_name}",
                log_file,
                debug=True,
            )
            continue

        trigger_steps = sample_non_overlapping_trigger_steps(
            num_triggers=num_triggers_per_object,
            min_step=min_trigger_step,
            max_step=max_trigger_step,
            duration=duration,
            max_episode_steps=max_steps,
        )

        if len(trigger_steps) == 0:
            log_message(
                f"[DISPLACE] {group_name}: no valid trigger steps sampled for {body_name}",
                log_file,
                debug=True,
            )
            continue

        for step in trigger_steps:
            total_delta_xy = sample_random_xy_displacement(displacement_magnitude)
            delta_xy_per_step = total_delta_xy / float(duration)

            event = {
                "group": group_name,
                "body_name": body_name,
                "total_delta_xy": total_delta_xy,
                "delta_xy_per_step": delta_xy_per_step,
                "displacement_magnitude": displacement_magnitude,
                "duration": duration,
            }

            displacement_schedule[step].append(event)

        log_message(
            f"[DISPLACE] {group_name}: body={body_name} "
            f"triggers={trigger_steps} "
            f"magnitude={displacement_magnitude * 100:.2f}cm "
            f"duration={duration}",
            log_file,
            debug=True,
        )


def build_task_displacement_schedule(
    env,
    task_id: int,
    args,
    max_steps: int,
    log_file=None,
) -> Dict[int, List[Dict[str, Any]]]:
    displacement_schedule = defaultdict(list)

    if not object_displacement_enabled(args):
        return displacement_schedule

    if args.task_suite_name != "libero_10":
        raise ValueError(
            "Task-specific source/target displacement is currently implemented "
            "only for --task_suite_name libero_10."
        )

    if args.obj_displace_duration <= 0:
        raise ValueError("--obj_displace_duration must be positive.")

    task_objects = get_task_displacement_objects(task_id)
    available_body_names = set(name for name in list_body_names(env) if name is not None)

    add_displacement_events_for_group(
        env=env,
        group_name="source",
        object_names=task_objects["source_objects"],
        num_triggers_per_object=args.source_obj_displace_num_triggers,
        min_trigger_step=args.source_obj_displace_min_trigger_step,
        max_trigger_step=args.source_obj_displace_max_trigger_step,
        displacement_magnitude=args.source_obj_displace_magnitude,
        duration=args.obj_displace_duration,
        max_steps=max_steps,
        available_body_names=available_body_names,
        displacement_schedule=displacement_schedule,
        log_file=log_file,
    )

    add_displacement_events_for_group(
        env=env,
        group_name="target",
        object_names=task_objects["target_objects"],
        num_triggers_per_object=args.target_obj_displace_num_triggers,
        min_trigger_step=args.target_obj_displace_min_trigger_step,
        max_trigger_step=args.target_obj_displace_max_trigger_step,
        displacement_magnitude=args.target_obj_displace_magnitude,
        duration=args.obj_displace_duration,
        max_steps=max_steps,
        available_body_names=available_body_names,
        displacement_schedule=displacement_schedule,
        log_file=log_file,
    )

    total_events = sum(len(events) for events in displacement_schedule.values())

    log_message(
        f"[DISPLACE] Built schedule for task_id={task_id}: total_events={total_events}",
        log_file,
        debug=True,
    )

    return displacement_schedule


def schedule_from_manifest_event(manifest_event: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """Build a displacement schedule with exactly one precomputed event.

    Bypasses all random trigger-step/direction sampling entirely -- the manifest
    is the sole source of truth, so two separate processes (e.g. an Adapter run
    and an OFT run) reading the same manifest file get identical perturbations,
    independent of any other randomness/execution-order in either process.
    """
    duration = int(manifest_event["duration"])
    magnitude = float(manifest_event["magnitude"])
    delta_xy = np.asarray(manifest_event["delta_xy"], dtype=np.float64)
    delta_xy_per_step = delta_xy / float(duration)

    trigger_step = int(manifest_event["trigger_step"])

    event = {
        "group": manifest_event.get("group", "target"),
        "body_name": manifest_event["body_name"],
        "total_delta_xy": delta_xy,
        "delta_xy_per_step": delta_xy_per_step,
        "displacement_magnitude": magnitude,
        "duration": duration,
    }

    schedule = defaultdict(list)
    schedule[trigger_step].append(event)
    return schedule


def run_episode(
    env,
    task_id: int,
    task_description: str,
    initial_state: np.ndarray,
    args,
    episode_global_id: int,
    log_file=None,
    displacement_override: Dict[str, Any] = None,
    direct_bundle: Dict[str, Any] = None,
) -> Tuple[bool, list, list, Dict[str, Any]]:
    env.reset()
    obs = env.set_init_state(initial_state)

    source_initial_z_map = get_source_initial_z_map(env, task_id)

    action_queue = deque()
    replay_images = []
    action_log = []  # reproducibility check: (t, source, action) for every executed step, purely additive

    t = 0
    chunk_id = 0
    success = False
    max_steps = TASK_MAX_STEPS[args.task_suite_name]

    total_env_time = 0.0
    total_policy_time = 0.0
    policy_time_by_source = defaultdict(float)
    chunks_by_source = defaultdict(int)

    fixed_obs_noise_params = make_fixed_visual_noise_params(args)

    if displacement_override is not None:
        displacement_schedule = schedule_from_manifest_event(displacement_override)
        log_message(
            f"[DISPLACE] Using manifest event for task_id={task_id}: {displacement_override}",
            log_file,
            debug=True,
        )
    else:
        displacement_schedule = build_task_displacement_schedule(
            env=env,
            task_id=task_id,
            args=args,
            max_steps=max_steps,
            log_file=log_file,
        )

    active_displacements = {}

    # Stage-2 diagnostics: only populated when displacement_override is a single
    # precomputed manifest event. Purely additive -- does not affect control flow.
    displacement_diagnostics = None
    if displacement_override is not None:
        displacement_diagnostics = {
            "body_name": displacement_override["body_name"],
            "group": displacement_override.get("group", "target"),
            "requested_trigger_step": int(displacement_override["trigger_step"]),
            "requested_delta_xy": list(displacement_override["delta_xy"]),
            "requested_magnitude": float(displacement_override["magnitude"]),
            "fired": False,
            "cancelled_early": False,
            "never_reached": True,
            "episode_terminated_before_trigger": False,
            "position_before": None,
            "position_after": None,
            "realized_delta_xy": None,
        }

    while t < max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            t_env0 = time.perf_counter()
            obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
            total_env_time += time.perf_counter() - t_env0
            t += 1
            continue

        clean_observation = prepare_observation(obs)
        observation = apply_observation_noise(
            clean_observation,
            args,
            fixed_obs_noise_params,
        )

        if args.save_noisy_video:
            replay_images.append(observation["full_image"])
        else:
            replay_images.append(clean_observation["full_image"])

        if len(action_queue) == 0:
            source = choose_policy_source(
                policy=args.policy,
                chunk_id=chunk_id,
                mixed_first=args.mixed_first,
            )
            t_policy0 = time.perf_counter()

            if direct_bundle is not None and source == "oft":
                actions = call_policy_direct(
                    bundle=direct_bundle,
                    observation=observation,
                    task_description=task_description,
                )
            else:
                endpoint = get_endpoint_for_source(args, source)
                actions = call_policy_server(
                    endpoint=endpoint,
                    observation=observation,
                    task_description=task_description,
                    timeout=args.server_timeout,
                )

            t_policy1 = time.perf_counter()
            policy_time = t_policy1 - t_policy0

            total_policy_time += policy_time
            policy_time_by_source[source] += policy_time
            chunks_by_source[source] += 1

            actions = actions[: args.chunk_size]

            if len(actions) == 0:
                raise ValueError(
                    f"{source} server returned empty action chunk. "
                    f"Expected at least {args.chunk_size} actions."
                )

            for a in actions:
                action_queue.append((source, a))

            if args.print_timing:
                log_message(
                    f"[episode {episode_global_id}] "
                    f"t={t:04d} chunk={chunk_id:03d} source={source} "
                    f"policy_call={policy_time:.4f}s "
                    f"actions={actions.shape}",
                    log_file,
                )

            chunk_id += 1

        source, action = action_queue.popleft()
        action = process_action_for_libero(action)
        action = apply_action_noise_for_libero(action, args)
        action_log.append((t, source, action.tolist()))

        episode_step = t - args.num_steps_wait

        if episode_step in displacement_schedule:
            for event in displacement_schedule[episode_step]:
                body_name = event["body_name"]

                if body_name in active_displacements:
                    log_message(
                        f"[DISPLACE] Skipping event because body is already active: "
                        f"episode_step={episode_step} body={body_name}",
                        log_file,
                        debug=True,
                    )
                    continue

                if event["group"] == "source":
                    if not source_displacement_allowed(
                        env=env,
                        obs=obs,
                        task_id=task_id,
                        body_name=body_name,
                        source_initial_z_map=source_initial_z_map,
                        args=args,
                        log_file=log_file,
                    ):
                        continue

                if event["group"] == "target":
                    if plate_target_has_mug_on_it(
                        env=env,
                        task_id=task_id,
                        target_body=body_name,
                        args=args,
                    ):
                        log_message(
                            f"[DISPLACE] Skip target plate {body_name}: "
                            f"mapped mug is already on/near plate.",
                            log_file,
                            debug=True,
                        )
                        continue

                active_displacements[body_name] = {
                    "group": event["group"],
                    "remaining": event["duration"],
                    "delta_xy_per_step": event["delta_xy_per_step"],
                    "total_delta_xy": event["total_delta_xy"],
                    "displacement_magnitude": event["displacement_magnitude"],
                }

                if displacement_diagnostics is not None and body_name == displacement_diagnostics["body_name"]:
                    displacement_diagnostics["fired"] = True
                    displacement_diagnostics["never_reached"] = False
                    displacement_diagnostics["position_before"] = get_body_position(env, body_name).tolist()

                log_message(
                    f"[DISPLACE] Fired group={event['group']} "
                    f"episode_step={episode_step} "
                    f"body={body_name} "
                    f"total_delta_xy={event['total_delta_xy']} "
                    f"magnitude={event['displacement_magnitude'] * 100:.2f}cm "
                    f"duration={event['duration']}",
                    log_file,
                    debug=True,
                )

        for body_name in list(active_displacements.keys()):
            active = active_displacements[body_name]

            if active["group"] == "source":
                if not source_displacement_allowed(
                    env=env,
                    obs=obs,
                    task_id=task_id,
                    body_name=body_name,
                    source_initial_z_map=source_initial_z_map,
                    args=args,
                    log_file=log_file,
                ):
                    log_message(
                        f"[DISPLACE] Cancel active source displacement: {body_name}",
                        log_file,
                        debug=True,
                    )
                    if displacement_diagnostics is not None and body_name == displacement_diagnostics["body_name"]:
                        displacement_diagnostics["cancelled_early"] = True
                        displacement_diagnostics["position_after"] = get_body_position(env, body_name).tolist()
                    del active_displacements[body_name]
                    continue

            if active["group"] == "target":
                if plate_target_has_mug_on_it(
                    env=env,
                    task_id=task_id,
                    target_body=body_name,
                    args=args,
                ):
                    log_message(
                        f"[DISPLACE] Cancel active target plate displacement: "
                        f"{body_name}; mapped mug is already on/near plate.",
                        log_file,
                        debug=True,
                    )
                    del active_displacements[body_name]
                    continue

            delta_xy = active["delta_xy_per_step"]
            carry_sources = []

            if active["group"] == "target":
                carry_sources = get_sources_to_carry_with_target(
                    env=env,
                    task_id=task_id,
                    target_body=body_name,
                    active_displacements=active_displacements,
                    args=args,
                )

            moved = translate_body_xy(
                env,
                body_name,
                delta_xy,
                zero_velocity=args.obj_displace_zero_velocity,
            )

            if not moved:
                log_message(
                    f"[DISPLACE] Could not move {body_name}; body may be static/non-free.",
                    log_file,
                    debug=True,
                )
                if displacement_diagnostics is not None and body_name == displacement_diagnostics["body_name"]:
                    displacement_diagnostics["cancelled_early"] = True
                    displacement_diagnostics["position_after"] = get_body_position(env, body_name).tolist()
                del active_displacements[body_name]
                continue

            if active["group"] == "target":
                for source_body in carry_sources:
                    source_moved = translate_body_xy(
                        env,
                        source_body,
                        delta_xy,
                        zero_velocity=args.obj_displace_zero_velocity,
                    )

                    if source_moved:
                        log_message(
                            f"[DISPLACE] Carry source with target: "
                            f"target={body_name} source={source_body} delta_xy={delta_xy}",
                            log_file,
                            debug=True,
                        )

        t_env0 = time.perf_counter()
        obs, reward, done, info = env.step(action.tolist())
        total_env_time += time.perf_counter() - t_env0

        for body_name in list(active_displacements.keys()):
            active_displacements[body_name]["remaining"] -= 1

            if active_displacements[body_name]["remaining"] <= 0:
                if displacement_diagnostics is not None and body_name == displacement_diagnostics["body_name"] \
                        and displacement_diagnostics["position_after"] is None:
                    displacement_diagnostics["position_after"] = get_body_position(env, body_name).tolist()
                del active_displacements[body_name]

        if done:
            success = True
            break

        t += 1

    log_message(
        f"[episode {episode_global_id}] finished: "
        f"success={success} t={t} chunks={chunk_id} "
        f"oft_chunks={chunks_by_source['oft']} "
        f"adapter_chunks={chunks_by_source['adapter']} "
        f"total_policy_time={total_policy_time:.3f}s "
        f"oft_policy_time={policy_time_by_source['oft']:.3f}s "
        f"adapter_policy_time={policy_time_by_source['adapter']:.3f}s "
        f"total_env_time={total_env_time:.3f}s",
        log_file,
    )

    if displacement_diagnostics is not None:
        if displacement_diagnostics["never_reached"]:
            displacement_diagnostics["episode_terminated_before_trigger"] = True
        if displacement_diagnostics["position_before"] is not None and displacement_diagnostics["position_after"] is not None:
            before = displacement_diagnostics["position_before"]
            after = displacement_diagnostics["position_after"]
            displacement_diagnostics["realized_delta_xy"] = [after[0] - before[0], after[1] - before[1]]
        log_message(
            f"[DISPLACE][diagnostics] episode {episode_global_id}: {displacement_diagnostics}",
            log_file,
            debug=True,
        )

    return success, replay_images, action_log, displacement_diagnostics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--policy",
        choices=["oft", "adapter", "mixed"],
        default="mixed",
        help="Policy mode: oft, adapter, or mixed alternating chunks.",
    )
    parser.add_argument(
        "--mixed_first",
        choices=["oft", "adapter"],
        default="oft",
        help="For --policy mixed, choose which model supplies chunk 0.",
    )

    parser.add_argument("--task_suite_name", default="libero_10")
    parser.add_argument("--num_trials_per_task", type=int, default=1)
    parser.add_argument("--max_tasks", type=int, default=1)

    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--env_img_res", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--oft_endpoint", default="http://127.0.0.1:8001/act")
    parser.add_argument("--adapter_endpoint", default="http://127.0.0.1:8002/act")
    parser.add_argument("--server_timeout", type=int, default=120)

    parser.add_argument("--save_videos", action="store_true")
    parser.add_argument("--no_timing", action="store_true")

    parser.add_argument("--local_log_dir", default="./experiments/logs")
    parser.add_argument("--run_id_note", default=None)

    parser.add_argument(
        "--episode_list",
        default=None,
        help="Path to a JSON file listing exact [task_id, trial_index] pairs to run, "
        "in place of the default dense range(max_tasks) x range(num_trials_per_task).",
    )
    parser.add_argument(
        "--displacement_manifest",
        default=None,
        help="Path to a JSON file mapping '{task_id}_{trial_index}' -> a precomputed "
        "displacement event (trigger_step, delta_xy, body_name, magnitude, duration). "
        "When set, bypasses random trigger-step/direction sampling for episodes present "
        "in the manifest, so two separate runs (e.g. Adapter and OFT) reading the same "
        "manifest get identical perturbation realizations.",
    )
    parser.add_argument(
        "--direct_oft_checkpoint",
        default=None,
        help="Diagnostic: load an OFT policy in-process (no HTTP/msgpack) instead of "
        "calling --oft_endpoint, to isolate whether the server/serialization layer "
        "affects results vs. calling get_action() directly. Only affects --policy oft "
        "(and 'oft' chunks under --policy mixed).",
    )

    # Observation / visual noise
    parser.add_argument(
        "--obs_noise_type",
        choices=[
            "none",
            "gaussian",
            "salt_pepper",
            "blur",
            "image_shift",
            "image_rotation",
            "enhanced_color_jitter",
        ],
        default="none",
    )
    parser.add_argument("--obs_noise", type=float, default=0.0)
    parser.add_argument(
        "--obs_noise_apply_to",
        choices=["both", "full", "wrist"],
        default="both",
    )
    parser.add_argument("--obs_salt_pepper_probability", type=float, default=0.1)
    parser.add_argument("--obs_blur_kernel_size", type=int, default=5)
    parser.add_argument("--obs_blur_sigma", type=float, default=1.0)
    parser.add_argument("--obs_image_shift_ratio", type=float, default=0.1)
    parser.add_argument("--obs_image_rotation_angle", type=float, default=30.0)
    parser.add_argument("--obs_enhanced_color_jitter_factor", type=float, default=3.0)
    parser.add_argument("--save_noisy_video", action="store_true")

    # Action noise
    parser.add_argument(
        "--action_noise_type",
        choices=["none", "uniform", "gaussian", "constant", "salt_pepper", "impulse"],
        default="none",
    )
    parser.add_argument("--action_noise_magnitude", type=float, default=0.0)
    parser.add_argument("--action_salt_pepper_probability", type=float, default=0.1)
    parser.add_argument("--action_impulse_probability", type=float, default=0.05)
    parser.add_argument(
        "--action_noise_robot_dims_only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Task-aware source/target object displacement
    parser.add_argument("--source_obj_displace_num_triggers", type=int, default=0)
    parser.add_argument("--source_obj_displace_min_trigger_step", type=int, default=0)
    parser.add_argument("--source_obj_displace_max_trigger_step", type=int, default=-1)
    parser.add_argument("--source_obj_displace_magnitude", type=float, default=0.0)

    parser.add_argument("--target_obj_displace_num_triggers", type=int, default=0)
    parser.add_argument("--target_obj_displace_min_trigger_step", type=int, default=0)
    parser.add_argument("--target_obj_displace_max_trigger_step", type=int, default=-1)
    parser.add_argument("--target_obj_displace_magnitude", type=float, default=0.0)

    parser.add_argument("--obj_displace_duration", type=int, default=1)
    parser.add_argument(
        "--obj_displace_zero_velocity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Source-object displacement safety gates
    parser.add_argument(
        "--source_displace_require_on_table",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--source_displace_table_z_tol",
        type=float,
        default=0.025,
        help="Skip source displacement if object z rises more than this from initial z.",
    )
    parser.add_argument(
        "--source_displace_min_gripper_dist",
        type=float,
        default=0.08,
        help="Skip source displacement if gripper/eef is closer than this distance.",
    )

    # Target/container carry behavior
    parser.add_argument(
        "--target_displace_carry_sources",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--target_displace_carry_xy_tol",
        type=float,
        default=0.12,
        help="Carry/source-stop threshold: XY distance to target.",
    )
    parser.add_argument(
        "--target_displace_carry_z_tol",
        type=float,
        default=0.12,
        help="Carry/source-stop threshold: Z distance to target.",
    )

    # Debug
    parser.add_argument("--print_body_names", action="store_true")

    args = parser.parse_args()
    args.print_timing = not args.no_timing

    if args.obs_blur_kernel_size % 2 == 0:
        raise ValueError("--obs_blur_kernel_size must be odd, e.g., 3, 5, 7.")

    if args.obj_displace_duration <= 0:
        raise ValueError("--obj_displace_duration must be positive.")

    log_file = None

    try:
        log_file, local_log_filepath, run_id = setup_logging(args)

        set_seed_everywhere(args.seed)

        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite_name]()
        num_tasks = min(task_suite.n_tasks, args.max_tasks)

        total_episodes = 0
        total_successes = 0
        episode_records = []
        episode_records_path = os.path.join(args.local_log_dir, run_id + ".episode_records.pkl")

        log_message("=" * 80, log_file)
        log_message(f"Run ID: {run_id}", log_file)
        log_message(f"Local log file: {local_log_filepath}", log_file)
        log_message(f"Task suite: {args.task_suite_name}", log_file)
        log_message(f"Policy: {args.policy}", log_file)
        if args.policy == "mixed":
            log_message(f"Mixed first: {args.mixed_first}", log_file)
        log_message(f"Trials per task: {args.num_trials_per_task}", log_file)
        log_message(f"Max tasks: {num_tasks}", log_file)
        log_message(f"Chunk size: {args.chunk_size}", log_file)
        log_message(f"Seed: {args.seed}", log_file)
        log_message(f"Save videos: {args.save_videos}", log_file)
        log_message(f"Save noisy video: {args.save_noisy_video}", log_file)
        log_message(f"OFT endpoint: {args.oft_endpoint}", log_file)
        log_message(f"Adapter endpoint: {args.adapter_endpoint}", log_file)

        log_message("-" * 80, log_file)
        log_message(f"Obs noise type: {args.obs_noise_type}", log_file)
        log_message(f"Obs noise magnitude: {args.obs_noise}", log_file)
        log_message(f"Obs noise apply to: {args.obs_noise_apply_to}", log_file)
        log_message(f"Action noise type: {args.action_noise_type}", log_file)
        log_message(f"Action noise magnitude: {args.action_noise_magnitude}", log_file)
        log_message(f"Action noise robot dims only: {args.action_noise_robot_dims_only}", log_file)

        log_message("-" * 80, log_file)
        log_message(f"Source displace num triggers per object: {args.source_obj_displace_num_triggers}", log_file)
        log_message(f"Source displace min trigger step: {args.source_obj_displace_min_trigger_step}", log_file)
        log_message(f"Source displace max trigger step: {args.source_obj_displace_max_trigger_step}", log_file)
        log_message(f"Source displace magnitude: {args.source_obj_displace_magnitude}", log_file)
        log_message(f"Target displace num triggers per object: {args.target_obj_displace_num_triggers}", log_file)
        log_message(f"Target displace min trigger step: {args.target_obj_displace_min_trigger_step}", log_file)
        log_message(f"Target displace max trigger step: {args.target_obj_displace_max_trigger_step}", log_file)
        log_message(f"Target displace magnitude: {args.target_obj_displace_magnitude}", log_file)
        log_message(f"Object displace duration: {args.obj_displace_duration}", log_file)
        log_message(f"Object displace zero velocity: {args.obj_displace_zero_velocity}", log_file)
        log_message(f"Source require on table: {args.source_displace_require_on_table}", log_file)
        log_message(f"Source table z tol: {args.source_displace_table_z_tol}", log_file)
        log_message(f"Source min gripper dist: {args.source_displace_min_gripper_dist}", log_file)
        log_message(f"Target carry sources: {args.target_displace_carry_sources}", log_file)
        log_message(f"Target carry XY tol: {args.target_displace_carry_xy_tol}", log_file)
        log_message(f"Target carry Z tol: {args.target_displace_carry_z_tol}", log_file)
        log_message("=" * 80, log_file)

        if args.episode_list is not None:
            with open(args.episode_list) as f:
                episode_pairs = [tuple(p) for p in json.load(f)]
            task_to_trials = {}
            for tid, ep in episode_pairs:
                task_to_trials.setdefault(tid, []).append(ep)
            task_ids_to_run = list(task_to_trials.keys())
            log_message(
                f"Using --episode_list: {len(episode_pairs)} episodes across "
                f"{len(task_ids_to_run)} tasks (overrides max_tasks/num_trials_per_task).",
                log_file,
            )
        else:
            task_ids_to_run = list(range(num_tasks))
            task_to_trials = {tid: list(range(args.num_trials_per_task)) for tid in task_ids_to_run}

        if args.displacement_manifest is not None:
            with open(args.displacement_manifest) as f:
                displacement_manifest = json.load(f)
            log_message(
                f"Using --displacement_manifest: {len(displacement_manifest)} precomputed events.",
                log_file,
            )
        else:
            displacement_manifest = {}

        direct_bundle = None
        if args.direct_oft_checkpoint is not None:
            direct_bundle = load_direct_oft_bundle(
                checkpoint=args.direct_oft_checkpoint,
                chunk_size=args.chunk_size,
                seed=args.seed,
            )
            log_message(
                f"Using --direct_oft_checkpoint: {args.direct_oft_checkpoint} "
                "(in-process, no HTTP/msgpack, for the direct-vs-server diagnostic).",
                log_file,
            )

        for task_id in tqdm.tqdm(task_ids_to_run, desc="tasks"):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)

            env, task_description = get_libero_env(
                task,
                "openvla",
                resolution=args.env_img_res,
            )

            if args.print_body_names:
                names = [name for name in list_body_names(env) if name is not None]
                log_message("\n[BODY NAMES]\n" + "\n".join(names), log_file)
                env.close()
                return

            task_successes = 0

            try:
                for ep in task_to_trials[task_id]:
                    log_message("\n" + "-" * 80, log_file)
                    log_message(f"Task {task_id}: {task_description}", log_file)
                    log_message(f"Episode {ep}", log_file)

                    initial_state = initial_states[ep]
                    displacement_override = displacement_manifest.get(f"{task_id}_{ep}")

                    success, replay_images, action_log, displacement_diagnostics = run_episode(
                        env=env,
                        task_id=task_id,
                        task_description=task_description,
                        initial_state=initial_state,
                        args=args,
                        episode_global_id=total_episodes,
                        log_file=log_file,
                        displacement_override=displacement_override,
                        direct_bundle=direct_bundle,
                    )

                    episode_records.append({
                        "episode_global_id": total_episodes,
                        "task_id": task_id,
                        "trial_index": ep,
                        "task_description": task_description,
                        "success": success,
                        "num_steps": len(action_log),
                        "action_log": action_log,
                        "displacement_diagnostics": displacement_diagnostics,
                    })
                    with open(episode_records_path, "wb") as f:
                        pickle.dump(episode_records, f)

                    total_episodes += 1

                    if success:
                        total_successes += 1
                        task_successes += 1

                    log_message(f"Success: {success}", log_file)
                    log_message(
                        "Total success rate so far: "
                        f"{total_successes}/{total_episodes} = "
                        f"{100.0 * total_successes / total_episodes:.1f}%",
                        log_file,
                    )

                    if args.save_videos:
                        save_rollout_video(
                            replay_images,
                            total_episodes,
                            success=success,
                            task_description=task_description,
                            log_file=log_file,
                        )

            finally:
                try:
                    env.close()
                except Exception as e:
                    log_message(f"Env close warning: {e}", log_file)

                del env
                gc.collect()

            log_message(
                f"Task {task_id} success rate: "
                f"{task_successes}/{len(task_to_trials[task_id])}",
                log_file,
            )

        final_sr = total_successes / max(total_episodes, 1)

        log_message("\n" + "=" * 80, log_file)
        log_message("FINAL RESULTS", log_file)
        log_message(f"Policy: {args.policy}", log_file)
        if args.policy == "mixed":
            log_message(f"Mixed first: {args.mixed_first}", log_file)
        log_message(f"Total episodes: {total_episodes}", log_file)
        log_message(f"Total successes: {total_successes}", log_file)
        log_message(f"Overall success rate: {final_sr:.4f} ({100.0 * final_sr:.1f}%)", log_file)
        log_message("=" * 80, log_file)

    finally:
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()
