import numpy as np


def apply_force_to_object(env, object_name: str, force: np.ndarray, torque: np.ndarray = None):
    """Apply an external force and optional torque to a named MuJoCo body.

    Kept for compatibility, but the recommended perturbation mode for your
    current experiment is direct XY displacement, not force.
    """
    body_id = env.sim.model.body_name2id(object_name)
    env.sim.data.xfrc_applied[body_id, :3] = force
    env.sim.data.xfrc_applied[body_id, 3:] = torque if torque is not None else [0.0, 0.0, 0.0]


def clear_force_on_object(env, object_name: str):
    """Zero out any applied force/torque on a named MuJoCo body."""
    body_id = env.sim.model.body_name2id(object_name)
    env.sim.data.xfrc_applied[body_id, :] = 0.0


def sample_random_force(
    magnitude: float,
    horizontal_only: bool = False,
    tilt_up: float = 0.0,
) -> np.ndarray:
    """Sample a random unit-direction force vector scaled by magnitude.

    Kept for compatibility. For controlled small motion, use
    sample_random_xy_displacement(...) + translate_body_xy(...).
    """
    if horizontal_only:
        direction = np.array([np.random.randn(), np.random.randn(), tilt_up], dtype=np.float64)
    else:
        direction = np.random.randn(3)
        direction[2] += tilt_up

    norm = np.linalg.norm(direction)

    if norm < 1e-12:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        direction = direction / norm

    return direction * magnitude


def list_body_names(env):
    """Return all body names in the current scene."""
    return [env.sim.model.body_id2name(i) for i in range(env.sim.model.nbody)]


def get_body_position(env, body_name: str) -> np.ndarray:
    """Return current world position of a MuJoCo body."""
    body_id = env.sim.model.body_name2id(body_name)
    return env.sim.data.body_xpos[body_id].copy()


def get_body_z(env, body_name: str) -> float:
    """Return current world z position of a MuJoCo body."""
    return float(get_body_position(env, body_name)[2])


def sample_random_xy_direction() -> np.ndarray:
    """Sample a random unit direction in the XY/table plane."""
    direction = np.array([np.random.randn(), np.random.randn()], dtype=np.float64)
    norm = np.linalg.norm(direction)

    if norm < 1e-12:
        return np.array([1.0, 0.0], dtype=np.float64)

    return direction / norm


def sample_random_xy_displacement(displacement_m: float) -> np.ndarray:
    """Sample a random XY displacement vector with exact requested magnitude."""
    return sample_random_xy_direction() * float(displacement_m)


def _get_joint_qpos_and_dof_lengths(model, joint_id: int):
    qpos_adr = int(model.jnt_qposadr[joint_id])
    dof_adr = int(model.jnt_dofadr[joint_id])

    if joint_id + 1 < model.njnt:
        qpos_end = int(model.jnt_qposadr[joint_id + 1])
        dof_end = int(model.jnt_dofadr[joint_id + 1])
    else:
        qpos_end = int(model.nq)
        dof_end = int(model.nv)

    qpos_len = qpos_end - qpos_adr
    dof_len = dof_end - dof_adr

    return qpos_adr, qpos_len, dof_adr, dof_len


def _find_free_joint_on_body(env, body_id: int):
    """Find a free joint directly attached to a body.

    A MuJoCo free joint has:
      qpos length = 7  -> x, y, z, qw, qx, qy, qz
      dof length  = 6  -> linear xyz + angular xyz velocity
    """
    model = env.sim.model

    num_joints = int(model.body_jntnum[body_id])

    if num_joints <= 0:
        return None

    first_joint_id = int(model.body_jntadr[body_id])

    for offset in range(num_joints):
        joint_id = first_joint_id + offset

        qpos_adr, qpos_len, dof_adr, dof_len = _get_joint_qpos_and_dof_lengths(
            model,
            joint_id,
        )

        if qpos_len == 7 and dof_len == 6:
            return {
                "body_id": body_id,
                "joint_id": joint_id,
                "qpos_adr": qpos_adr,
                "qpos_len": qpos_len,
                "dof_adr": dof_adr,
                "dof_len": dof_len,
            }

    return None


def get_free_joint_info_for_body(env, body_name: str):
    """Return free-joint info for body_name, or a parent body if needed.

    Some object-related body names may be child bodies. This walks upward until
    it finds a parent with a free joint. Static scene bodies return None.
    """
    model = env.sim.model
    body_id = int(model.body_name2id(body_name))

    current_body_id = body_id

    while current_body_id > 0:
        info = _find_free_joint_on_body(env, current_body_id)

        if info is not None:
            info["requested_body_id"] = body_id
            info["requested_body_name"] = body_name
            info["free_body_id"] = current_body_id
            info["free_body_name"] = model.body_id2name(current_body_id)
            return info

        current_body_id = int(model.body_parentid[current_body_id])

    return None


def body_has_free_joint(env, body_name: str) -> bool:
    """Return True if body_name or one of its parents has a free joint."""
    try:
        return get_free_joint_info_for_body(env, body_name) is not None
    except Exception:
        return False


def translate_body_xy(
    env,
    body_name: str,
    delta_xy: np.ndarray,
    zero_velocity: bool = True,
) -> bool:
    """Directly translate a free body in XY without changing height/orientation.

    This is the recommended perturbation for controlled small object movement.

    Args:
        env: LIBERO/MuJoCo environment.
        body_name: MuJoCo body name, e.g. alphabet_soup_1_main.
        delta_xy: np.array([dx, dy]) in metres.
        zero_velocity: If True, zero object velocity after the translation.

    Returns:
        True if the body was moved.
        False if the body is static or has no free joint.
    """
    info = get_free_joint_info_for_body(env, body_name)

    if info is None:
        return False

    sim = env.sim
    qpos_adr = info["qpos_adr"]
    dof_adr = info["dof_adr"]
    dof_len = info["dof_len"]

    delta_xy = np.asarray(delta_xy, dtype=np.float64)

    if delta_xy.shape != (2,):
        raise ValueError(f"Expected delta_xy shape (2,), got {delta_xy.shape}")

    # Free joint qpos starts with x, y, z. Keep z and quaternion unchanged.
    sim.data.qpos[qpos_adr + 0] += float(delta_xy[0])
    sim.data.qpos[qpos_adr + 1] += float(delta_xy[1])

    if zero_velocity:
        sim.data.qvel[dof_adr:dof_adr + min(dof_len, 6)] = 0.0

    sim.forward()

    return True


_G = 9.81


def get_body_mass_and_friction(env, body_name: str):
    """Read mass and average sliding-friction coefficient for a body.

    Kept for compatibility with force-based experiments.
    """
    sim = env.sim
    bid = sim.model.body_name2id(body_name)
    mass = float(sim.model.body_mass[bid])

    geom_ids = [
        g for g in range(sim.model.ngeom)
        if sim.model.geom_bodyid[g] == bid
    ]

    mu = float(np.mean([sim.model.geom_friction[g][0] for g in geom_ids])) if geom_ids else 0.5

    return mass, mu


def get_control_dt(env) -> float:
    """Return policy-level timestep, sim_dt multiplied by control substeps."""
    sim_dt = float(env.sim.model.opt.timestep)

    try:
        n_sub = max(1, int(round(env.env.control_timestep / sim_dt)))
    except Exception:
        n_sub = 1

    return sim_dt * n_sub


def force_for_displacement(
    mass: float,
    mu: float,
    dx: float,
    duration_steps: int,
    dt: float,
) -> float:
    """Closed-form force needed to displace a body by dx metres.

    Kept for compatibility. This does not prevent rotation/toppling.
    """
    t = duration_steps * dt

    if t <= 0 or dx <= 0:
        return 0.0

    C = 2.0 * dx / (mu * _G * t * t)

    return mass * mu * _G * (1.0 + np.sqrt(1.0 + 4.0 * C)) / 2.0