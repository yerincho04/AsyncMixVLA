import numpy as np
from typing import Union, Optional


def generate_uniform_action_noise(action_noise_magnitude: float, action_shape: tuple) -> np.ndarray:
    """Generate uniform distributed action noise."""
    return np.random.uniform(
        low=-action_noise_magnitude,
        high=action_noise_magnitude,
        size=action_shape
    )


def generate_constant_action_noise(action_noise_magnitude: float, action_shape: tuple) -> np.ndarray:
    """Generate constant action noise."""
    return np.ones(action_shape) * action_noise_magnitude


def generate_gaussian_action_noise(action_noise_magnitude: float, action_shape: tuple) -> np.ndarray:
    """Generate Gaussian distributed action noise."""
    return np.random.normal(0, action_noise_magnitude, action_shape)


def generate_salt_pepper_action_noise(action: np.ndarray, noise_probability: float = 0.1,
                                     salt_value: float = 1.0, pepper_value: float = -1.0) -> np.ndarray:
    """Generate Salt and Pepper action noise."""
    noisy_action = action.copy()
    noise_mask = np.random.random(action.shape) < noise_probability

    salt_mask = np.random.random(action.shape) < 0.5

    noisy_action[noise_mask & salt_mask] = salt_value
    noisy_action[noise_mask & ~salt_mask] = pepper_value

    return noisy_action


def generate_impulse_action_noise(action: np.ndarray, impulse_probability: float = 0.05,
                                 impulse_magnitude: float = 2.0) -> np.ndarray:
    """Generate impulse noise (sudden high-amplitude pulses)."""
    noisy_action = action.copy()
    impulse_mask = np.random.random(action.shape) < impulse_probability

    impulse_directions = np.random.choice([-1, 1], action.shape)
    impulse_noise = impulse_directions * impulse_magnitude

    noisy_action[impulse_mask] += impulse_noise[impulse_mask]

    return noisy_action


def apply_action_noise(
    action: np.ndarray,
    noise_type: str,
    action_noise_magnitude: float,
    action_shape: Optional[tuple] = None,
    **kwargs
) -> np.ndarray:
    """Apply noise to actions.

    Args:
        action: Original action
        action_noise_magnitude: Noise magnitude
        noise_type: Noise type: "uniform", "gaussian", "constant", "salt_pepper", "impulse"
        action_shape: Action shape (only needed for uniform, gaussian and constant types)
        **kwargs: Other noise type specific parameters

    Returns:
        Action with noise applied
    """
    if action_noise_magnitude <= 0 and noise_type not in ["salt_pepper", "random_scaling", "impulse"]:
        return action

    if noise_type == "uniform":
        if action_shape is None:
            action_shape = action.shape
        noise = generate_uniform_action_noise(action_noise_magnitude, action_shape)
        return action + noise
    elif noise_type == "gaussian":
        if action_shape is None:
            action_shape = action.shape
        noise = generate_gaussian_action_noise(action_noise_magnitude, action_shape)
        return action + noise
    elif noise_type == "constant":
        if action_shape is None:
            action_shape = action.shape
        noise = generate_constant_action_noise(action_noise_magnitude, action_shape)
        return action + noise
    elif noise_type == "salt_pepper":
        salt_pepper_probability = kwargs.get('salt_pepper_probability', 0.1)
        salt_value = kwargs.get('salt_value', action_noise_magnitude)
        pepper_value = kwargs.get('pepper_value', -action_noise_magnitude)
        return generate_salt_pepper_action_noise(action, salt_pepper_probability, salt_value, pepper_value)
    elif noise_type == "impulse":
        impulse_probability = kwargs.get('impulse_probability', 0.05)
        impulse_magnitude = kwargs.get('impulse_magnitude', action_noise_magnitude)
        return generate_impulse_action_noise(action, impulse_probability, impulse_magnitude)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
