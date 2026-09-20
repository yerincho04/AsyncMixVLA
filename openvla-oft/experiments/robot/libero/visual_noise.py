import numpy as np
from typing import Union, Optional, Tuple
import cv2


def generate_gaussian_visual_noise(noise_magnitude: float, image_shape: tuple) -> np.ndarray:
    """Generate Gaussian distributed visual noise."""
    return np.random.normal(0, noise_magnitude, image_shape).astype(np.int16)


def generate_salt_pepper_visual_noise(image: np.ndarray, noise_probability: float = 0.1, 
                                     salt_value: int = 255, pepper_value: int = 0) -> np.ndarray:
    """Generate salt and pepper noise."""
    noisy_image = image.copy()
    noise_mask = np.random.random(image.shape[:2]) < noise_probability
    
    # Randomly choose between salt and pepper noise
    salt_mask = np.random.random(image.shape[:2]) < 0.5
    
    # Apply salt and pepper noise
    if len(image.shape) == 3:  # Color image
        noisy_image[noise_mask & salt_mask] = salt_value
        noisy_image[noise_mask & ~salt_mask] = pepper_value
    else:  # Grayscale image
        noisy_image[noise_mask & salt_mask] = salt_value
        noisy_image[noise_mask & ~salt_mask] = pepper_value
    
    return noisy_image


def generate_blur_noise(image: np.ndarray, blur_type: str = "gaussian", 
                       kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Blur"""
    if blur_type == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    elif blur_type == "motion":
        # Motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        return cv2.filter2D(image, -1, kernel)
    elif blur_type == "average":
        return cv2.blur(image, (kernel_size, kernel_size))
    else:
        raise ValueError(f"Unknown blur type: {blur_type}")


def generate_image_shift(image: np.ndarray, max_shift_ratio: float = 0.1, 
                         shift_params: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Random image shift towards upper-left."""
    h, w = image.shape[:2]
    
    if shift_params is not None:
        shift_x, shift_y = shift_params
    else:
        # Calculate maximum shift pixels
        max_shift_x = int(w * max_shift_ratio)
        max_shift_y = int(h * max_shift_ratio)
        
        # Generate random shift amount (negative values for upper-left shift)
        shift_x = -np.random.randint(0, max_shift_x + 1)
        shift_y = -np.random.randint(0, max_shift_y + 1)
    
    # Create translation matrix
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    
    # Apply affine transformation
    shifted_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return shifted_image


def generate_image_rotation(image: np.ndarray, max_angle: float = 30.0, 
                           rotation_angle: Optional[float] = None) -> np.ndarray:
    """Random counterclockwise image rotation.Rotated image"""
    h, w = image.shape[:2]
    
    if rotation_angle is not None:
        angle = rotation_angle
    else:
        # Generate random rotation angle (positive for counterclockwise)
        angle = np.random.uniform(0, max_angle)
    
    # Calculate rotation center
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation transformation
    rotated_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return rotated_image


def generate_enhanced_color_jitter(image: np.ndarray, max_factor: float = 3.0, 
                                   jitter_params: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
    """Enhanced color jitter (randomly change saturation, brightness, contrast and sharpness).
    
    Args:
        image: Original image
        max_factor: Maximum random factor (e.g. 3 means up to 300% increase or decrease)
        jitter_params: Pre-generated jitter parameters (brightness_factor, contrast_factor, saturation_factor, sharpness_factor)
    """
    if len(image.shape) != 3:
        return image  # No color jitter for grayscale images
    
    result = image.astype(np.float32)
    
    if jitter_params is not None:
        brightness_factor, contrast_factor, saturation_factor, sharpness_factor = jitter_params
    else:
        # Brightness adjustment factor
        brightness_factor = np.random.uniform(-max_factor, max_factor)
        # Contrast adjustment factor
        contrast_factor = np.random.uniform(1 - max_factor/3, 1 + max_factor/3)
        # Saturation adjustment factor
        saturation_factor = np.random.uniform(1 - max_factor/3, 1 + max_factor/3)
        # Sharpness adjustment factor
        sharpness_factor = np.random.uniform(1 - max_factor/3, 1 + max_factor/3)
    
    # Brightness adjustment (additive)
    brightness_delta = brightness_factor * 255 / max_factor  # Convert factor to pixel value change
    result = result + brightness_delta
    
    # Contrast adjustment (multiplicative)
    mean_val = np.mean(result)
    result = (result - mean_val) * contrast_factor + mean_val
    
    # Saturation adjustment (HSV space)
    hsv = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    
    # Sharpness adjustment (using Laplacian operator)
    if sharpness_factor != 1.0:
        # Create sharpening kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(result, -1, kernel)
        # Blend original and sharpened image
        result = result * (1 - (sharpness_factor - 1)) + sharpened * (sharpness_factor - 1)
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_visual_noise(
    image: np.ndarray,
    noise_type: str,
    noise_magnitude: float = 0.0,
    fixed_params: Optional[dict] = None,
    **kwargs
) -> np.ndarray:
    """Apply noise to image.
    
    Args:
        image: Original image
        noise_type: Noise type: "gaussian", "salt_pepper", "blur", 
                   "image_shift", "image_rotation", "enhanced_color_jitter"
        noise_magnitude: Noise magnitude
        fixed_params: Pre-generated fixed parameters for noise types that need to remain constant within episode
        **kwargs: Other noise type specific parameters
    
    Returns:
        Image with noise applied
    """
    if noise_magnitude <= 0 and noise_type not in ["blur", "image_shift", "image_rotation", "enhanced_color_jitter"]:
        return image
    
    if noise_type == "gaussian":
        noise = generate_gaussian_visual_noise(noise_magnitude, image.shape)
        noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy_image
    elif noise_type == "salt_pepper":
        noise_probability = kwargs.get('salt_pepper_probability', 0.1)
        salt_value = kwargs.get('salt_value', 255)
        pepper_value = kwargs.get('pepper_value', 0)
        return generate_salt_pepper_visual_noise(image, noise_probability, salt_value, pepper_value)
    elif noise_type == "blur":
        if fixed_params and 'blur_params' in fixed_params:
            kernel_size, sigma = fixed_params['blur_params']
        else:
            kernel_size = kwargs.get('kernel_size', 5)
            sigma = kwargs.get('sigma', noise_magnitude if noise_magnitude > 0 else 1.0)
        # Use Gaussian blur
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    elif noise_type == "image_shift":
        max_shift_ratio = kwargs.get('max_shift_ratio', 0.1)
        shift_params = fixed_params.get('shift_params') if fixed_params else None
        return generate_image_shift(image, max_shift_ratio, shift_params)
    elif noise_type == "image_rotation":
        max_angle = kwargs.get('max_angle', 30.0)
        rotation_angle = fixed_params.get('rotation_angle') if fixed_params else None
        return generate_image_rotation(image, max_angle, rotation_angle)
    elif noise_type == "enhanced_color_jitter":
        max_factor = kwargs.get('max_factor', 3.0)
        jitter_params = fixed_params.get('jitter_params') if fixed_params else None
        return generate_enhanced_color_jitter(image, max_factor, jitter_params)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def generate_fixed_noise_params(noise_type: str, image_shape: tuple, **kwargs) -> dict:
    """Pre-generate parameters for noise types that need fixed parameters.
    
    Args:
        noise_type: Noise type
        image_shape: Image shape (H, W, C)
        **kwargs: Noise type specific parameters
    
    Returns:
        Pre-generated parameter dictionary
    """
    h, w = image_shape[:2]
    params = {}
    
    if noise_type == "image_shift":
        max_shift_ratio = kwargs.get('max_shift_ratio', 0.1)
        max_shift_x = int(w * max_shift_ratio)
        max_shift_y = int(h * max_shift_ratio)
        shift_x = -np.random.randint(0, max_shift_x + 1)
        shift_y = -np.random.randint(0, max_shift_y + 1)
        params['shift_params'] = (shift_x, shift_y)
        
    elif noise_type == "image_rotation":
        max_angle = kwargs.get('max_angle', 30.0)
        angle = np.random.uniform(0, max_angle)
        params['rotation_angle'] = angle
        
    elif noise_type == "enhanced_color_jitter":
        max_factor = kwargs.get('max_factor', 3.0)
        brightness_factor = np.random.uniform(-max_factor, max_factor)
        contrast_factor = np.random.uniform(1 - max_factor/3, 1 + max_factor/3)
        saturation_factor = np.random.uniform(1 - max_factor/3, 1 + max_factor/3)
        sharpness_factor = np.random.uniform(1 - max_factor/3, 1 + max_factor/3)
        params['jitter_params'] = (brightness_factor, contrast_factor, saturation_factor, sharpness_factor)
        
    elif noise_type == "blur":
        # Blur noise type uses fixed kernel_size and sigma parameters
        kernel_size = kwargs.get('kernel_size', 5)
        sigma = kwargs.get('sigma', 1.0)
        params['blur_params'] = (kernel_size, sigma)
        
    return params