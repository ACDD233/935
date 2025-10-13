"""
Image augmentation functions for rice leaf disease dataset
"""

import cv2
import numpy as np
import random
from typing import Tuple, Optional
import config


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image by specified angle
    
    Args:
        image: Input image
        angle: Rotation angle in degrees
        
    Returns:
        Rotated image
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated


def flip_image_horizontal(image: np.ndarray) -> np.ndarray:
    """Flip image horizontally"""
    return cv2.flip(image, 1)


def flip_image_vertical(image: np.ndarray) -> np.ndarray:
    """Flip image vertically"""
    return cv2.flip(image, 0)


def scale_image(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Scale image while maintaining aspect ratio and size
    
    Args:
        image: Input image
        scale_factor: Scaling factor (1.0 = no change)
        
    Returns:
        Scaled image with same dimensions as input
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    # Resize image
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    if scale_factor < 1.0:
        # Pad if scaling down
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        scaled = cv2.copyMakeBorder(
            scaled,
            pad_h, h - new_h - pad_h,
            pad_w, w - new_w - pad_w,
            cv2.BORDER_REFLECT
        )
    else:
        # Crop if scaling up
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        scaled = scaled[start_h:start_h + h, start_w:start_w + w]
    
    return scaled


def apply_gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Apply Gaussian blur to image
    
    Args:
        image: Input image
        kernel_size: Blur kernel size (must be odd)
        
    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def adjust_image_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Adjust image brightness
    
    Args:
        image: Input image (BGR format)
        factor: Brightness factor (1.0 = no change)
        
    Returns:
        Brightness-adjusted image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_image_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Adjust image contrast
    
    Args:
        image: Input image
        factor: Contrast factor (1.0 = no change)
        
    Returns:
        Contrast-adjusted image
    """
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def add_gaussian_noise(image: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Add Gaussian noise to image
    
    Args:
        image: Input image
        noise_level: Standard deviation of noise
        
    Returns:
        Noisy image
    """
    noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


def create_composite_augmentation(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Apply multiple augmentations in sequence
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (augmented_image, description)
    """
    img = image.copy()
    descriptions = []
    
    # Randomly apply different augmentations
    if random.random() > 0.5:
        angle = random.uniform(-20, 20)
        img = rotate_image(img, angle)
        descriptions.append(f"rot{angle:.1f}")
    
    if random.random() > 0.5:
        img = flip_image_horizontal(img)
        descriptions.append("flipH")
    
    if random.random() > 0.4:
        kernel = random.choice([3, 5])
        img = apply_gaussian_blur(img, kernel)
        descriptions.append(f"blur{kernel}")
    
    if random.random() > 0.4:
        factor = random.uniform(0.8, 1.2)
        img = adjust_image_brightness(img, factor)
        descriptions.append(f"br{factor:.2f}")
    
    desc = "+".join(descriptions) if descriptions else "combo"
    return img, desc


class ImageAugmentor:
    """Main class for handling image augmentation operations"""
    
    def __init__(self):
        """Initialize augmentor with random seed"""
        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
    
    def augment_image(self, image: np.ndarray, aug_type: str, 
                     params: Optional[dict] = None) -> Tuple[np.ndarray, str]:
        """
        Apply specified augmentation to image
        
        Args:
            image: Input image
            aug_type: Type of augmentation to apply
            params: Optional parameters for augmentation
            
        Returns:
            Tuple of (augmented_image, parameter_description)
        """
        if aug_type == "rotate":
            if params is None:
                angle = random.uniform(
                    config.AUGMENTATION_PARAMS["rotate"]["min_angle"],
                    config.AUGMENTATION_PARAMS["rotate"]["max_angle"]
                )
            else:
                angle = params.get("angle", 0)
            result = rotate_image(image, angle)
            return result, f"angle={angle:.1f}"
        
        elif aug_type == "flip_h":
            result = flip_image_horizontal(image)
            return result, "horizontal"
        
        elif aug_type == "flip_v":
            result = flip_image_vertical(image)
            return result, "vertical"
        
        elif aug_type == "scale":
            if params is None:
                scale = random.uniform(
                    config.AUGMENTATION_PARAMS["scale"]["min_scale"],
                    config.AUGMENTATION_PARAMS["scale"]["max_scale"]
                )
            else:
                scale = params.get("scale", 1.0)
            result = scale_image(image, scale)
            return result, f"scale={scale:.2f}"
        
        elif aug_type == "blur":
            if params is None:
                kernel = random.choice(config.AUGMENTATION_PARAMS["blur"]["kernel_sizes"])
            else:
                kernel = params.get("kernel", 3)
            result = apply_gaussian_blur(image, kernel)
            return result, f"kernel={kernel}"
        
        elif aug_type == "brightness":
            if params is None:
                factor = random.uniform(
                    config.AUGMENTATION_PARAMS["brightness"]["min_factor"],
                    config.AUGMENTATION_PARAMS["brightness"]["max_factor"]
                )
            else:
                factor = params.get("factor", 1.0)
            result = adjust_image_brightness(image, factor)
            return result, f"bright={factor:.2f}"
        
        elif aug_type == "contrast":
            if params is None:
                factor = random.uniform(
                    config.AUGMENTATION_PARAMS["contrast"]["min_factor"],
                    config.AUGMENTATION_PARAMS["contrast"]["max_factor"]
                )
            else:
                factor = params.get("factor", 1.0)
            result = adjust_image_contrast(image, factor)
            return result, f"contrast={factor:.2f}"
        
        elif aug_type == "noise":
            if params is None:
                level = random.uniform(
                    config.AUGMENTATION_PARAMS["noise"]["min_level"],
                    config.AUGMENTATION_PARAMS["noise"]["max_level"]
                )
            else:
                level = params.get("level", 10)
            result = add_gaussian_noise(image, level)
            return result, f"noise={level:.1f}"
        
        elif aug_type == "combo":
            result, desc = create_composite_augmentation(image)
            return result, desc
        
        else:
            # Return original image if augmentation type not recognized
            return image, "none"
    
    def get_augmentation_types(self) -> list:
        """Get list of available augmentation types"""
        return config.AUGMENTATION_TYPES.copy()
