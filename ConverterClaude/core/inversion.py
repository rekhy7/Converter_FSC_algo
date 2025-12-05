"""
Film negative to positive inversion.
"""

import numpy as np
from typing import Optional, Tuple


def invert_negative(
    negative: np.ndarray,
    inversion_constant: float = 0.01,
    black_clip: float = 0.001,
    stretch_method: str = "percentile"
) -> np.ndarray:
    """Invert film negative to positive using division method."""

    negative_clipped = np.clip(negative, black_clip, 1.0)
    positive = inversion_constant / negative_clipped

    if stretch_method == "percentile":
        positive = _stretch_percentile(positive, low=1.0, high=99.0)
    elif stretch_method == "minmax":
        positive = _stretch_minmax(positive)
    elif stretch_method == "none":
        positive = np.clip(positive, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown stretch method: {stretch_method}")

    return positive.astype(np.float32)

def invert_negative_with_crop_analysis(
    negative: np.ndarray,
    crop_rect: tuple[int, int, int, int],
    stretch_method: str = "percentile",
    black_clip: float = 0.01,
    inversion_constant: float = 0.18
) -> np.ndarray:
    """
    Invert negative using division with better contrast.
    Analyzes inner 80% of crop to avoid border contamination.
    Converts FULL image, not just crop.

    Args:
        negative: Full negative image
        crop_rect: (x, y, w, h) region to analyze
        stretch_method: "percentile" (only option currently)
        black_clip: Minimum value to avoid division by zero
        inversion_constant: Numerator for division

    Returns:
        Full positive image (same size as input)
    """

    # Extract crop region for analysis
    x, y, w, h = crop_rect

    # Clip to avoid division by zero
    negative_clipped = np.clip(negative, black_clip, 1.0)

    # Invert FULL image: positive = constant / negative
    inverted = inversion_constant / negative_clipped
    inverted = np.clip(inverted, 0.0, 10.0)

    # Get crop for percentile analysis
    crop_inverted = inverted[y:y+h, x:x+w, :]

    # Analyze inner 80% zone (10% margin to ignore film border)
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)

    # Inner zone without borders
    inner_zone = crop_inverted[margin_y:h-margin_y, margin_x:w-margin_x, :]

    # Calculate percentiles from INNER ZONE only (global, not per-channel)
    p_low = float(np.percentile(inner_zone, 0.5))
    p_high = float(np.percentile(inner_zone, 99.5))

    print(f"DEBUG: Crop analysis - p_low={p_low:.4f}, p_high={p_high:.4f}")

    # Stretch FULL image based on inner zone analysis
    positive = (inverted - p_low) / (p_high - p_low + 1e-6)
    positive = np.clip(positive, 0.0, 1.0)

    return positive

def _stretch_percentile(
    image: np.ndarray,
    low: float = 1.0,
    high: float = 99.0
) -> np.ndarray:
    """Stretch image using percentile-based normalization."""

    result = np.empty_like(image, dtype=np.float32)

    for c in range(image.shape[2]):
        channel = image[:, :, c]

        p_low = np.percentile(channel, low)
        p_high = np.percentile(channel, high)

        if p_high > p_low + 1e-6:
            stretched = (channel - p_low) / (p_high - p_low)
        else:
            stretched = channel

        result[:, :, c] = np.clip(stretched, 0.0, 1.0)

    return result


def _stretch_minmax(image: np.ndarray) -> np.ndarray:
    """Stretch image using min/max normalization."""

    result = np.empty_like(image, dtype=np.float32)

    for c in range(image.shape[2]):
        channel = image[:, :, c]

        min_val = channel.min()
        max_val = channel.max()

        if max_val > min_val + 1e-6:
            stretched = (channel - min_val) / (max_val - min_val)
        else:
            stretched = channel

        result[:, :, c] = stretched

    return result
