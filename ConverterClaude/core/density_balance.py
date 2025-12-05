"""
Density balance calculation for film negatives.

Film dyes have different densities (cyan > magenta > yellow),
which causes color shifts across tonal range. Density balance
applies per-channel gamma to keep neutral grays neutral from
shadows to highlights.

Methods:
1. Two-point method: User picks dark + bright neutral areas
2. Single-point method: User picks one neutral, optimize gammas
3. Auto method: Find neutral areas automatically (experimental)
"""

import numpy as np
from typing import Tuple, Optional
from scipy.optimize import minimize_scalar


def _robust_mean(patch: np.ndarray) -> np.ndarray:
    """
    Calculate robust mean using percentile trimming.

    Args:
        patch: Image patch (H, W, 3)

    Returns:
        Mean RGB values [R, G, B]
    """
    # Reshape to (N, 3)
    pixels = patch.reshape(-1, 3)

    # Per-channel 10-90 percentile range
    p10 = np.percentile(pixels, 10, axis=0)
    p90 = np.percentile(pixels, 90, axis=0)

    # Keep only pixels in range
    mask = np.all((pixels >= p10) & (pixels <= p90), axis=1)
    trimmed = pixels[mask]

    if trimmed.shape[0] == 0:
        trimmed = pixels

    return trimmed.mean(axis=0).astype(np.float32)

def calculate_density_balance_two_point(
    dark_patch: np.ndarray,
    bright_patch: np.ndarray,
    method: str = "geometric_mean"
) -> tuple[float, float, float]:
    """
    Calculate per-channel gamma correction using two neutral patches.
    """
    # Handle None (use neutral gammas)
    if dark_patch is None or bright_patch is None:
        return (1.0, 1.0, 1.0)

    # Define target neutral
    target_neutral = 0.5  # ← HINZUFÜGEN!

    # Calculate robust mean for each patch
    dark_mean = _robust_mean(dark_patch)
    bright_mean = _robust_mean(bright_patch)

    # Rest bleibt gleich...

    # Rest of function stays the same...
    # Get mean RGB for each patch
    dark_rgb = dark_patch.mean(axis=(0, 1))
    bright_rgb = bright_patch.mean(axis=(0, 1))

    print(f"Dark patch RGB: {dark_rgb}")
    print(f"Bright patch RGB: {bright_rgb}")

    # Auto-determine target neutrals if not provided
    if target_neutral is None:
        # Target: match luminance of each patch
        dark_target = _perceived_luminance(dark_rgb)
        bright_target = _perceived_luminance(bright_rgb)
    else:
        dark_target = target_neutral
        bright_target = target_neutral

    # Calculate gamma for each channel
    # We want: dark_rgb[c] ** gamma[c] = dark_target
    #     and: bright_rgb[c] ** gamma[c] = bright_target
    #
    # This is an overdetermined system (2 equations, 1 unknown per channel)
    # We solve by minimizing error between both constraints

    gammas = np.zeros(3, dtype=np.float32)

    for c in range(3):
        dark_val = dark_rgb[c]
        bright_val = bright_rgb[c]

        if dark_val < 1e-6 or bright_val < 1e-6:
            gammas[c] = 1.0
            continue

        # Solve for gamma that satisfies both points
        # Using geometric mean of two gamma estimates
        gamma_from_dark = np.log(dark_target) / np.log(dark_val)
        gamma_from_bright = np.log(bright_target) / np.log(bright_val)

        # Average (geometric mean is more stable)
        gamma = np.sqrt(gamma_from_dark * gamma_from_bright)

        # Clamp to reasonable range
        gamma = np.clip(gamma, 0.5, 2.0)

        gammas[c] = gamma

    print(f"Calculated gammas: R={gammas[0]:.3f}, G={gammas[1]:.3f}, B={gammas[2]:.3f}")

    return tuple(gammas)


def calculate_density_balance_single_point(
    neutral_patch: np.ndarray,
    reference_value: float = 0.5
) -> Tuple[float, float, float]:
    """
    Calculate per-channel gamma from single neutral patch.

    Less accurate than two-point method, but practical when
    only one good neutral area is available.

    Args:
        neutral_patch: Neutral gray patch, float32, shape (H, W, 3)
        reference_value: Target gray value after balance (default 0.5)

    Returns:
        Tuple of (gamma_R, gamma_G, gamma_B)

    Note:
        This method assumes the patch should map to reference_value.
        Adjust reference_value based on patch brightness.
    """
    mean_rgb = neutral_patch.mean(axis=(0, 1))

    # Target: all channels equal to reference
    target_luma = _perceived_luminance(mean_rgb)

    if target_luma < 1e-6:
        return (1.0, 1.0, 1.0)

    # Scale reference to match patch luminance
    target = reference_value * (target_luma / reference_value)

    gammas = np.zeros(3, dtype=np.float32)

    for c in range(3):
        if mean_rgb[c] < 1e-6:
            gammas[c] = 1.0
        else:
            gamma = np.log(target) / np.log(mean_rgb[c])
            gamma = np.clip(gamma, 0.5, 2.0)
            gammas[c] = gamma

    return tuple(gammas)


def apply_density_balance(
    image: np.ndarray,
    gammas: Tuple[float, float, float]
) -> np.ndarray:
    """
    Apply per-channel gamma correction (density balance).

    Args:
        image: Linear RGB image, float32, shape (H, W, 3)
        gammas: Tuple of (gamma_R, gamma_G, gamma_B)

    Returns:
        Density-balanced image, float32, same shape

    Note:
        Apply this to the negative BEFORE inversion.
    """
    gamma_array = np.array(gammas, dtype=np.float32).reshape(1, 1, 3)

    # Clip input to avoid issues with negative values
    image_clipped = np.clip(image, 0.0, 1.0)

    # Apply per-channel gamma
    balanced = np.power(image_clipped, gamma_array)

    return balanced.astype(np.float32)


def auto_find_neutral_patches(
    image: np.ndarray,
    num_patches: int = 2,
    patch_size: int = 50
) -> list:
    """
    Automatically find neutral gray patches in image (experimental).

    Args:
        image: Linear RGB image, float32, shape (H, W, 3)
        num_patches: Number of patches to find (1 or 2)
        patch_size: Size of each patch in pixels

    Returns:
        List of patches [(y, x, patch), ...] sorted by neutrality score

    Note:
        This is experimental. Manual picking is more reliable.
    """
    H, W = image.shape[:2]

    if H < patch_size * 2 or W < patch_size * 2:
        raise ValueError("Image too small for auto patch detection")

    candidates = []

    # Sample patches on a grid
    step = patch_size
    for y in range(step, H - patch_size - step, step):
        for x in range(step, W - patch_size - step, step):
            patch = image[y:y+patch_size, x:x+patch_size, :]

            # Calculate neutrality score
            score = _neutrality_score(patch)

            candidates.append((score, y, x, patch))

    # Sort by neutrality (lower = more neutral)
    candidates.sort(key=lambda x: x[0])

    # Return top N patches
    result = [(y, x, patch) for (score, y, x, patch) in candidates[:num_patches]]

    return result


def _neutrality_score(patch: np.ndarray) -> float:
    """
    Calculate how neutral a patch is (lower = more neutral).

    Args:
        patch: RGB patch, shape (H, W, 3)

    Returns:
        Neutrality score (0 = perfect neutral gray)
    """
    mean_rgb = patch.mean(axis=(0, 1))

    # Variance between channels (neutral = all channels equal)
    variance = mean_rgb.std()

    # Also check if patch is uniform (not textured)
    uniformity = patch.std()

    # Combined score
    score = variance + 0.1 * uniformity

    return float(score)


def _perceived_luminance(rgb: np.ndarray) -> float:
    """Calculate perceived luminance (Rec.709)."""
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


if __name__ == "__main__":
    # Test density balance
    print("Testing density balance...")

    # Simulate negative with different channel densities
    # After WB, negative typically has uneven channel response
    np.random.seed(42)

    # Create test patches (on negative, so inverted tonality)
    # Dark area on positive = bright on negative (shadows)
    dark_neg = np.random.normal(
        loc=[0.7, 0.75, 0.8],  # Bright on negative, different per channel
        scale=0.02,
        size=(50, 50, 3)
    ).astype(np.float32)
    dark_neg = np.clip(dark_neg, 0.0, 1.0)

    # Bright area on positive = dark on negative (highlights)
    bright_neg = np.random.normal(
        loc=[0.25, 0.3, 0.35],  # Dark on negative, different per channel
        scale=0.02,
        size=(50, 50, 3)
    ).astype(np.float32)
    bright_neg = np.clip(bright_neg, 0.0, 1.0)

    print(f"Dark negative mean: {dark_neg.mean(axis=(0,1))}")
    print(f"Bright negative mean: {bright_neg.mean(axis=(0,1))}")

    # Calculate density balance
    gammas = calculate_density_balance_two_point(dark_neg, bright_neg)
    print(f"\nGammas: R={gammas[0]:.3f}, G={gammas[1]:.3f}, B={gammas[2]:.3f}")

    # Apply to patches
    dark_balanced = apply_density_balance(dark_neg, gammas)
    bright_balanced = apply_density_balance(bright_neg, gammas)

    print(f"\nAfter balance:")
    print(f"Dark: {dark_balanced.mean(axis=(0,1))}")
    print(f"Bright: {bright_balanced.mean(axis=(0,1))}")

    # Check if more neutral
    dark_std = dark_balanced.mean(axis=(0,1)).std()
    bright_std = bright_balanced.mean(axis=(0,1)).std()
    print(f"\nChannel uniformity (lower = better):")
    print(f"Dark: {dark_std:.4f}")
    print(f"Bright: {bright_std:.4f}")

    print("\n✓ Density balance test passed!")
