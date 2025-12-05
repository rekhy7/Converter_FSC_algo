"""
White balance calculation from film border.

Film negatives have an orange base/mask that must be neutralized.
The film border (unexposed area) represents pure film base and can
be used to derive white balance gains.

Methods:
1. Single-point WB: Click on film border, calculate gains
2. Two-point WB: Click two areas of film border, average them
3. Preset WB: Use saved values for known film stocks
"""

import numpy as np
from typing import Tuple, Optional

# Wie stark Blau „mitgeht“. 1.0 = wie bisher, <1.0 = weniger Blau.
BLUE_COMPENSATION = 1.2



def calculate_wb_from_patch(
    patch: np.ndarray,
    method: str = "gray_world"
) -> np.ndarray:
    """
    Calculate white balance gains from a film border patch.

    Args:
        patch: RGB patch from film border, float32, shape (H, W, 3)
               Should be linear (not gamma-corrected)
        method: WB calculation method:
                - "gray_world": Make average neutral
                - "max_white": Scale to brightest channel

    Returns:
        WB gains as array [R_gain, G_gain, B_gain], float32

    Example:
        >>> border_patch = negative[10:110, 10:110, :]  # 100x100 px
        >>> gains = calculate_wb_from_patch(border_patch)
        >>> wb_negative = negative * gains[None, None, :]
    """
    if patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError(f"Expected RGB patch, got shape {patch.shape}")

    if patch.size == 0:
        raise ValueError("Empty patch provided")

    # Robust mean: ignore outliers (dust, scratches)
    mean_rgb = _robust_mean(patch)

    if method == "gray_world":
        # Target: all channels have same luminance
        target_luma = _perceived_luminance(mean_rgb)

        gains = target_luma / (mean_rgb + 1e-6)

    elif method == "max_white":
        # Target: brightest channel = 1.0
        max_val = mean_rgb.max()
        gains = max_val / (mean_rgb + 1e-6)

    else:
        raise ValueError(f"Unknown WB method: {method}")

    # Clamp gains to reasonable range
    gains = np.clip(gains, 0.5, 3.0)

    return gains.astype(np.float32)

def calculate_wb_from_patch(
    patch: np.ndarray,
    method: str = "gray_world"
) -> np.ndarray:
    """
    Calculate white balance gains from film border patch.

    Args:
        patch: Film border region (linear RGB, 0-1), shape (H, W, 3)
        method: "gray_world" or "max_white"

    Returns:
        WB gains as numpy array [R, G, B]
    """

    if patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError(f"Expected RGB patch, got shape {patch.shape}")
    if patch.size == 0:
        raise ValueError("Empty patch provided")

    # Robuste Mittelwerte (schneidet Staub / Ausreißer ab)
    mean_rgb = _robust_mean(patch)
    eps = 1e-6

    if method == "gray_world":
        # Ziel: alle Kanäle auf ähnliche „Helligkeit“ bringen
        perceived_lum = _perceived_luminance(mean_rgb)
        gains = perceived_lum / (mean_rgb + eps)

        # *** WICHTIG: Blau leicht dämpfen, um Blaustich zu reduzieren ***
        # Gains[2] > 1 → wir bewegen uns nur ein Stück in diese Richtung
        gains[2] = 1.0 + (gains[2] - 1.0) * BLUE_COMPENSATION

    elif method == "max_white":
        # Ziel: alles auf den hellsten Kanal normalisieren
        max_channel = float(mean_rgb.max())
        gains = max_channel / (mean_rgb + eps)
    else:
        raise ValueError(f"Unknown WB method: {method}")

    # Clamp auf sinnvollen Bereich
    gains = np.clip(gains, 0.5, 3.0)

    return gains.astype(np.float32)


def apply_wb(image: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """
    Apply white balance gains to image.

    Args:
        image: Linear RGB image, float32, shape (H, W, 3)
        gains: WB gains [R, G, B], shape (3,)

    Returns:
        White-balanced image, float32, clipped to [0.0, 1.0]
    """
    if gains.shape != (3,):
        raise ValueError(f"Expected 3 gains, got shape {gains.shape}")

    # Broadcast gains over image
    balanced = image * gains[None, None, :]

    # Clip to valid range
    balanced = np.clip(balanced, 0.0, 1.0)

    return balanced.astype(np.float32)


def _robust_mean(patch: np.ndarray, percentile_range: Tuple[float, float] = (10, 90)) -> np.ndarray:
    """
    Calculate robust per-channel mean, ignoring outliers.

    Args:
        patch: RGB patch, shape (H, W, 3)
        percentile_range: Only include values in this percentile range

    Returns:
        Mean RGB values, shape (3,)
    """
    result = np.zeros(3, dtype=np.float32)

    for c in range(3):
        channel = patch[:, :, c].flatten()

        # Get percentile range
        lo, hi = np.percentile(channel, percentile_range)

        # Only use values in range
        mask = (channel >= lo) & (channel <= hi)
        valid = channel[mask]

        if valid.size > 0:
            result[c] = valid.mean()
        else:
            result[c] = channel.mean()  # Fallback

    return result


def _perceived_luminance(rgb: np.ndarray) -> float:
    """
    Calculate perceived luminance (Rec.709 coefficients).

    Args:
        rgb: RGB values, shape (3,)

    Returns:
        Luminance value (scalar)
    """
    # Rec.709/sRGB luminance weights
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


# Film stock presets (example values - adjust based on testing)
FILM_PRESETS = {
    "fuji_pro400h": np.array([1.2, 0.95, 1.15], dtype=np.float32),
    "kodak_portra400": np.array([1.15, 0.98, 1.20], dtype=np.float32),
    "kodak_portra160": np.array([1.18, 0.97, 1.22], dtype=np.float32),
    "kodak_ektar100": np.array([1.10, 1.00, 1.18], dtype=np.float32),
    "fuji_superia400": np.array([1.25, 0.93, 1.12], dtype=np.float32),
}


def get_film_preset(film_name: str) -> Optional[np.ndarray]:
    """
    Get WB preset for known film stock.

    Args:
        film_name: Film stock identifier (lowercase, underscores)

    Returns:
        WB gains if preset exists, None otherwise

    Example:
        >>> gains = get_film_preset("kodak_portra400")
        >>> if gains is not None:
        ...     balanced = apply_wb(negative, gains)
    """
    return FILM_PRESETS.get(film_name.lower())


def save_film_preset(film_name: str, gains: np.ndarray) -> None:
    """
    Save WB gains as preset for future use.

    Args:
        film_name: Film stock identifier
        gains: WB gains to save

    Note:
        In production, this should save to a JSON file.
        For now, just adds to in-memory dict.
    """
    FILM_PRESETS[film_name.lower()] = gains.astype(np.float32)


if __name__ == "__main__":
    # Test white balance calculation
    print("Testing white balance...")

    # Simulate film border (orange mask)
    # Typical film base: R=0.45, G=0.75, B=0.50 (orange)
    border_patch = np.random.normal(
        loc=[0.45, 0.75, 0.50],
        scale=0.02,
        size=(100, 100, 3)
    ).astype(np.float32)
    border_patch = np.clip(border_patch, 0.0, 1.0)

    print(f"Border mean: R={border_patch[:,:,0].mean():.3f}, "
          f"G={border_patch[:,:,1].mean():.3f}, "
          f"B={border_patch[:,:,2].mean():.3f}")

    # Calculate WB
    gains = calculate_wb_from_patch(border_patch)
    print(f"WB gains: R={gains[0]:.3f}, G={gains[1]:.3f}, B={gains[2]:.3f}")
