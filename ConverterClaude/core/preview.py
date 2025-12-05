import numpy as np

def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB to sRGB."""
    srgb = np.empty_like(linear, dtype=np.float32)

    mask = linear <= 0.0031308
    srgb[mask] = 12.92 * linear[mask]
    srgb[~mask] = 1.055 * np.power(linear[~mask], 1.0/2.4) - 0.055

    return np.clip(srgb, 0.0, 1.0)

def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear RGB."""
    linear = np.empty_like(srgb, dtype=np.float32)

    mask = srgb <= 0.04045
    linear[mask] = srgb[mask] / 12.92
    linear[~mask] = np.power((srgb[~mask] + 0.055) / 1.055, 2.4)

    return linear
