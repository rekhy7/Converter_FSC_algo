"""
Geometric transformations for images.
"""
import numpy as np


def rot90_ccw(img):
    """Rotate 90° counter-clockwise."""
    return np.rot90(img, k=1, axes=(0, 1)).copy()


def rot90_cw(img):
    """Rotate 90° clockwise."""
    return np.rot90(img, k=3, axes=(0, 1)).copy()


def rot180(img):
    """Rotate 180°."""
    return np.rot90(img, k=2, axes=(0, 1)).copy()


def mirror_h(img):
    """Mirror horizontally (flip left-right)."""
    return np.flip(img, axis=1).copy()


def mirror_v(img):
    """Mirror vertically (flip top-bottom)."""
    return np.flip(img, axis=0).copy()
