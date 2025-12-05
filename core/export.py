# core/export.py
import numpy as np
import cv2

def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    low  = x <= 0.0031308
    high = ~low
    y = np.empty_like(x, dtype=np.float32)
    y[low]  = 12.92 * x[low]
    y[high] = (1+a) * np.power(x[high], 1/2.4) - a
    return y

def save_jpeg_srgb(img_linear: np.ndarray, path: str, quality: int = 95) -> None:
    """
    Save linear RGB float32 (0..1) to sRGB JPEG (8-bit). Uses OpenCV.
    """
    srgb = _linear_to_srgb(img_linear)
    u8 = (srgb * 255.0 + 0.5).astype(np.uint8)
    bgr = u8[..., ::-1]  # RGB->BGR for OpenCV
    cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
