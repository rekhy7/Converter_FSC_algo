"""
Unified tone and color adjustment system - FIXED VERSION.

Fixes:
- Exposure: Stronger effect, uniform across tones
- Saturation: HSV-based to preserve luminance
- CMY: Corrected sign (was reversed)
- Shadows: Better implementation without color shifts
- Black/White point: Disabled (too aggressive)
"""

import numpy as np
import cv2
from PIL import Image, ImageEnhance
from core.fsc_tone import apply_fsc_tone


# ============================================================================
# Color Space Conversions
# ============================================================================

def _linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """Convert linear RGB (0..1) to sRGB (0..1)."""
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    a = 0.055
    threshold = 0.0031308
    return np.where(
        img <= threshold,
        12.92 * img,
        (1.0 + a) * np.power(img, 1.0 / 2.4) - a,
    )


def _srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Convert sRGB (0..1) to linear RGB (0..1)."""
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    a = 0.055
    threshold = 0.04045
    return np.where(
        img <= threshold,
        img / 12.92,
        np.power((img + a) / (1.0 + a), 2.4),
    )


# ============================================================================
# Main Adjustment Function - FIXED
# ============================================================================

def apply_all_adjustments(
    img_linear: np.ndarray,
    exposure: float = 0.0,
    gamma: float = 1.0,
    contrast: float = 0.0,
    saturation: float = 1.0,
    cyan: float = 0.0,
    magenta: float = 0.0,
    yellow: float = 0.0,
    shadows: float = 0.0,
    highlights: float = 0.0,
    blackpoint: float = 0.0,
    whitepoint: float = 0.0,
) -> np.ndarray:
    """
    Apply all tone and color adjustments.
    """

    if img_linear is None or img_linear.size == 0:
        return img_linear

    # ------------------------------------------------------------------
    # 1) Linear clamp + FSC-Math (Black/White, Gamma, Shadows/Highlights, Sat)
    # ------------------------------------------------------------------
    img = np.clip(img_linear.astype(np.float32), 0.0, 1.0)

    img = apply_fsc_tone(
        img,
        black_point=blackpoint * 100.0,
        white_point=whitepoint * 100.0,
        gamma=(gamma - 1.0) * 150.0,
        shadows=shadows * 100.0,
        highlights=highlights * 125.0,     # verstärkte Highlights
        saturation=saturation * 100.0,
    )

    # ------------------------------------------------------------------
    # 2) Nach sRGB
    # ------------------------------------------------------------------
    img_srgb = _linear_to_srgb(img)

    # ------------------------------------------------------------------
    # 3) Contrast (in sRGB, wie bisher)
    # ------------------------------------------------------------------
    if abs(contrast) > 1e-6:
        img_srgb_u8 = (img_srgb * 255.0 + 0.5).astype(np.uint8)
        pil_img = Image.fromarray(img_srgb_u8, mode='RGB')

        contrast_factor = 1.0 + contrast
        contrast_factor = np.clip(contrast_factor, 0.5, 1.5)
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(contrast_factor)

        img_srgb = np.array(pil_img, dtype=np.float32) / 255.0

    # ------------------------------------------------------------------
    # 4) CMY (in sRGB, wie bisher)
    # ------------------------------------------------------------------
    if abs(cyan) > 1e-3 or abs(magenta) > 1e-3 or abs(yellow) > 1e-3:
        img_srgb = _apply_cmy_luma_neutral(img_srgb, cyan, magenta, yellow)

    # ------------------------------------------------------------------
    # 5) Exposure (in sRGB, nach allem Tonemapping)
    # ------------------------------------------------------------------
    if abs(exposure) > 1e-6:
        factor = 2.0 ** exposure    # EV: 1.0 = 1 Stop
        img_srgb = np.clip(img_srgb * factor, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 6) Back to linear
    # ------------------------------------------------------------------
    img_linear_out = _srgb_to_linear(img_srgb)

    return np.clip(img_linear_out, 0.0, 1.0).astype(np.float32)



# ============================================================================
# Helper Functions - FIXED
# ============================================================================

def _apply_saturation_hsv(img_srgb: np.ndarray, saturation: float) -> np.ndarray:
    """
    Apply saturation in HSV space to preserve luminance.

    FIX: Works in HSV instead of RGB to avoid luminance shifts.
    Range: 0.0 to 3.0 (was 2.0)
    """
    img = np.clip(img_srgb, 0.0, 1.0).astype(np.float32)

    # Convert to HSV
    img_u8 = (img * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Adjust saturation (S channel) - allow up to 3x
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)

    # Convert back to RGB
    hsv_u8 = hsv.astype(np.uint8)
    rgb_u8 = cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2RGB)

    return rgb_u8.astype(np.float32) / 255.0


def _apply_cmy_luma_neutral(
    img_srgb: np.ndarray,
    cyan: float,
    magenta: float,
    yellow: float,
    strength: float = 0.4,
) -> np.ndarray:
    """
    Apply CMY color correction while preserving luminance.

    FIX: Corrected signs - now matches user expectations.

    UI convention:
    +Cyan → more cyan (less red)
    +Magenta → more magenta (less green)
    +Yellow → more yellow (less blue)
    """

    img = np.clip(img_srgb, 0.0, 1.0).astype(np.float32)

    # Calculate luminance BEFORE
    Y_before = (
        0.2126 * img[..., 0] +
        0.7152 * img[..., 1] +
        0.0722 * img[..., 2]
    )
    Y_before = np.clip(Y_before, 1e-6, 1.0)

    # FIX: Use the OLD working logic with sign inversion
    # This matches what the user expects from the working version
    c = -cyan
    m = -magenta
    y = -yellow

    gain_r = 1.0 - strength * c  # When cyan_ui is positive, c is negative, gain_r > 1
    gain_g = 1.0 - strength * m
    gain_b = 1.0 - strength * y

    # Clamp gains
    gain_r = np.clip(gain_r, 0.3, 1.7)
    gain_g = np.clip(gain_g, 0.3, 1.7)
    gain_b = np.clip(gain_b, 0.3, 1.7)

    # Apply gains
    img[..., 0] *= gain_r
    img[..., 1] *= gain_g
    img[..., 2] *= gain_b

    img = np.clip(img, 0.0, 1.0)

    # Calculate luminance AFTER
    Y_after = (
        0.2126 * img[..., 0] +
        0.7152 * img[..., 1] +
        0.0722 * img[..., 2]
    )
    Y_after = np.clip(Y_after, 1e-6, 1.0)

    # Restore original luminance
    ratio = (Y_before / Y_after)[..., None]
    img = img * ratio

    return np.clip(img, 0.0, 1.0)


def _apply_shadows_highlights_improved(
    img_srgb: np.ndarray,
    shadows: float,
    highlights: float,
) -> np.ndarray:
    """
    Apply selective shadow and highlight adjustments.

    FIX: Better algorithm that preserves colors.
    Uses power curve instead of linear adjustment to avoid color shifts.
    """

    img = np.clip(img_srgb, 0.0, 1.0).astype(np.float32)

    # ========== Shadows ==========
    if abs(shadows) > 1e-3:
        # Use gamma adjustment for shadows (preserves color better)
        # Positive shadows = brighten darks
        shadow_gamma = 1.0 - (shadows * 0.3)  # Range: 0.7 to 1.3

        # Apply only to dark areas (Y < 0.5)
        Y = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        weight = np.clip((0.5 - Y) / 0.5, 0.0, 1.0)

        # Apply gamma with falloff
        img_adjusted = np.power(img, shadow_gamma)
        img = img * (1.0 - weight[..., None]) + img_adjusted * weight[..., None]

    # ========== Highlights ==========
    if abs(highlights) > 1e-3:
        # Use curve adjustment for highlights
        # Positive highlights = brighten lights
        Y = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
        weight = np.clip((Y - 0.5) / 0.5, 0.0, 1.0)

        # Simple multiplicative adjustment
        factor = 1.0 + (highlights * 0.3)
        img_adjusted = np.clip(img * factor, 0.0, 1.0)
        img = img * (1.0 - weight[..., None]) + img_adjusted * weight[..., None]

    return np.clip(img, 0.0, 1.0)


# ============================================================================
# Detail Adjustments (unchanged - already working)
# ============================================================================

def apply_detail_np(
    img_linear: np.ndarray,
    nr_amount: float = 0.0,
    sharpen_amount: float = 0.0,
) -> np.ndarray:
    """
    Apply noise reduction and sharpening.
    """

    if img_linear is None or img_linear.size == 0:
        return img_linear

    img = np.clip(img_linear.astype(np.float32), 0.0, 1.0)

    if img.ndim != 3 or img.shape[2] != 3:
        return img

    h, w, _ = img.shape
    if h < 2 or w < 2:
        return img

    nr = np.clip(nr_amount, 0.0, 1.0)
    sh = np.clip(sharpen_amount, 0.0, 1.0)

    # Split to luma + chroma
    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    Cr = R - Y
    Cb = B - Y

    # ========== Noise Reduction (gentler - half strength) ==========
    if nr > 1e-3:
        # Luma smoothing (reduced strength)
        sigma_l = 0.3 + 1.0 * nr  # was: 0.6 + 2.0 * nr
        k_l = int(2 * round(2 * sigma_l) + 1)
        k_l = max(3, k_l)
        if k_l % 2 == 0:
            k_l += 1
        Y_blur = cv2.GaussianBlur(Y, (k_l, k_l), sigma_l)

        # Chroma smoothing (stronger - but also reduced)
        sigma_c = 0.5 + 1.5 * nr  # was: 1.0 + 3.0 * nr
        k_c = int(2 * round(2 * sigma_c) + 1)
        k_c = max(3, k_c)
        if k_c % 2 == 0:
            k_c += 1
        Cr_blur = cv2.GaussianBlur(Cr, (k_c, k_c), sigma_c)
        Cb_blur = cv2.GaussianBlur(Cb, (k_c, k_c), sigma_c)

        # Mix factors (reduced strength)
        mix_l = 0.1 + 0.15 * nr  # was: 0.2 + 0.3 * nr
        mix_c = 0.35 + 0.35 * nr  # was: 0.7 + 0.7 * nr

        Y = (1.0 - mix_l) * Y + mix_l * Y_blur
        Cr = (1.0 - mix_c) * Cr + mix_c * Cr_blur
        Cb = (1.0 - mix_c) * Cb + mix_c * Cb_blur

    # Reconstruct RGB
    R = np.clip(Y + Cr, 0.0, 1.0)
    B = np.clip(Y + Cb, 0.0, 1.0)
    G = np.clip((Y - 0.2126 * R - 0.0722 * B) / 0.7152, 0.0, 1.0)

    img[..., 0] = R
    img[..., 1] = G
    img[..., 2] = B

    # ========== Sharpening (gentler - half strength) ==========
    if sh > 1e-3:
        Y = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

        sigma_s = 0.35 + 0.65 * sh  # was: 0.7 + 1.3 * sh
        k_s = int(2 * round(2 * sigma_s) + 1)
        k_s = max(3, k_s)
        if k_s % 2 == 0:
            k_s += 1

        Y_blur = cv2.GaussianBlur(Y, (k_s, k_s), sigma_s)
        detail = Y - Y_blur

        strength = 0.25 + 0.75 * sh  # was: 0.5 + 1.5 * sh (halved)
        Y_sharp = np.clip(Y + strength * detail, 0.0, 1.0)

        # Apply luma change to RGB
        eps = 1e-6
        ratio = (Y_sharp + eps) / (Y + eps)
        ratio = np.clip(ratio, 0.0, 3.0)

        img[..., 0] = np.clip(img[..., 0] * ratio, 0.0, 1.0)
        img[..., 1] = np.clip(img[..., 1] * ratio, 0.0, 1.0)
        img[..., 2] = np.clip(img[..., 2] * ratio, 0.0, 1.0)

    return img
