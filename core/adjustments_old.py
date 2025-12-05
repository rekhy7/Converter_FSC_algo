import numpy as np
import cv2

from PIL import Image, ImageEnhance

def _highlight_rolloff(arr: np.ndarray, knee: float = 0.85, strength: float = 0.7) -> np.ndarray:
    """
    Compress highlights smoothly above 'knee' to reduce clipping.
    arr: float32 0..255
    knee: 0..1, start of the shoulder (e.g. 0.85)
    strength: >0, bigger = stronger compression
    """
    x = np.clip(arr / 255.0, 0.0, 1.0)
    if knee <= 0.0 or knee >= 1.0 or strength <= 0.0:
        return arr

    y = x.copy()
    mask = x > knee
    if np.any(mask):
        t = (x[mask] - knee) / (1.0 - knee)  # 0..1 in highlight zone
        # compress towards 1 with a smooth shoulder
        t_new = 1.0 - (1.0 - t) / (1.0 + strength * t)
        y[mask] = knee + (1.0 - knee) * t_new

    return np.clip(y * 255.0, 0.0, 255.0).astype(arr.dtype)

def apply_adjustments_np(
    img_lin: np.ndarray,
    gamma: float,
    brightness: float,
    contrast: float,
    saturation: float,
    cmy: tuple[float, float, float],
    shadow: float,
    highlight: float,
    blackpoint: float,
    whitepoint: float,
) -> np.ndarray:
    """
    Float version of Negativ_Konverter19.apply_adjustments, adapted to 0..1.

    Parameters (same semantics as Tk version expected):
      gamma      ~ 0.5..2.5   (1.0 = neutral)
      brightness ~ 0.5..2.0   (1.0 = neutral)
      contrast   ~ 0.5..2.5   (1.0 = neutral)
      saturation ~ 0.0..4.0   (1.0 = neutral, 2.0 = your Tk default)
      cmy        ~ (c, m, y) additive offsets, typically -50..+50
      shadow     ~ -100..+100
      highlight  ~ -100..+100
      blackpoint ~ -50..+50
      whitepoint ~ -50..+50
    """

    # work in 0..255 like in Tk, then convert back
    arr = np.clip(img_lin.astype(np.float32), 0.0, 1.0) * 255.0

    # --- CMY fine-balance (additive, like in Tk) ---
    c, m, y = cmy
    if c or m or y:
        arr[..., 0] = np.clip(arr[..., 0] + c, 0.0, 255.0)
        arr[..., 1] = np.clip(arr[..., 1] + m, 0.0, 255.0)
        arr[..., 2] = np.clip(arr[..., 2] + y, 0.0, 255.0)

    # --- Shadows / Highlights ---
    if shadow != 0:
        f = 1.0 + (shadow / 200.0)
        arr = np.power(np.clip(arr, 0.0, 255.0) / 255.0, 1.0 / f) * 255.0

    if highlight != 0:
        f = 1.0 + (highlight / 200.0)
        arr = np.clip(arr * f, 0.0, 255.0)

    # --- Black / White point (symmetric, ±) ---
    bp = float(blackpoint)
    wp = float(whitepoint)
    if bp != 0 or wp != 0:
        # allow both positive and negative shifts
        low = bp            # base around 0
        high = 255.0 + wp   # base around 255
        # keep sane ordering
        if high <= low + 1.0:
            high = low + 1.0
        arr = np.clip(arr, low, high)
        arr = (arr - low) * (255.0 / (high - low))


    # --- Gamma via LUT (exactly like Tk) ---
    arr8 = arr.astype(np.uint8)
    if gamma != 1.0:
        lut = [int(((i / 255.0) ** (1.0 / gamma)) * 255.0) for i in range(256)]
        img_pil = Image.fromarray(arr8, "RGB").point(lut * 3)
    else:
        img_pil = Image.fromarray(arr8, "RGB")

    # --- Brightness / Contrast / Saturation via ImageEnhance ---
    if brightness != 1.0:
        img_pil = ImageEnhance.Brightness(img_pil).enhance(brightness)
    if contrast != 1.0:
        img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)
    if saturation != 1.0:
        img_pil = ImageEnhance.Color(img_pil).enhance(saturation)

        out = np.asarray(img_pil, dtype=np.float32)

    # final highlight roll-off to reduce clipping
    out = _highlight_rolloff(out, knee=0.85, strength=0.7)

    out = out / 255.0
    return np.clip(out, 0.0, 1.0)

def apply_detail_np(img_linear: np.ndarray,
                    nr_amount: float = 0.0,
                    sharpen_amount: float = 0.0) -> np.ndarray:
    """
    Detail processing in linear 0..1:

      - nr_amount: 0..1, stronger = more smoothing of grain
      - sharpen_amount: 0..1, stronger = more local edge contrast

    Noise reduction works in luma + chroma.
    Sharpening works on luma only to avoid color halos.
    """
    if img_linear is None:
        return img_linear

    x = np.clip(img_linear.astype(np.float32), 0.0, 1.0)
    if x.ndim != 3 or x.shape[2] != 3:
        return x

    h, w, _ = x.shape
    if h < 2 or w < 2:
        return x

    nr = float(np.clip(nr_amount, 0.0, 1.0))
    sh = float(np.clip(sharpen_amount, 0.0, 1.0))

    # split to Y + chroma
    R = x[..., 0]
    G = x[..., 1]
    B = x[..., 2]

    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    Cr = R - Y
    Cb = B - Y

    # --- Noise reduction: luma + chroma ---
    if nr > 0.0:
        # luma smoothing
        sigma_l = 0.6 + 2.0 * nr
        k_l = int(2 * round(2 * sigma_l) + 1)
        if k_l < 3:
            k_l = 3
        if k_l % 2 == 0:
            k_l += 1
        Y_blur = cv2.GaussianBlur(Y, (k_l, k_l), sigma_l)

        # chroma smoothing stronger
        sigma_c = 1.0 + 3.0 * nr
        k_c = int(2 * round(2 * sigma_c) + 1)
        if k_c < 3:
            k_c = 3
        if k_c % 2 == 0:
            k_c += 1
        Cr_blur = cv2.GaussianBlur(Cr, (k_c, k_c), sigma_c)
        Cb_blur = cv2.GaussianBlur(Cb, (k_c, k_c), sigma_c)

        mix_l = 0.2 + 0.3 * nr   # how much luma smoothing
        mix_c = 0.7 + 0.7 * nr   # how much chroma smoothing

        Y = (1.0 - mix_l) * Y + mix_l * Y_blur
        Cr = (1.0 - mix_c) * Cr + mix_c * Cr_blur
        Cb = (1.0 - mix_c) * Cb + mix_c * Cb_blur

    # reconstruct RGB from (Y, Cr, Cb)
    R = np.clip(Y + Cr, 0.0, 1.0)
    B = np.clip(Y + Cb, 0.0, 1.0)
    G = np.clip((Y - 0.2126 * R - 0.0722 * B) / 0.7152, 0.0, 1.0)

    x[..., 0] = R
    x[..., 1] = G
    x[..., 2] = B

    # --- Sharpen: unsharp mask on luma only ---
    if sh > 0.0:
        Y = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]

        sigma_s = 0.7 + 1.3 * sh
        k_s = int(2 * round(2 * sigma_s) + 1)
        if k_s < 3:
            k_s = 3
        if k_s % 2 == 0:
            k_s += 1

        Y_blur = cv2.GaussianBlur(Y, (k_s, k_s), sigma_s)
        detail = Y - Y_blur

        strength = 0.5 + 1.5 * sh  # up to ~3x detail
        Y_sharp = np.clip(Y + strength * detail, 0.0, 1.0)

        # apply luma change back to RGB by scaling along luma
        eps = 1e-6
        ratio = (Y_sharp + eps) / (Y + eps)
        ratio = np.clip(ratio, 0.0, 3.0)

        x[..., 0] = np.clip(x[..., 0] * ratio, 0.0, 1.0)
        x[..., 1] = np.clip(x[..., 1] * ratio, 0.0, 1.0)
        x[..., 2] = np.clip(x[..., 2] * ratio, 0.0, 1.0)

    return x
