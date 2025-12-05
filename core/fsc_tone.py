import numpy as np

def apply_fsc_tone(
    img: np.ndarray,
    black_point: float,   # -100 .. +100 (FSC-Style)
    white_point: float,   # -100 .. +100
    gamma: float,         # -100 .. +100
    shadows: float,       # -100 .. +100
    highlights: float,    # -100 .. +100
    saturation: float,    # 0 .. 200, 100 = neutral
) -> np.ndarray:
    """
    img: float32, 0..1, shape (H, W, 3), RGB
    Alle Parameter grob an FSC angelehnt, aber auf 0..1 skaliert.
    """

    if img is None:
        return img

    arr = img.astype(np.float32, copy=True)
    arr = np.clip(arr, 0.0, 1.0)

    # ---------- 1) Black / White Point (FSC hist_EQ-Style) ----------
    # sensitivity wie in FSC
    sensitivity = 0.2

    # Per-Kanal-Percentile aus dem ganzen Bild (du kannst hier später eine Analyse-Region einsetzen)
    sample = arr.reshape(-1, 3)

    black_percentile = 0.5   # wie default_parameters['black_point_percentile'] :contentReference[oaicite:2]{index=2}
    white_percentile = 99.0  # wie default_parameters['white_point_percentile']

    black = np.percentile(sample, black_percentile, axis=0)
    white = np.percentile(sample, white_percentile, axis=0)

    # Slider: -100..+100 -> Offsets/Mulitplikatoren
    bp = float(black_point)
    wp = float(white_point)

    # Versatz der Schwarzwert-Lage (analog zu FSC black_offsets, nur auf 0..1)
    black_offsets = (bp / 100.0) * sensitivity - black
    arr = arr + black_offsets[None, None, :]

    # White-Multiplikator (analog zu FSC white_multipliers, aber mit 1 statt 65535)
    eps = 1e-6
    white_multipliers = (1.0 + (wp / 100.0) * sensitivity) / np.maximum(white, eps)
    arr = arr * white_multipliers[None, None, :]

    arr = np.clip(arr, 0.0, 1.0)

    # ---------- 2) Gamma + Shadows + Highlights (FSC exposure) ----------
    # Gamma
    g = float(gamma)
    gamma_exp = 2.0 ** (-g / 100.0)
    arr = np.power(arr, gamma_exp, dtype=np.float32)

    # Shadows/Highlights-Koeffizienten wie in FSC :contentReference[oaicite:3]{index=3}
    sh = float(shadows)
    hl = float(highlights)

    shadows_coeff = 4.15e-5 * sh * sh + 0.02185 * sh
    highlights_coeff = -4.15e-5 * hl * hl + 0.02185 * hl

    # Shadows: wirken unterhalb von ~0.75
    delta_shadow = np.minimum(arr - 0.75, 0.0)
    arr = arr + (shadows_coeff * (delta_shadow ** 2)) * arr

    # Highlights: wirken oberhalb von ~0.25
    delta_high = np.maximum(arr - 0.25, 0.0)
    arr = arr + (highlights_coeff * (delta_high ** 2)) * (1.0 - arr)

    arr = np.clip(arr, 0.0, 1.0)

    # ---------- 3) Saturation (FSC sat_adjust-Style) ----------
    sat = float(saturation)
    if sat != 100.0:
        sat_factor = sat / 100.0
        # Very simple RGB->HSV->RGB, nur für Sättigung
        # (wir implementieren es leichtgewichtig statt matplotlib.colors zu holen)
        r = arr[..., 0]
        g = arr[..., 1]
        b = arr[..., 2]

        maxc = np.max(arr, axis=-1)
        minc = np.min(arr, axis=-1)
        v = maxc
        s = np.zeros_like(v)
        mask = maxc > 0
        s[mask] = (maxc[mask] - minc[mask]) / maxc[mask]

        # Sättigung skalieren
        s = np.clip(s * sat_factor, 0.0, 1.0)

        # Hue berechnen
        # Achtung: einfache Implementation, reicht für Fotomaterial völlig
        h = np.zeros_like(v)
        denom = (maxc - minc) + 1e-6

        mask_r = (maxc == r) & (denom > 0)
        mask_g = (maxc == g) & (denom > 0)
        mask_b = (maxc == b) & (denom > 0)

        h[mask_r] = ((g - b)[mask_r] / denom[mask_r]) % 6.0
        h[mask_g] = ((b - r)[mask_g] / denom[mask_g]) + 2.0
        h[mask_b] = ((r - g)[mask_b] / denom[mask_b]) + 4.0

        h = h / 6.0  # 0..1

        # HSV -> RGB zurück
        i = np.floor(h * 6.0).astype(np.int32)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)

        r2 = np.zeros_like(v)
        g2 = np.zeros_like(v)
        b2 = np.zeros_like(v)

        i_mod = i % 6
        mask0 = i_mod == 0
        mask1 = i_mod == 1
        mask2 = i_mod == 2
        mask3 = i_mod == 3
        mask4 = i_mod == 4
        mask5 = i_mod == 5

        r2[mask0], g2[mask0], b2[mask0] = v[mask0], t[mask0], p[mask0]
        r2[mask1], g2[mask1], b2[mask1] = q[mask1], v[mask1], p[mask1]
        r2[mask2], g2[mask2], b2[mask2] = p[mask2], v[mask2], t[mask2]
        r2[mask3], g2[mask3], b2[mask3] = p[mask3], q[mask3], v[mask3]
        r2[mask4], g2[mask4], b2[mask4] = t[mask4], p[mask4], v[mask4]
        r2[mask5], g2[mask5], b2[mask5] = v[mask5], p[mask5], q[mask5]

        arr = np.stack([r2, g2, b2], axis=-1)
        arr = np.clip(arr, 0.0, 1.0)

    return arr
