# engine.py

from pathlib import Path
import numpy as np

from PyQt6.QtGui import QImage, QTransform
from PyQt6.QtCore import Qt
from core.fsc_tone import apply_fsc_tone


from ConverterClaude.core.image_processor import ImageProcessor


class Engine:
    def __init__(self) -> None:
        self.processor = ImageProcessor()
        self.raw_path: Path | None = None

        # Negativ-Mapping
        self.neg_shape: tuple[int, int] | None = None
        self.preview_shape: tuple[int, int] | None = None
        self.scale_x: float | None = None
        self.scale_y: float | None = None

        # Positives Bild (Original + aktueller Crop)
        self.pos_full_orig: np.ndarray | None = None  # komplettes Positiv
        self.pos_crop_rect: tuple[int, int, int, int] | None = None  # (x,y,w,h) in Original

        # Preview für normalen Modus (gecroppter Ausschnitt)
        self.pos_preview_shape: tuple[int, int] | None = None  # (H, W) der Preview
        self.pos_scale_x: float | None = None  # Preview -> Crop
        self.pos_scale_y: float | None = None

        # Preview für Crop-Modus (full image)
        self.full_preview_shape: tuple[int, int] | None = None  # (H, W)
        self.full_scale_x: float | None = None  # Preview -> Original
        self.full_scale_y: float | None = None

        # View-Crop (aktueller Ausschnitt) + Analyse-Crop (für Konversion)
        self.pos_crop_rect: tuple[int, int, int, int] | None = None  # (x, y, w, h) – View/Export
        self.analysis_crop_rect: tuple[int, int, int, int] | None = None  # für Konversion/Recalculate

        # Tone-Slider
        self.gamma_slider: int = 0      # -100..100
        self.contrast_slider: int = 0   # -100..100
        self.cyan_slider: int = 0       # -100..100
        self.magenta_slider: int = 0    # -100..100
        self.yellow_slider: int = 0     # -100..100


    @staticmethod
    def _linear_to_srgb_u8(img: np.ndarray) -> np.ndarray:
        """
        lineares 0..1-RGB -> sRGB u8 (0..255)

        - nutzt echte sRGB-Transferfunktion (1/2.4 statt 1/2.2 + Toe)
        """
        img = np.clip(img, 0.0, 1.0)

        # echte sRGB EOTF
        a = 0.055
        threshold = 0.0031308
        img_srgb = np.where(
            img <= threshold,
            12.92 * img,
            (1 + a) * np.power(img, 1.0 / 2.4) - a,
        )

        img_srgb = np.clip(img_srgb, 0.0, 1.0)
        img_u8 = (img_srgb * 255.0 + 0.5).astype(np.uint8)
        return img_u8


    @staticmethod
    def _qimage_from_rgb_u8(arr: np.ndarray) -> QImage:
        """
        arr: HxWx3, uint8, RGB
        """
        h, w, ch = arr.shape
        assert ch == 3
        # Qt erwartet BGR? Nein: Format_RGB888 ist RGB in row-major
        bytes_per_line = 3 * w
        qimg = QImage(
            arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        # Kopie erzwingen, damit das NumPy-Array freigegeben werden kann
        return qimg.copy()

    # ---------- Public API ----------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "base_preview_np") and self.base_preview_np is not None:
            # Anzeige neu zeichnen
            self._refresh_tone_preview()


    def _make_negative_preview(self) -> QImage:
        """
        Aus dem aktuellen Negativ (ImageProcessor) eine sRGB-Preview (QImage)
        mit max. 1200px Kantenlänge erzeugen und Scale-Faktoren aktualisieren.
        """
        if self.processor.negative_linear is None:
            raise RuntimeError("Kein Negativ geladen.")

        neg = self.processor.get_negative_for_display()
        if neg is None:
            neg = self.processor.negative_linear

        if neg is None:
            raise RuntimeError("Claude lieferte kein negatives Bild")

        neg = np.asarray(neg, dtype=np.float32)
        self.neg_shape = neg.shape[:2]  # (H, W)

        neg_u8 = self._linear_to_srgb_u8(neg)
        qimg = self._qimage_from_rgb_u8(neg_u8)

        # auf max. 1200 px Kantenlänge skalieren
        max_dim = 1200
        w0 = qimg.width()
        h0 = qimg.height()
        if max(w0, h0) > max_dim:
            qimg = qimg.scaled(
                max_dim,
                max_dim,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        # Preview-Infos merken (für Mapping von Klick -> Original)
        w1 = qimg.width()
        h1 = qimg.height()
        self.preview_shape = (h1, w1)
        if self.neg_shape is not None:
            H, W = self.neg_shape
            self.scale_x = w1 / float(W)
            self.scale_y = h1 / float(H)
        else:
            self.scale_x = self.scale_y = None

        return qimg


    def _apply_tone_sliders(self, img: np.ndarray) -> np.ndarray:
        """
        Wendet Gamma, Kontrast, CMY auf ein lineares Positiv (0..1) an.
        img: H x W x 3, float32
        """
        out = img.astype(np.float32, copy=True)

        # Gamma: -100..100 -> Faktor ~ 0.5..1.5
        if self.gamma_slider != 0:
            g = 1.0 + (self.gamma_slider / 200.0)  # 0.5..1.5
            g = max(0.2, min(5.0, g))
            # gamma >1 -> bildet ab wie "flacher" / steiler Kurve, daher invertieren
            out = np.clip(out, 0.0, 1.0) ** (1.0 / g)

        # Kontrast um 0.5 herum
        if self.contrast_slider != 0:
            c = 1.0 + (self.contrast_slider / 100.0)  # 0.0..2.0
            mid = 0.5
            out = (out - mid) * c + mid

        # CMY: einfache Kanal-Skalierung
        def channel_factor(v: int) -> float:
            # -100..100 -> ~0.5..1.5
            return 1.0 - (v / 200.0)

        if self.cyan_slider != 0:
            f = channel_factor(self.cyan_slider)
            out[..., 0] *= f  # Rot

        if self.magenta_slider != 0:
            f = channel_factor(self.magenta_slider)
            out[..., 1] *= f  # Grün

        if self.yellow_slider != 0:
            f = channel_factor(self.yellow_slider)
            out[..., 2] *= f  # Blau

        return np.clip(out, 0.0, 1.0)


    def _make_cropped_preview(self) -> QImage:
        """
        Erstellt eine 1200px-Preview des aktuellen Positivs basierend auf pos_crop_rect.
        """
        if self.pos_full_orig is None:
            raise RuntimeError("Kein positives Bild vorhanden.")

        import numpy as np
        from PyQt6.QtCore import Qt

        H_full, W_full = self.pos_full_orig.shape[:2]

        # View-Crop bestimmen
        if self.pos_crop_rect is None:
            x, y, w, h = 0, 0, W_full, H_full
        else:
            x, y, w, h = self.pos_crop_rect

        x = max(0, min(W_full - 1, x))
        y = max(0, min(H_full - 1, y))
        w = max(1, min(W_full - x, w))
        h = max(1, min(H_full - y, h))

        crop = self.pos_full_orig[y:y + h, x:x + w, :]

        # Tone-Slider anwenden
        crop_adj = self._apply_tone_sliders(crop)

        # linear -> sRGB u8 -> QImage
        pos_u8 = self._linear_to_srgb_u8(crop_adj)

        qimg = self._qimage_from_rgb_u8(pos_u8)

        # auf max 1200px skalieren
        max_dim = 1200
        w0 = qimg.width()
        h0 = qimg.height()
        if max(w0, h0) > max_dim:
            qimg = qimg.scaled(
                max_dim,
                max_dim,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        # Skalen für spätere Crop-Mappings aktualisieren
        w1 = qimg.width()
        h1 = qimg.height()
        self.pos_preview_shape = (h1, w1)
        self.pos_scale_x = w1 / float(w)
        self.pos_scale_y = h1 / float(h)

        return qimg


    def reset_crop(self) -> QImage:
        if self.pos_full_orig is None:
            raise RuntimeError("Kein positives Bild vorhanden.")

        H, W = self.pos_full_orig.shape[:2]
        self.pos_crop_rect = (0, 0, W, H)

        return self._make_cropped_preview()


    def apply_geom_to_positive(self, fn):
        """
        Wendet eine geometrische Transformation (rot90/mirror) auf das
        volle Positiv und die zugehörigen Crop-Rects an und baut dann
        die Preview neu auf.
        """
        import math
        import numpy as np

        if self.pos_full_orig is None:
            raise RuntimeError("Kein positives Bild vorhanden.")

        pos = self.pos_full_orig
        H, W = pos.shape[:2]

        def _transform_rect(rect):
            if rect is None:
                return None

            x, y, w, h = rect
            if w <= 0 or h <= 0:
                return None

            # Alte Eckpunkte im Originalbild
            x0, x1 = x, x + w
            y0, y1 = y, y + h
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

            xs_new = []
            ys_new = []

            name = getattr(fn, "__name__", "")

            # neue Bildgröße nach der Transformation
            Hn, Wn = H, W
            if name in ("rot90_ccw", "rot90_cw"):
                Hn, Wn = W, H  # 90° tauscht H/W

            for xx, yy in corners:
                xx = max(0.0, min(float(W), float(xx)))
                yy = max(0.0, min(float(H), float(yy)))

                if name == "rot90_ccw":
                    # (x, y) -> (x', y') im CCW-Bild
                    x_new = yy
                    y_new = W - xx
                elif name == "rot90_cw":
                    x_new = H - yy
                    y_new = xx
                elif name == "rot180":
                    x_new = W - xx
                    y_new = H - yy
                elif name == "mirror_h":
                    x_new = W - xx
                    y_new = yy
                elif name == "mirror_v":
                    x_new = xx
                    y_new = H - yy
                else:
                    # unbekannte Transformation -> Rect unverändert
                    return rect

                xs_new.append(x_new)
                ys_new.append(y_new)

            # Bounding Box im neuen Koordinatensystem
            min_x = min(xs_new)
            max_x = max(xs_new)
            min_y = min(ys_new)
            max_y = max(ys_new)

            min_x = max(0, min(Wn - 1, int(math.floor(min_x))))
            min_y = max(0, min(Hn - 1, int(math.floor(min_y))))
            max_x = max(min_x + 1, min(Wn, int(math.ceil(max_x))))
            max_y = max(min_y + 1, min(Hn, int(math.ceil(max_y))))

            new_w = max(1, max_x - min_x)
            new_h = max(1, max_y - min_y)

            return (min_x, min_y, new_w, new_h)

        # Crop-Metadaten drehen/spiegeln
        self.pos_crop_rect = _transform_rect(self.pos_crop_rect)
        self.analysis_crop_rect = _transform_rect(self.analysis_crop_rect)

        # Volles Positiv drehen/spiegeln
        self.pos_full_orig = fn(pos)

        # Preview neu bauen
        return self._make_cropped_preview()


        # Crop-Metadaten drehen/spiegeln
        self.pos_crop_rect = _transform_rect(self.pos_crop_rect)
        self.analysis_crop_rect = _transform_rect(self.analysis_crop_rect)

        # Volles Positiv drehen/spiegeln
        self.pos_full_orig = fn(pos)

        # Preview neu bauen (1200px etc. wie gehabt)
        return self._make_cropped_preview()


    def load_raw(self, path: str) -> QImage:
        """
        RAW mit Claude laden und Negativ-Preview als QImage zurückgeben.
        Arbeitet ab hier nur noch mit einem verkleinerten Arbeitsbild
        (maximal 1200px Kantenlänge), um Konvertierung und Slider schneller
        zu machen.
        """
        from pathlib import Path
        import numpy as np
        import cv2

        self.raw_path = Path(path)
        self.manual_wb = False  # bei neuem Bild WB-Status zurücksetzen

        # RAW laden (vollauflösend in negative_linear)
        self.processor.load_raw(path)

        # ---- NEU: negative_linear auf Arbeitsgröße verkleinern ----
        neg = self.processor.negative_linear
        if neg is not None:
            neg = np.asarray(neg, dtype=np.float32)
            H, W = neg.shape[:2]
            max_dim = 1200

            long_edge = max(H, W)
            if long_edge > max_dim:
                scale = max_dim / float(long_edge)
                new_w = int(round(W * scale))
                new_h = int(round(H * scale))

                neg_small = cv2.resize(
                    neg,
                    (new_w, new_h),
                    interpolation=cv2.INTER_AREA,
                )

                self.processor.negative_linear = neg_small
                # WB / balanced werden später neu berechnet, daher hier zurücksetzen
                self.processor.negative_wb = None
                self.processor.negative_balanced = None

                print(f"DEBUG: Arbeitsbild {new_w}x{new_h} (Skalierung {scale:.3f})")
            else:
                print(f"DEBUG: Arbeitsbild bleibt {W}x{H} (<= {max_dim}px)")

        # Negativ-Preview erzeugen (nutzt jetzt das verkleinerte Arbeitsbild)
        return self._make_negative_preview()


    def set_wb_from_border_preview_rect(self, x_p: int, y_p: int, w_p: int, h_p: int) -> None:
        """
        Setzt WB vom Rand, basierend auf einem Rechteck in Preview-Koordinaten
        (also auf dem skalierten Negativ, das im UI angezeigt wird).
        """
        if self.processor.negative_linear is None:
            raise RuntimeError("Kein Negativ geladen.")

        if self.scale_x is None or self.scale_y is None or self.neg_shape is None:
            raise RuntimeError("Preview-Skalierung nicht bekannt.")

        # Preview-Rect -> Original-Koordinaten
        x0_full = int(x_p / self.scale_x)
        y0_full = int(y_p / self.scale_y)
        x1_full = int((x_p + w_p) / self.scale_x)
        y1_full = int((y_p + h_p) / self.scale_y)

        H, W = self.neg_shape
        x0_full = max(0, min(W - 1, x0_full))
        y0_full = max(0, min(H - 1, y0_full))
        x1_full = max(x0_full + 1, min(W, x1_full))
        y1_full = max(y0_full + 1, min(H, y1_full))

        # Patch aus dem Negativ holen (Display-Variante wie in simple_ui)
        neg_disp = self.processor.get_negative_for_display()
        if neg_disp is None:
            neg_disp = self.processor.negative_linear

        patch = neg_disp[y0_full:y1_full, x0_full:x1_full, :]
        if patch.size == 0:
            raise RuntimeError("WB-Patch ist leer.")

        self.processor.set_wb_from_border(patch)
        self.manual_wb = True


    def convert_to_positive(self, use_analysis_crop: bool = False) -> QImage:
        """
        Negativ -> Positiv mit Claude-Logik + Auto-Levels.

        - use_analysis_crop = False:
            normale Konvertierung (Convert-Button),
            nutzt ggf. pos_crop_rect als Analysebereich
        - use_analysis_crop = True:
            nutzt analysis_crop_rect (Recalculate)
        """
        if self.processor.negative_linear is None:
            raise RuntimeError("Kein Negativ geladen.")

        import numpy as np

        # Quelle (vollauflösend)
        if self.processor.negative_balanced is not None:
            source = self.processor.negative_balanced
            print("DEBUG: Using negative_balanced")
        elif self.processor.negative_wb is not None:
            source = self.processor.negative_wb
            print("DEBUG: Using negative_wb")
        else:
            source = self.processor.negative_linear
            print("DEBUG: Using negative_linear (NO WB!)")

        source = np.asarray(source, dtype=np.float32)
        H, W = source.shape[:2]
        print(f"DEBUG: Source range: {source.min():.4f} - {source.max():.4f}")

        # Analyse-Crop in Full-Res bestimmen
        if use_analysis_crop and self.analysis_crop_rect is not None:
            ax, ay, aw, ah = self.analysis_crop_rect
        elif self.pos_crop_rect is not None:
            ax, ay, aw, ah = self.pos_crop_rect
        else:
            ax, ay, aw, ah = 0, 0, W, H

        ax = max(0, min(W - 1, ax))
        ay = max(0, min(H - 1, ay))
        aw = max(1, min(W - ax, aw))
        ah = max(1, min(H - ay, ah))

        region = source[ay:ay + ah, ax:ax + aw, :]
        H_r, W_r = region.shape[:2]

        # Auto-WB (wenn nicht manuell)
        if not self.manual_wb:
            y0, y1 = 10, min(110, H_r)
            x0, x1 = 10, min(110, W_r)
            if y1 > y0 and x1 > x0:
                border_patch = region[y0:y1, x0:x1, :]
                try:
                    self.processor.set_wb_from_border(border_patch)
                except Exception as e:
                    print("WARN: set_wb_from_border failed:", e)

        # Crop an Claude übergeben
        self.processor.set_crop(ax, ay, aw, ah)

        # Auto-Density innerhalb Analyse-Region (wie gehabt)
        ph = min(50, H_r)
        pw = min(50, W_r)

        y_dark = int(H_r * 0.3)
        x_dark = int(W_r * 0.3)
        y_bright = int(H_r * 0.6)
        x_bright = int(W_r * 0.6)

        y_dark = max(0, min(H_r - ph, y_dark))
        x_dark = max(0, min(W_r - pw, x_dark))
        y_bright = max(0, min(H_r - ph, y_bright))
        x_bright = max(0, min(W_r - pw, x_bright))

        dark_patch = region[y_dark:y_dark + ph, x_dark:x_dark + pw, :]
        bright_patch = region[y_bright:y_bright + ph, x_bright:x_bright + pw, :]

        try:
            self.processor.set_density_balance_two_point(dark_patch, bright_patch)
        except Exception as e:
            print("WARN: density balance failed:", e)

        # Konvertierung
        pos = self.processor.convert()
        if pos is None:
            raise RuntimeError("Claude convert() lieferte kein positives Bild.")

        pos = np.clip(np.asarray(pos, dtype=np.float32), 0.0, 1.0)

        # FSC-Grundlook direkt in der Konversion:
        pos = apply_fsc_tone(
            pos,
            black_point=0,
            white_point=0,
            gamma=-30,
            shadows=0,
            highlights=0,
            saturation=120.0,  # 100 = neutral
)

        # Vollbild merken
        self.pos_full_orig = pos.copy()
        H_full, W_full = pos.shape[:2]

        # View-Crop initialisieren, falls noch nicht gesetzt
        if self.pos_crop_rect is None:
            self.pos_crop_rect = (ax, ay, aw, ah)

        # und Preview für den aktuellen Crop zurückgeben
        return self._make_cropped_preview()


    def _apply_auto_levels(
        self,
        pos: np.ndarray,
        crop_rect: tuple[int, int, int, int] | None,
    ) -> np.ndarray:
        """
        Auto-Levels ähnlich Film-Scan-Converter:

        - arbeitet auf float32, 0..1
        - nimmt pro Farbkanal (R,G,B) einen unteren/oberen Perzentilwert
          als Schwarz-/Weißpunkt
        - skaliert alle Pixel entsprechend (per-Kanal-Stretch)
        """

        import numpy as np

        if pos is None:
            return pos

        H_full, W_full = pos.shape[:2]

        # ----- Analyse-Region bestimmen -----
        if crop_rect is not None:
            x, y, w, h = crop_rect
            x = max(0, min(W_full - 1, x))
            y = max(0, min(H_full - 1, y))
            w = max(1, min(W_full - x, w))
            h = max(1, min(H_full - y, h))
        else:
            x, y, w, h = 0, 0, W_full, H_full

        region = pos[y:y + h, x:x + w, :]
        if region.size == 0:
            return pos

        # ----- Per-Kanal-Perzentile (ähnlich FSC hist_EQ) -----
        # Werte in [0,1] angenommen
        region_flat = region.reshape(-1, 3).astype(np.float32, copy=False)

        # leicht an FSC angelehnt: ~0.5% / 99% (statt 1/99 Luminanz)
        black_percentile = 0.5
        white_percentile = 99.0

        black = np.percentile(region_flat, black_percentile, axis=0)
        white = np.percentile(region_flat, white_percentile, axis=0)

        # Division absichern
        eps = 1e-4
        scale = 1.0 / np.maximum(white - black, eps)

        # ----- Per-Kanal-Stretch auf das ganze Bild -----
        out = pos.astype(np.float32, copy=False)
        out = (out - black[None, None, :]) * scale[None, None, :]
        out = np.clip(out, 0.0, 1.0)

        return out


    def recalculate(self) -> QImage:
        """
        Rechnet die Konvertierung neu, basierend auf dem aktuellen View-Crop.
        """
        if self.processor.negative_linear is None:
            raise RuntimeError("Kein Negativ geladen.")

        # aktuellen View-Crop als Analyse-Crop übernehmen
        if self.pos_crop_rect is not None:
            self.analysis_crop_rect = self.pos_crop_rect
        else:
            self.analysis_crop_rect = None

        return self.convert_to_positive(use_analysis_crop=True)


    def set_view_crop_from_negative_preview_rect(
        self,
        x_p: int,
        y_p: int,
        w_p: int,
        h_p: int,
    ) -> None:
        """
        Setzt den View-Crop (pos_crop_rect) anhand eines Rechtecks in der
        Negativ-Preview. Wird benutzt, wenn vor der Konvertierung gecroppt wird.
        """
        if self.neg_shape is None or self.scale_x is None or self.scale_y is None:
            raise RuntimeError("Negativ-Preview-Skalierung nicht bekannt.")

        H, W = self.neg_shape  # Originalgröße des Negativs

        # Preview -> Full (Negativ-Koordinaten)
        x0_full = int(x_p / self.scale_x)
        y0_full = int(y_p / self.scale_y)
        x1_full = int((x_p + w_p) / self.scale_x)
        y1_full = int((y_p + h_p) / self.scale_y)

        x0_full = max(0, min(W - 1, x0_full))
        y0_full = max(0, min(H - 1, y0_full))
        x1_full = max(x0_full + 1, min(W, x1_full))
        y1_full = max(y0_full + 1, min(H, y1_full))

        w_full = x1_full - x0_full
        h_full = y1_full - y0_full

        self.pos_crop_rect = (x0_full, y0_full, w_full, h_full)
        # Analyse-Crop setzen wir erst bei Recalculate explizit


    def set_wb_from_border_preview_point(self, x_p: int, y_p: int, patch_size: int = 20) -> QImage:
        """
        Setzt WB vom Rand basierend auf einem Klick in der Preview
        und gibt eine aktualisierte Negativ-Preview zurück.
        x_p, y_p sind Preview-Koordinaten (nach Skalierung).
        """
        if self.processor.negative_linear is None:
            raise RuntimeError("Kein Negativ geladen.")

        if self.scale_x is None or self.scale_y is None or self.neg_shape is None:
            raise RuntimeError("Preview-Skalierung nicht bekannt.")

        # Preview-Klick -> Original-Koordinaten (Zentrum des Patches)
        cx_full = int(x_p / self.scale_x)
        cy_full = int(y_p / self.scale_y)

        H, W = self.neg_shape
        cx_full = max(0, min(W - 1, cx_full))
        cy_full = max(0, min(H - 1, cy_full))

        half = patch_size // 2
        x0 = max(0, cx_full - half)
        y0 = max(0, cy_full - half)
        x1 = min(W, cx_full + half)
        y1 = min(H, cy_full + half)

        neg_disp = self.processor.get_negative_for_display()
        if neg_disp is None:
            neg_disp = self.processor.negative_linear

        patch = neg_disp[y0:y1, x0:x1, :]
        if patch.size == 0:
            raise RuntimeError("WB-Patch ist leer.")

        self.processor.set_wb_from_border(patch)
        self.manual_wb = True

        # aktualisierte Negativ-Preview zurückgeben
        return self._make_negative_preview()


    def apply_crop_from_full_preview_rect(self, x_p: int, y_p: int, w_p: int, h_p: int) -> QImage:
        """
        Setzt pos_crop_rect basierend auf einem Rechteck in der Full-Preview
        (Crop-Modus) und gibt danach die normale gecroppte Preview zurück.
        """
        if self.pos_full_orig is None:
            raise RuntimeError("Kein positives Bild vorhanden.")
        if self.full_scale_x is None or self.full_scale_y is None:
            raise RuntimeError("Full-Preview-Skalierung nicht bekannt.")

        H_full, W_full = self.pos_full_orig.shape[:2]

        # Preview-Rect -> Original-Koordinaten
        x0_full = int(x_p / self.full_scale_x)
        y0_full = int(y_p / self.full_scale_y)
        x1_full = int((x_p + w_p) / self.full_scale_x)
        y1_full = int((y_p + h_p) / self.full_scale_y)

        x0_full = max(0, min(W_full - 1, x0_full))
        y0_full = max(0, min(H_full - 1, y0_full))
        x1_full = max(x0_full + 1, min(W_full, x1_full))
        y1_full = max(y0_full + 1, min(H_full, y1_full))

        w_full = x1_full - x0_full
        h_full = y1_full - y0_full

        self.pos_crop_rect = (x0_full, y0_full, w_full, h_full)

        # normale gecroppte Preview generieren
        return self._make_cropped_preview()


    def get_full_preview_for_crop(self) -> QImage:
        """
        Liefert eine Preview des vollständigen Positivs für den Crop-Modus.
        """
        if self.pos_full_orig is None:
            raise RuntimeError("Kein positives Bild vorhanden.")

        pos = self.pos_full_orig
        H, W = pos.shape[:2]

        pos_u8 = self._linear_to_srgb_u8(pos)
        qimg = self._qimage_from_rgb_u8(pos_u8)

        max_dim = 1200
        w0 = qimg.width()
        h0 = qimg.height()
        if max(w0, h0) > max_dim:
            qimg = qimg.scaled(
                max_dim,
                max_dim,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        w1 = qimg.width()
        h1 = qimg.height()
        self.full_preview_shape = (h1, w1)
        self.full_scale_x = w1 / float(W)
        self.full_scale_y = h1 / float(H)

        return qimg

    # --- Export
    def export_jpeg(self, path: str) -> None:
        if self.pos_full_orig is None:
            raise RuntimeError("Kein positives Bild zum Export – bitte zuerst Convert ausführen.")

        H_full, W_full = self.pos_full_orig.shape[:2]
        if self.pos_crop_rect is None:
            x, y, w, h = 0, 0, W_full, H_full
        else:
            x, y, w, h = self.pos_crop_rect

        x = max(0, min(W_full - 1, x))
        y = max(0, min(H_full - 1, y))
        w = max(1, min(W_full - x, w))
        h = max(1, min(H_full - y, h))

        crop = self.pos_full_orig[y:y+h, x:x+w, :]
        pos_u8 = self._linear_to_srgb_u8(crop)
        qimg = self._qimage_from_rgb_u8(pos_u8)

        ok = qimg.save(path, "JPEG")
        if not ok:
            raise RuntimeError(f"Speichern als JPEG fehlgeschlagen: {path}")
