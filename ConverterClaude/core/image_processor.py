"""
Image processor orchestrating the conversion pipeline.
"""

import numpy as np
from typing import Optional

from .raw_loader import load_raw_linear, get_raw_info, RawLoadError
from .white_balance import (
    calculate_wb_from_patch,
    apply_wb,
    FILM_PRESETS
)
from .density_balance import (
    calculate_density_balance_two_point,
    apply_density_balance
)
from .inversion import invert_negative_with_crop_analysis


class ImageProcessor:
    """
    Main orchestrator for film negative conversion pipeline.

    Workflow:
        1. load_raw() - Load and demosaic RAW file
        2. set_wb_from_border() - White balance from film border
        3. set_crop() - Set crop region for analysis
        4. set_density_balance_two_point() - Optional gamma correction
        5. convert() - Convert negative to positive
        6. get_positive() - Get result (cropped)
    """

    def __init__(self):
        # Image buffers
        self.negative_linear: Optional[np.ndarray] = None
        self.negative_wb: Optional[np.ndarray] = None
        self.negative_balanced: Optional[np.ndarray] = None
        self.positive_full: Optional[np.ndarray] = None

        # Metadata
        self.raw_info: dict = {}
        self.wb_gains: Optional[np.ndarray] = None
        self.density_gammas: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self.crop_rect: Optional[tuple[int, int, int, int]] = None

    def load_raw(self, filepath: str) -> dict:
        """
        Load RAW file and return info.

        Args:
            filepath: Path to RAW file

        Returns:
            Dictionary with camera info
        """
        self.negative_linear = load_raw_linear(filepath)
        self.raw_info = get_raw_info(filepath)

        # Reset pipeline
        self.negative_wb = None
        self.negative_balanced = None
        self.positive_full = None
        self.wb_gains = None
        self.density_gammas = (1.0, 1.0, 1.0)
        self.crop_rect = None

        return self.raw_info

    def get_negative_for_display(self) -> Optional[np.ndarray]:
        """Get current negative (for preview)."""
        if self.negative_balanced is not None:
            return self.negative_balanced
        if self.negative_wb is not None:
            return self.negative_wb
        return self.negative_linear

    def set_wb_from_border(
        self,
        border_patch: np.ndarray,
        method: str = "gray_world"
    ) -> np.ndarray:
        """
        Set white balance from film border patch.

        Args:
            border_patch: Film border region (linear RGB)
            method: "gray_world" or "max_white"

        Returns:
            WB gains [R, G, B]
        """
        if self.negative_linear is None:
            raise ValueError("No RAW loaded")

        # Calculate gains
        self.wb_gains = calculate_wb_from_patch(
            border_patch,
            method=method
        )

        # Apply WB to original negative
        self.negative_wb = apply_wb(self.negative_linear, self.wb_gains)

        # Reset density balance (needs to be recalculated with new WB)
        self.negative_balanced = None
        self.density_gammas = (1.0, 1.0, 1.0)

        return self.wb_gains

    def set_crop(self, x: int, y: int, width: int, height: int):
        """
        Set crop rectangle for analysis and output.

        Args:
            x, y: Top-left corner
            width, height: Crop dimensions
        """
        self.crop_rect = (x, y, width, height)

    def set_density_balance_two_point(
        self,
        dark_patch: np.ndarray,
        bright_patch: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Calculate and apply density balance from two neutral patches.

        Args:
            dark_patch: Dark neutral area (bright on negative)
            bright_patch: Bright neutral area (dark on negative)

        Returns:
            Gamma values (R, G, B)
        """
        if self.negative_wb is None:
            raise ValueError("White balance must be set first")

        # Calculate gammas
        self.density_gammas = calculate_density_balance_two_point(
            dark_patch,
            bright_patch
        )

        # Apply to WB negative
        self.negative_balanced = apply_density_balance(
            self.negative_wb,
            self.density_gammas
        )

        return self.density_gammas

    def convert(self) -> np.ndarray:
        """
        Convert negative to positive.
        Analyzes crop region but converts FULL image.

        Returns:
            Full positive image (use get_positive() for cropped version)
        """
        if self.negative_linear is None:
            raise ValueError("No RAW loaded")

        if self.crop_rect is None:
            raise ValueError("Crop not set")

        # Determine source image
        if self.negative_balanced is not None:
            source = self.negative_balanced
            print("DEBUG: Using negative_balanced")
        elif self.negative_wb is not None:
            source = self.negative_wb
            print("DEBUG: Using negative_wb")
        else:
            source = self.negative_linear
            print("DEBUG: Using negative_linear (NO WB!)")

        print(f"DEBUG: Source range: {source.min():.4f} - {source.max():.4f}")

        # Invert using crop analysis (converts FULL image!)
        self.positive_full = invert_negative_with_crop_analysis(
            source,
            self.crop_rect,
            stretch_method="percentile",
            black_clip=0.01,
            inversion_constant=0.18
        )

        return self.positive_full

    def get_positive(self) -> Optional[np.ndarray]:
        """
        Get positive image (cropped if crop is set).

        Returns:
            Cropped positive image for display
        """
        if self.positive_full is None:
            return None

        # Return cropped region
        if self.crop_rect is not None:
            x, y, w, h = self.crop_rect
            return self.positive_full[y:y+h, x:x+w, :].copy()

        return self.positive_full

    def get_positive_full(self) -> Optional[np.ndarray]:
        """
        Get full positive image (uncropped, for export).

        Returns:
            Full positive image
        """
        return self.positive_full

    def get_crop_region(self) -> Optional[np.ndarray]:
        """
        Get cropped region of current working image.

        Returns:
            Cropped region
        """
        if self.crop_rect is None:
            return None

        img = self.get_negative_for_display()
        if img is None:
            return None

        x, y, w, h = self.crop_rect
        return img[y:y+h, x:x+w, :].copy()
