"""
RAW file loading and demosaicing.

Loads RAW files (RAF, DNG, NEF, ARW, CR2, CR3) and returns
linear RGB in scene-referred color space.

Uses rawpy (LibRaw) with:
- Markesteijn demosaicing for Fuji X-Trans sensors
- AHD for Bayer sensors (default)
- No color correction, no white balance applied
- Output is linear (gamma 1.0)
"""

import numpy as np
import rawpy
from pathlib import Path


class RawLoadError(Exception):
    """Raised when RAW file cannot be loaded or processed."""
    pass


def load_raw_linear(filepath: str | Path) -> np.ndarray:
    """
    Load RAW file and return linear RGB array in scene-referred space.

    Args:
        filepath: Path to RAW file (.RAF, .DNG, .NEF, .ARW, .CR2, .CR3)

    Returns:
        Linear RGB array, float32, shape (H, W, 3), range [0.0, 1.0]
        Color space: Camera native (scene-referred)

    Raises:
        RawLoadError: If file cannot be loaded or processed

    Example:
        >>> img = load_raw_linear("scan001.RAF")
        >>> img.shape
        (4000, 6000, 3)
        >>> img.dtype
        dtype('float32')
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise RawLoadError(f"File not found: {filepath}")

    try:
        with rawpy.imread(str(filepath)) as raw:
            # Determine demosaicing algorithm based on sensor type
            if _is_xtrans(raw):
                demosaic = rawpy.DemosaicAlgorithm.AHD  # Markesteijn for X-Trans
                # Note: rawpy uses 'AHD' but applies Markesteijn for X-Trans
            else:
                demosaic = rawpy.DemosaicAlgorithm.AHD  # AHD for Bayer

            # Process with minimal corrections
            rgb = raw.postprocess(
                demosaic_algorithm=demosaic,
                output_color=rawpy.ColorSpace.raw,  # No color correction
                gamma=(1, 1),                       # Linear (no gamma curve)
                no_auto_bright=True,                # No auto exposure
                use_camera_wb=False,                # No white balance
                use_auto_wb=False,
                output_bps=16                       # 16-bit output
            )

            # Convert to float32 [0.0, 1.0]
            linear = rgb.astype(np.float32) / 65535.0

            return linear

    except rawpy.LibRawError as e:
        raise RawLoadError(f"Failed to process RAW file: {e}")
    except Exception as e:
        raise RawLoadError(f"Unexpected error loading RAW: {e}")


def _is_xtrans(raw: rawpy.RawPy) -> bool:
    """
    Check if sensor is Fuji X-Trans (6x6 pattern).

    Args:
        raw: Opened rawpy object

    Returns:
        True if X-Trans sensor, False otherwise
    """
    # X-Trans has 6x6 color filter array pattern
    # Bayer has 2x2 pattern
    pattern = raw.raw_pattern
    return pattern.shape == (6, 6)


def get_raw_info(filepath: str | Path) -> dict:
    """Get metadata from RAW file."""
    filepath = Path(filepath)

    try:
        with rawpy.imread(str(filepath)) as raw:
            # Try to get camera info (attributes vary by rawpy version)
            try:
                camera = f"{raw.raw_image.camera_make} {raw.raw_image.camera_model}".strip()
            except:
                try:
                    camera = f"{raw.color_desc.decode()} camera"
                except:
                    camera = "Unknown camera"

            info = {
                'camera': camera,
                'iso': getattr(raw, 'camera_iso_speed', 0),
                'shutter': getattr(raw, 'camera_shutter_speed', 0),
                'aperture': getattr(raw, 'camera_aperture', 0),
                'width': raw.sizes.width,
                'height': raw.sizes.height,
                'sensor_type': 'X-Trans' if _is_xtrans(raw) else 'Bayer'
            }
            return info

    except Exception as e:
        raise RawLoadError(f"Failed to read RAW info: {e}")


if __name__ == "__main__":
    # Simple test
    import sys

    if len(sys.argv) < 2:
        print("Usage: python raw_loader.py <raw_file>")
        sys.exit(1)

    filepath = sys.argv[1]

    try:
        print(f"Loading: {filepath}")
        info = get_raw_info(filepath)
        print(f"Camera: {info['camera']}")
        print(f"Sensor: {info['sensor_type']}")
        print(f"Size: {info['width']}×{info['height']}")

        print("\nProcessing RAW...")
        img = load_raw_linear(filepath)
        print(f"Shape: {img.shape}")
        print(f"Dtype: {img.dtype}")
        print(f"Range: [{img.min():.4f}, {img.max():.4f}]")
        print("✓ Success!")

    except RawLoadError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
