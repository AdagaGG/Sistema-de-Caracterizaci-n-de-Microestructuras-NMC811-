from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tifffile

from src.utils.error_codes import raise_prd_error


def load_tiff_uint16(path: str | Path) -> np.ndarray:
    """Load a TIFF image preserving 16-bit fidelity.

    Raises PrdError(ERR_IO_003) when TIFF is unreadable or corrupt.
    """
    file_path = Path(path)

    try:
        image = tifffile.imread(file_path)
    except Exception:
        raise_prd_error("ERR_IO_003", stage="ingestion", context={"path": str(file_path)})

    if image.dtype != np.uint16:
        image = image.astype(np.uint16, copy=False)

    return image


def apply_clahe_to_uint8(
    image_uint16: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to a 16-bit image and return model-ready uint8 array."""
    if image_uint16.dtype != np.uint16:
        raise ValueError("image_uint16 must have dtype uint16")

    normalized_uint8 = np.empty_like(image_uint16, dtype=np.uint8)
    cv2.normalize(
        image_uint16,
        normalized_uint8,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(normalized_uint8)


def preprocess_tiff_for_model(
    path: str | Path,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Load 16-bit TIFF and return CLAHE-processed uint8 image."""
    image_uint16 = load_tiff_uint16(path)
    return apply_clahe_to_uint8(image_uint16, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
