from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile

from src.preprocessing.pipeline import apply_clahe_to_uint8, load_tiff_uint16
from src.utils.error_codes import ERROR_CATALOG, PrdError


def test_load_tiff_uint16_preserves_16bit_fidelity(tmp_path: Path) -> None:
    expected = np.arange(0, 64 * 64, dtype=np.uint16).reshape(64, 64)
    tiff_path = tmp_path / "sample_16bit.tif"
    tifffile.imwrite(tiff_path, expected)

    loaded = load_tiff_uint16(tiff_path)

    assert loaded.dtype == np.uint16
    assert loaded.shape == expected.shape
    assert np.array_equal(loaded, expected)


def test_apply_clahe_to_uint8_returns_model_ready_uint8_array() -> None:
    base = np.tile(np.linspace(12000, 12200, 64, dtype=np.uint16), (64, 1))
    image_16bit = base.copy()
    image_16bit[20:44, 20:44] = image_16bit[20:44, 20:44] + 60

    result = apply_clahe_to_uint8(image_16bit, clip_limit=2.0, tile_grid_size=(8, 8))
    direct_normalized = np.empty_like(image_16bit, dtype=np.uint8)
    cv2.normalize(image_16bit, direct_normalized, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    expected = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(direct_normalized)

    assert result.dtype == np.uint8
    assert result.shape == image_16bit.shape
    assert int(result.min()) >= 0
    assert int(result.max()) <= 255
    assert not np.array_equal(result, direct_normalized)
    assert np.array_equal(result, expected)


def test_load_tiff_uint16_raises_err_io_003_for_corrupt_file(tmp_path: Path) -> None:
    broken_tiff = tmp_path / "broken.tif"
    broken_tiff.write_bytes(b"not-a-valid-tiff")

    with pytest.raises(PrdError) as exc_info:
        load_tiff_uint16(broken_tiff)

    error = exc_info.value
    assert error.code == "ERR_IO_003"
    assert error.message == ERROR_CATALOG["ERR_IO_003"]
    assert error.to_dict()["code"] == "ERR_IO_003"
    assert error.to_dict()["message"] == ERROR_CATALOG["ERR_IO_003"]
