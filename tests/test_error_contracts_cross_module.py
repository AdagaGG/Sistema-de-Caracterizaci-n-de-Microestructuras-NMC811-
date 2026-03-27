from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.metrics.geometry_extractor import extract_metrics
from src.preprocessing.pipeline import load_tiff_uint16
from src.segmentation.mobilesam_inference import MobileSamInferenceCore
from src.utils.error_codes import PrdError
from src.validation.filters import apply_validation_filters


def _prd_errors() -> dict[str, str]:
    prd_path = Path(__file__).resolve().parents[1] / "docs" / "PRD.yaml"
    payload = yaml.safe_load(prd_path.read_text(encoding="utf-8"))
    return {item["code"]: item["message"] for item in payload["errors"]}


class _FailingGenerator:
    def generate(self, image: np.ndarray) -> list[dict[str, object]]:
        raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")


def test_err_io_003_emits_exact_prd_message_from_preprocessing_path(tmp_path: Path) -> None:
    expected = _prd_errors()
    invalid_tiff = tmp_path / "invalid_fixture.tif"
    invalid_tiff.write_bytes(b"invalid-tiff-bytes")

    with pytest.raises(PrdError) as exc_info:
        load_tiff_uint16(invalid_tiff)

    err = exc_info.value
    assert err.code == "ERR_IO_003"
    assert err.message == expected["ERR_IO_003"]
    assert err.stage == "ingestion"
    assert err.context == {"path": str(invalid_tiff)}


def test_err_vram_001_emits_exact_prd_message_from_segmentation_oom_path() -> None:
    expected = _prd_errors()
    core = MobileSamInferenceCore(model=object(), mask_generator=_FailingGenerator())

    with pytest.raises(PrdError) as exc_info:
        core.segment_image(np.zeros((4, 4), dtype=np.uint8))

    err = exc_info.value
    assert err.code == "ERR_VRAM_001"
    assert err.message == expected["ERR_VRAM_001"]
    assert err.stage == "segmentation"
    assert err.context == {"reason": "oom"}


def test_err_mask_002_emits_exact_prd_message_from_zero_valid_filter_path() -> None:
    expected = _prd_errors()
    metrics = pd.DataFrame(
        [
            {
                "id": 1,
                "area_px": 10.0,
                "aspect_ratio": 8.0,
                "circularity": 0.7,
            }
        ]
    )

    with pytest.raises(PrdError) as exc_info:
        apply_validation_filters(metrics)

    err = exc_info.value
    assert err.code == "ERR_MASK_002"
    assert err.message == expected["ERR_MASK_002"]
    assert err.stage == "validation"
    assert err.context == {
        "valid_particles": 0,
        "rejection_stats": {
            "area_range": 1,
            "aspect_ratio_range": 0,
            "edge_particle": 0,
            "circularity_min": 0,
        },
    }


def test_err_metric_004_emits_exact_prd_message_from_zero_perimeter_metrics_path() -> None:
    expected = _prd_errors()
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[3, 3] = 9

    metrics = extract_metrics(labels)
    row = metrics.iloc[0]

    assert int(row["id"]) == 9
    assert row["metric_error_code"] == "ERR_METRIC_004"
    assert row["metric_error_message"] == expected["ERR_METRIC_004"]
