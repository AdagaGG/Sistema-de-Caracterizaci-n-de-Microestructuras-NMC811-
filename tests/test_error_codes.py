from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.utils import error_codes
from src.utils.error_codes import PrdError


EXPECTED_MESSAGES = {
    "ERR_VRAM_001": "GPU memory exceeded during segmentation. Reduce image resolution or tile size.",
    "ERR_MASK_002": "No valid particles detected after filtering. Check thresholds or image quality.",
    "ERR_IO_003": "Failed to read 16-bit TIFF file. Verify file integrity and tifffile library version.",
    "ERR_METRIC_004": "Circularity calculation failed: perimeter is zero. Check contour extraction.",
}


def _errors_from_prd() -> dict[str, str]:
    prd_path = Path(__file__).resolve().parents[1] / "docs" / "PRD.yaml"
    data = yaml.safe_load(prd_path.read_text(encoding="utf-8"))
    return {item["code"]: item["message"] for item in data["errors"]}


def test_catalog_messages_match_expected_byte_for_byte() -> None:
    assert error_codes.ERROR_CATALOG == EXPECTED_MESSAGES


def test_catalog_messages_match_prd_source_of_truth() -> None:
    assert error_codes.ERROR_CATALOG == _errors_from_prd()


def test_get_error_returns_deterministic_payload() -> None:
    payload = error_codes.get_error("ERR_IO_003")
    assert payload == {
        "code": "ERR_IO_003",
        "message": EXPECTED_MESSAGES["ERR_IO_003"],
    }


def test_build_error_payload_includes_stage_and_context() -> None:
    payload = error_codes.build_error_payload(
        "ERR_MASK_002",
        stage="validation",
        context={"image_id": "img_RDBS_0200", "remaining": 0},
    )

    assert payload == {
        "code": "ERR_MASK_002",
        "message": EXPECTED_MESSAGES["ERR_MASK_002"],
        "stage": "validation",
        "context": {"image_id": "img_RDBS_0200", "remaining": 0},
    }


def test_raise_prd_error_raises_typed_exception_with_payload() -> None:
    with pytest.raises(PrdError) as exc_info:
        error_codes.raise_prd_error("ERR_METRIC_004", stage="metrics", context={"particle_id": 7})

    err = exc_info.value
    assert err.code == "ERR_METRIC_004"
    assert err.message == EXPECTED_MESSAGES["ERR_METRIC_004"]
    assert err.stage == "metrics"
    assert err.context == {"particle_id": 7}
    assert err.to_dict() == {
        "code": "ERR_METRIC_004",
        "message": EXPECTED_MESSAGES["ERR_METRIC_004"],
        "stage": "metrics",
        "context": {"particle_id": 7},
    }


def test_serialize_error_is_stable_for_prd_and_generic_exceptions() -> None:
    prd_error = PrdError("ERR_VRAM_001", EXPECTED_MESSAGES["ERR_VRAM_001"], stage="segmentation")
    serialized_prd = error_codes.serialize_error(prd_error)
    assert serialized_prd == {
        "code": "ERR_VRAM_001",
        "message": EXPECTED_MESSAGES["ERR_VRAM_001"],
        "stage": "segmentation",
        "context": {},
    }

    generic = RuntimeError("unexpected")
    serialized_generic = error_codes.serialize_error(generic)
    assert serialized_generic == {
        "code": "ERR_UNKNOWN",
        "message": "unexpected",
        "stage": None,
        "context": {},
    }


def test_log_prd_error_emits_consistent_message() -> None:
    class _Logger:
        def __init__(self) -> None:
            self.events: list[tuple[str, str, dict]] = []

        def error(self, message: str, extra: dict) -> None:
            self.events.append(("error", message, extra))

    logger = _Logger()
    payload = error_codes.log_prd_error(
        logger,
        "ERR_VRAM_001",
        stage="segmentation",
        context={"tile_size": 1024},
    )

    assert payload == {
        "code": "ERR_VRAM_001",
        "message": EXPECTED_MESSAGES["ERR_VRAM_001"],
        "stage": "segmentation",
        "context": {"tile_size": 1024},
    }
    assert logger.events == [
        (
            "error",
            "ERR_VRAM_001: GPU memory exceeded during segmentation. Reduce image resolution or tile size.",
            {"error": payload},
        )
    ]


def test_unknown_code_raises_key_error() -> None:
    with pytest.raises(KeyError):
        error_codes.get_error("ERR_NOT_DEFINED")
