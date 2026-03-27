from __future__ import annotations

import pandas as pd
import pytest

from src.utils.error_codes import PrdError
from src.validation.filters import ValidationConfig, apply_validation_filters


def _base_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "area_px": 500.0,
                "aspect_ratio": 1.0,
                "circularity": 0.90,
                "area_um2": 1.0,
                "perimeter_px": 10.0,
            },
            {
                "id": 2,
                "area_px": 20.0,
                "aspect_ratio": 1.0,
                "circularity": 0.85,
                "area_um2": 1.0,
                "perimeter_px": 10.0,
            },
            {
                "id": 3,
                "area_px": 900.0,
                "aspect_ratio": 8.0,
                "circularity": 0.80,
                "area_um2": 1.0,
                "perimeter_px": 10.0,
            },
            {
                "id": 4,
                "area_px": 900.0,
                "aspect_ratio": 1.5,
                "circularity": 0.80,
                "is_edge_particle": True,
                "area_um2": 1.0,
                "perimeter_px": 10.0,
            },
        ]
    )


def test_apply_validation_filters_enforces_area_aspect_and_edge_criteria() -> None:
    metrics = _base_metrics()

    result = apply_validation_filters(metrics)
    annotated = result["annotated_particles"]
    valid = result["valid_particles"]

    assert list(valid["id"]) == [1]

    row_by_id = {int(row.id): row for row in annotated.itertuples(index=False)}
    assert row_by_id[1].validation_status == "valid"
    assert pd.isna(row_by_id[1].rejection_reason)

    assert row_by_id[2].validation_status == "rejected"
    assert row_by_id[2].rejection_reason == "area_range"

    assert row_by_id[3].validation_status == "rejected"
    assert row_by_id[3].rejection_reason == "aspect_ratio_range"

    assert row_by_id[4].validation_status == "rejected"
    assert row_by_id[4].rejection_reason == "edge_particle"

    assert result["rejection_stats"] == {
        "area_range": 1,
        "aspect_ratio_range": 1,
        "edge_particle": 1,
        "circularity_min": 0,
    }


def test_apply_validation_filters_keeps_contract_fields_for_downstream_consumers() -> None:
    metrics = _base_metrics().iloc[[0]].copy()

    result = apply_validation_filters(metrics)
    annotated = result["annotated_particles"]

    assert set(metrics.columns).issubset(set(annotated.columns))
    assert {"validation_status", "rejection_reason", "is_edge_particle"}.issubset(set(annotated.columns))
    assert {"circularity", "validation_status", "rejection_reason"}.issubset(set(result["valid_particles"].columns))


def test_apply_validation_filters_null_circularity_min_does_not_reject_low_circularity() -> None:
    metrics = pd.DataFrame(
        [
            {
                "id": 10,
                "area_px": 300.0,
                "aspect_ratio": 1.2,
                "circularity": 0.05,
            }
        ]
    )

    result = apply_validation_filters(metrics, config=ValidationConfig(circularity_min=None))

    assert list(result["valid_particles"]["id"]) == [10]
    assert result["rejection_stats"]["circularity_min"] == 0


def test_apply_validation_filters_applies_circularity_threshold_when_configured() -> None:
    metrics = pd.DataFrame(
        [
            {
                "id": 11,
                "area_px": 300.0,
                "aspect_ratio": 1.2,
                "circularity": 0.40,
            }
        ]
    )

    with pytest.raises(PrdError) as exc_info:
        apply_validation_filters(metrics, config=ValidationConfig(circularity_min=0.5))

    err = exc_info.value
    assert err.code == "ERR_MASK_002"
    assert err.message == "No valid particles detected after filtering. Check thresholds or image quality."


def test_apply_validation_filters_raises_err_mask_002_when_no_valid_particles_remain() -> None:
    metrics = pd.DataFrame(
        [
            {
                "id": 21,
                "area_px": 25.0,
                "aspect_ratio": 10.0,
                "circularity": 0.6,
            }
        ]
    )

    with pytest.raises(PrdError) as exc_info:
        apply_validation_filters(metrics)

    err = exc_info.value
    assert err.code == "ERR_MASK_002"
    assert err.message == "No valid particles detected after filtering. Check thresholds or image quality."
    assert err.stage == "validation"
    assert err.context is not None
    assert err.context["valid_particles"] == 0
