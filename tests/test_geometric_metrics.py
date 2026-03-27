from __future__ import annotations

import numpy as np
import pytest

from src.metrics.geometry_extractor import PIXEL_SCALE_UM, extract_metrics


def test_extract_metrics_returns_expected_schema_and_compatibility_columns() -> None:
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[2:8, 2:8] = 1

    metrics = extract_metrics(labels)

    expected_columns = {
        "id",
        "area_px",
        "area_um2",
        "perimeter_px",
        "perimeter_um",
        "circularity",
        "equivalent_diameter_px",
        "equivalent_diameter_um",
        "aspect_ratio",
        "bbox",
        "contour_points",
        "centroid_xy",
        "area",
        "perimeter",
    }

    assert set(metrics.columns) >= expected_columns
    assert len(metrics) == 1
    row = metrics.iloc[0]
    assert int(row["id"]) == 1
    assert float(row["area"]) == float(row["area_px"])
    assert float(row["perimeter"]) == float(row["perimeter_px"])
    assert isinstance(row["contour_points"], list)
    assert len(row["contour_points"]) >= 3


def test_extract_metrics_computes_area_and_unit_conversions() -> None:
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[10:20, 5:15] = 7

    metrics = extract_metrics(labels)
    row = metrics.iloc[0]

    assert int(row["id"]) == 7
    assert float(row["area_px"]) == 100.0
    assert float(row["area_um2"]) == pytest.approx(100.0 * (PIXEL_SCALE_UM**2), rel=1e-9)
    assert float(row["equivalent_diameter_px"]) == pytest.approx(np.sqrt(4.0 * 100.0 / np.pi), rel=1e-6)
    assert float(row["equivalent_diameter_um"]) == pytest.approx(float(row["equivalent_diameter_px"]) * PIXEL_SCALE_UM, rel=1e-9)


def test_extract_metrics_circular_shape_has_higher_circularity_than_elongated_shape() -> None:
    labels = np.zeros((120, 120), dtype=np.int32)

    yy, xx = np.ogrid[:120, :120]
    circle = (xx - 35) ** 2 + (yy - 35) ** 2 <= 18**2
    labels[circle] = 1
    labels[70:78, 20:95] = 2

    metrics = extract_metrics(labels)

    circle_row = metrics.loc[metrics["id"] == 1].iloc[0]
    elongated_row = metrics.loc[metrics["id"] == 2].iloc[0]

    assert float(circle_row["circularity"]) > float(elongated_row["circularity"])
    assert float(circle_row["aspect_ratio"]) < float(elongated_row["aspect_ratio"])


def test_extract_metrics_handles_zero_perimeter_with_traceable_error_payload() -> None:
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[10, 10] = 3

    metrics = extract_metrics(labels)
    row = metrics.iloc[0]

    assert int(row["id"]) == 3
    assert float(row["perimeter_px"]) == 0.0
    assert float(row["circularity"]) == 0.0
    assert row["metric_error_code"] == "ERR_METRIC_004"
    assert row["metric_error_message"] == "Circularity calculation failed: perimeter is zero. Check contour extraction."


def test_extract_metrics_accepts_int32_labeled_masks_with_background_zero() -> None:
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[10:14, 11:15] = 2

    metrics = extract_metrics(labels)

    assert list(metrics["id"]) == [1, 2]
    assert all(metrics["area_px"] > 0)


def test_extract_metrics_internal_crack_penalizes_effective_circularity() -> None:
    labels = np.zeros((64, 64), dtype=np.int32)
    yy, xx = np.ogrid[:64, :64]
    particle = (xx - 32) ** 2 + (yy - 32) ** 2 <= 14**2
    labels[particle] = 1

    intensity = np.full((64, 64), 180, dtype=np.uint8)
    intensity[20:44, 30:34] = 10

    metrics = extract_metrics(labels, intensity_image=intensity)
    row = metrics.iloc[0]

    assert float(row["circularity"]) > 0.8
    assert float(row["crack_severity"]) > 0.0
    assert float(row["circularity_effective"]) < float(row["circularity"])
