from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.export.csv_export import (
    CSV_EXPORT_COLUMNS,
    consolidate_filtered_metrics_batch,
    write_analisis_gold_standard_csv,
)


def test_consolidate_filtered_metrics_batch_enforces_deterministic_schema_and_order() -> None:
    frame_b = pd.DataFrame(
        [
            {
                "id": 2,
                "area_px": 160.0,
                "area_um2": 1.2,
                "perimeter_px": 48.0,
                "circularity": 0.87,
                "circularity_effective": 0.80,
                "equivalent_diameter_px": 14.1,
                "equivalent_diameter_um": 1.18,
                "aspect_ratio": 0.9,
                "mean_intensity": 175.0,
                "std_intensity": 9.0,
                "dark_area_fraction": 0.12,
                "crack_severity": 0.0,
                "area": 160.0,
                "perimeter": 48.0,
                "metric_error_code": None,
                "metric_error_message": None,
                "validation_status": "valid",
                "rejection_reason": None,
                "is_edge_particle": False,
            }
        ]
    )
    frame_a = pd.DataFrame(
        [
            {
                "id": 1,
                "area_px": 120.0,
                "area_um2": 0.9,
                "perimeter_px": 44.0,
                "circularity": 0.78,
                "circularity_effective": 0.58,
                "equivalent_diameter_px": 12.3,
                "equivalent_diameter_um": 1.04,
                "aspect_ratio": 1.1,
                "mean_intensity": 150.0,
                "std_intensity": 20.0,
                "dark_area_fraction": 0.30,
                "crack_severity": 0.41,
                "area": 120.0,
                "perimeter": 44.0,
                "metric_error_code": None,
                "metric_error_message": None,
                "validation_status": "rejected",
                "rejection_reason": "edge_particle",
                "is_edge_particle": True,
            }
        ]
    )

    merged = consolidate_filtered_metrics_batch(
        [
            ("Image_B2", frame_b),
            ("Image_A1", frame_a),
        ]
    )

    assert list(merged.columns) == CSV_EXPORT_COLUMNS
    assert list(merged["image_id"]) == ["Image_A1", "Image_B2"]
    assert list(merged["particle_id"]) == [1, 2]


def test_write_analisis_gold_standard_csv_writes_excel_friendly_utf8_bom_and_roundtrip(tmp_path: Path) -> None:
    merged = pd.DataFrame(
        [
            {
                "image_id": "Image_A1",
                "particle_id": 1,
                "area_px": 100.0,
                "area_um2": 0.711,
                "perimeter_px": 40.0,
                "circularity": 0.785,
                "circularity_effective": 0.75,
                "equivalent_diameter_px": 11.28,
                "equivalent_diameter_um": 0.951,
                "aspect_ratio": 1.0,
                "mean_intensity": 170.0,
                "std_intensity": 8.0,
                "dark_area_fraction": 0.10,
                "crack_severity": 0.0,
                "area": 100.0,
                "perimeter": 40.0,
                "metric_error_code": None,
                "metric_error_message": None,
                "is_edge_particle": False,
                "validation_status": "valid",
                "rejection_reason": None,
            }
        ],
        columns=CSV_EXPORT_COLUMNS,
    )

    output_path = write_analisis_gold_standard_csv(merged, output_root=tmp_path)

    assert output_path.name == "analisis_gold_standard.csv"
    raw = output_path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")

    loaded = pd.read_csv(output_path, encoding="utf-8-sig")
    assert list(loaded.columns) == CSV_EXPORT_COLUMNS
    assert loaded.shape == (1, len(CSV_EXPORT_COLUMNS))
    assert loaded.loc[0, "image_id"] == "Image_A1"
    assert loaded.loc[0, "particle_id"] == 1


def test_consolidate_filtered_metrics_batch_preserves_validation_flags_for_all_rows() -> None:
    frame_valid = pd.DataFrame(
        [
            {
                "id": 4,
                "area_px": 220.0,
                "area_um2": 1.5,
                "perimeter_px": 61.0,
                "circularity": 0.74,
                "circularity_effective": 0.70,
                "equivalent_diameter_px": 16.7,
                "equivalent_diameter_um": 1.41,
                "aspect_ratio": 1.0,
                "mean_intensity": 180.0,
                "std_intensity": 7.0,
                "dark_area_fraction": 0.08,
                "crack_severity": 0.0,
                "area": 220.0,
                "perimeter": 61.0,
                "metric_error_code": None,
                "metric_error_message": None,
                "is_edge_particle": False,
                "validation_status": "valid",
                "rejection_reason": None,
            }
        ]
    )
    frame_rejected = pd.DataFrame(
        [
            {
                "id": 5,
                "area_px": 30.0,
                "area_um2": 0.2,
                "perimeter_px": 20.0,
                "circularity": 0.3,
                "circularity_effective": 0.15,
                "equivalent_diameter_px": 6.2,
                "equivalent_diameter_um": 0.5,
                "aspect_ratio": 5.5,
                "mean_intensity": 120.0,
                "std_intensity": 25.0,
                "dark_area_fraction": 0.50,
                "crack_severity": 1.0,
                "area": 30.0,
                "perimeter": 20.0,
                "metric_error_code": "ERR_METRIC_004",
                "metric_error_message": "Circularity calculation failed: perimeter is zero. Check contour extraction.",
                "is_edge_particle": True,
                "validation_status": "rejected",
                "rejection_reason": "aspect_ratio_range",
            }
        ]
    )

    merged = consolidate_filtered_metrics_batch(
        [("Image_A1", frame_valid), ("Image_A1", frame_rejected)]
    )

    assert merged.shape[0] == 2
    assert set(merged["validation_status"]) == {"valid", "rejected"}
    assert merged.loc[merged["particle_id"] == 5, "rejection_reason"].item() == "aspect_ratio_range"

