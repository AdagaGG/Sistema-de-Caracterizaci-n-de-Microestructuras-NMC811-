from __future__ import annotations

import pandas as pd

from src.ui.components.results_panel import _csv_summary_for_heatmap


def test_csv_summary_for_heatmap_filters_by_image_id_and_coerces_types(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {"image_id": "img_A", "circularity": "0.80", "area_um2": "10.5"},
            {"image_id": "img_A", "circularity": "0.60", "area_um2": "9.5"},
            {"image_id": "img_B", "circularity": "0.20", "area_um2": "1.0"},
        ]
    )

    particle_count, mean_circularity, mean_area_um2 = _csv_summary_for_heatmap(
        frame=frame,
        heatmap_path=str(tmp_path / "img_A__heatmap_circularity_overlay.png"),
    )

    assert particle_count == 2
    assert mean_circularity == 0.7
    assert mean_area_um2 == 10.0


def test_csv_summary_for_heatmap_handles_column_name_mismatch(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {"Image ID": "img_X", "Circularity": "0.50", "Area_um2": "8.0"},
            {"Image ID": "img_X", "Circularity": "0.70", "Area_um2": "12.0"},
        ]
    )

    particle_count, mean_circularity, mean_area_um2 = _csv_summary_for_heatmap(
        frame=frame,
        heatmap_path=str(tmp_path / "img_X__heatmap_circularity_overlay.png"),
    )

    assert particle_count == 2
    assert mean_circularity == 0.6
    assert mean_area_um2 == 10.0


def test_csv_summary_for_heatmap_returns_none_means_when_no_numeric_values(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {"image_id": "img_N", "circularity": "no-data", "area_um2": "n/a"},
            {"image_id": "img_N", "circularity": "-", "area_um2": ""},
        ]
    )

    particle_count, mean_circularity, mean_area_um2 = _csv_summary_for_heatmap(
        frame=frame,
        heatmap_path=str(tmp_path / "img_N__heatmap_circularity_overlay.png"),
    )

    assert particle_count == 2
    assert mean_circularity is None
    assert mean_area_um2 is None
