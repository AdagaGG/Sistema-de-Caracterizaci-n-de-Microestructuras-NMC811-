from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.visualization.heatmaps import (
    build_heatmap_output_paths,
    circularity_to_bgr,
    generate_circularity_heatmap_artifacts,
    write_heatmap_batch_manifest,
)


def _rect_contour(x1: int, y1: int, x2: int, y2: int) -> list[list[int]]:
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def test_circularity_to_bgr_maps_contract_anchors_and_direction() -> None:
    assert circularity_to_bgr(0.0) == (39, 48, 215)
    assert circularity_to_bgr(0.5) == (139, 224, 254)
    assert circularity_to_bgr(1.0) == (80, 152, 26)

    low = circularity_to_bgr(0.1)
    high = circularity_to_bgr(0.9)
    assert low[2] > low[1]
    assert high[1] > high[2]

    assert circularity_to_bgr(float("nan")) == (39, 48, 215)


def test_heatmap_output_naming_contract_is_deterministic(tmp_path) -> None:
    paths = build_heatmap_output_paths(image_stem="Image_A1", output_root=tmp_path)

    assert paths["overlay_path"].name == "Image_A1__heatmap_circularity_overlay.png"
    assert paths["mask_path"].name == "Image_A1__heatmap_circularity_mask.png"
    assert paths["legend_path"].name == "Image_A1__heatmap_circularity_legend.png"
    assert paths["meta_path"].name == "Image_A1__heatmap_circularity_meta.json"
    assert paths["overlay_path"].parent.name == "visualizations"
    assert paths["overlay_path"].parent == paths["mask_path"].parent == paths["legend_path"].parent == paths["meta_path"].parent


def test_generate_circularity_heatmap_artifacts_includes_rejected_semantics(tmp_path) -> None:
    image = np.zeros((80, 80), dtype=np.uint8)
    particles = pd.DataFrame(
        [
            {
                "particle_id": 1,
                "circularity": 0.9,
                "validation_status": "valid",
                "rejection_reason": None,
                "contour_points": _rect_contour(5, 5, 25, 25),
                "centroid_xy": [15.0, 15.0],
            },
            {
                "particle_id": 2,
                "circularity": 0.1,
                "validation_status": "valid",
                "rejection_reason": None,
                "contour_points": _rect_contour(30, 5, 50, 25),
                "centroid_xy": [40.0, 15.0],
            },
            {
                "particle_id": 3,
                "circularity": 0.8,
                "validation_status": "rejected",
                "rejection_reason": "edge_particle",
                "contour_points": _rect_contour(5, 30, 25, 50),
                "centroid_xy": [15.0, 40.0],
            },
        ]
    )

    artifacts = generate_circularity_heatmap_artifacts(
        image=image,
        annotated_particles=particles,
        image_stem="Image_A1",
        output_root=tmp_path,
    )

    assert artifacts["valid_particles"] == 2
    assert artifacts["rejected_particles"] == 1
    assert artifacts["clipped_count"] == 0

    for key in ("overlay_path", "mask_path", "legend_path", "meta_path"):
        assert artifacts[key].exists()

    with artifacts["meta_path"].open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    assert meta["rejected_particles"] == 1
    assert meta["rejected_style"]["fill_bgr"] == [128, 128, 128]


def test_generate_circularity_heatmap_artifacts_skips_missing_contours_without_crashing(tmp_path) -> None:
    image = np.zeros((48, 48), dtype=np.uint8)
    particles = pd.DataFrame(
        [
            {
                "particle_id": 1,
                "circularity": 0.66,
                "validation_status": "valid",
                "rejection_reason": None,
            }
        ]
    )

    artifacts = generate_circularity_heatmap_artifacts(
        image=image,
        annotated_particles=particles,
        image_stem="Image_NoContours",
        output_root=tmp_path,
    )

    assert artifacts["valid_particles"] == 0
    assert artifacts["rejected_particles"] == 0
    assert artifacts["clipped_count"] == 0
    assert artifacts["overlay_path"].exists()
    assert artifacts["mask_path"].exists()
    assert artifacts["legend_path"].exists()
    assert artifacts["meta_path"].exists()


def test_write_heatmap_batch_manifest_coerces_optional_counts(tmp_path) -> None:
    visualizations = tmp_path / "visualizations"
    visualizations.mkdir(parents=True, exist_ok=True)
    overlay_path = visualizations / "overlay.png"
    mask_path = visualizations / "mask.png"
    legend_path = visualizations / "legend.png"
    meta_path = visualizations / "meta.json"
    for path in (overlay_path, mask_path, legend_path, meta_path):
        path.write_text("x", encoding="utf-8")

    manifest_path = write_heatmap_batch_manifest(
        entries=[
            {
                "image_stem": "Image_A1",
                "overlay_path": overlay_path,
                "mask_path": mask_path,
                "legend_path": legend_path,
                "meta_path": meta_path,
                "valid_particles": None,
                "rejected_particles": None,
                "clipped_count": None,
            }
        ],
        output_root=tmp_path,
    )

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    assert manifest["files"][0]["valid_particles"] == 0
    assert manifest["files"][0]["rejected_particles"] == 0
    assert manifest["files"][0]["clipped_count"] == 0
