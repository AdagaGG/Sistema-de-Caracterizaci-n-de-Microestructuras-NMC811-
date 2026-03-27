from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.error_codes import ERROR_CATALOG, PrdError


class _StubSegmentationCore:
    def __init__(self, behavior_by_image: dict[str, str]) -> None:
        self.behavior_by_image = behavior_by_image
        self.current_image_id: str | None = None

    def segment_image(self, image: np.ndarray) -> dict[str, object]:
        if self.current_image_id is None:
            raise RuntimeError("current_image_id must be set before segment_image")

        behavior = self.behavior_by_image.get(self.current_image_id, "ok")
        if behavior == "segmentation_fail":
            raise PrdError(
                code="ERR_VRAM_001",
                message=ERROR_CATALOG["ERR_VRAM_001"],
                stage="segmentation",
                context={"reason": "stress"},
            )

        mask = np.zeros_like(image, dtype=bool)
        mask[2:5, 2:5] = True
        return {
            "masks": [{"segmentation": mask, "area": int(mask.sum()), "bbox": (2, 2, 3, 3)}],
            "telemetry": {"cleanup_invoked": True},
        }


def _build_annotated_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "particle_id": 1,
                "area_px": 120.0,
                "area_um2": 0.8,
                "perimeter_px": 44.0,
                "circularity": 0.78,
                "equivalent_diameter_px": 12.3,
                "equivalent_diameter_um": 1.0,
                "aspect_ratio": 1.0,
                "area": 120.0,
                "perimeter": 44.0,
                "metric_error_code": None,
                "metric_error_message": None,
                "is_edge_particle": False,
                "validation_status": "valid",
                "rejection_reason": None,
                "contour_points": [[2, 2], [4, 2], [4, 4], [2, 4]],
            }
        ]
    )


def test_run_batch_chains_all_stages_and_returns_deterministic_transitions(tmp_path: Path) -> None:
    from src.pipeline.orchestrator_engine import run_batch

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True)
    for name in ("Image_A1.tif", "Image_B2.tif"):
        (input_dir / name).write_bytes(b"placeholder")

    segmentation_core = _StubSegmentationCore(behavior_by_image={})
    call_trace: list[tuple[str, str]] = []

    def preprocess(path: str | Path, **_: object) -> np.ndarray:
        image_id = Path(path).stem
        segmentation_core.current_image_id = image_id
        call_trace.append(("preprocess", image_id))
        return np.zeros((8, 8), dtype=np.uint8)

    def refine(masks: list[dict[str, object]], **_: object) -> np.ndarray:
        call_trace.append(("refine", segmentation_core.current_image_id or ""))
        labels = np.zeros((8, 8), dtype=np.int32)
        labels[2:5, 2:5] = 1
        return labels

    def extract(mask_labels: np.ndarray, intensity_image: np.ndarray | None = None) -> pd.DataFrame:
        call_trace.append(("extract", segmentation_core.current_image_id or ""))
        assert mask_labels.dtype == np.int32
        assert intensity_image is not None
        return _build_annotated_frame().drop(columns=["validation_status", "rejection_reason", "is_edge_particle"])

    def validate(metrics: pd.DataFrame, config: object | None = None, image_shape: tuple[int, int] | None = None) -> dict[str, object]:
        call_trace.append(("validate", segmentation_core.current_image_id or ""))
        assert config is not None
        assert image_shape == (8, 8)
        annotated = _build_annotated_frame()
        return {
            "annotated_particles": annotated,
            "valid_particles": annotated.copy(deep=True),
            "rejection_stats": {"area_range": 0, "aspect_ratio_range": 0, "edge_particle": 0, "circularity_min": 0},
            "config": {},
        }

    summary = run_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        config={
            "dependencies": {
                "preprocess_tiff_for_model": preprocess,
                "segmentation_core": segmentation_core,
                "refine_masks_to_labels": refine,
                "extract_metrics": extract,
                "apply_validation_filters": validate,
            }
        },
    )

    assert summary["images_total"] == 2
    assert summary["images_succeeded"] == 2
    assert summary["images_failed"] == 0
    assert summary["status"] == "completed"
    assert Path(summary["exports"]["csv_path"]).exists()
    assert Path(summary["exports"]["heatmap_manifest_path"]).exists()

    for image_summary in summary["images"]:
        assert image_summary["status"] == "success"
        assert image_summary["state_transitions"] == [
            {"from": "raw", "to": "preprocessed"},
            {"from": "preprocessed", "to": "segmented"},
            {"from": "segmented", "to": "refined"},
            {"from": "refined", "to": "analyzed"},
            {"from": "analyzed", "to": "exported"},
        ]
        assert image_summary["timings"]["total_seconds"] >= 0.0
        assert set(image_summary["timings"].keys()) >= {
            "preprocessed",
            "segmented",
            "refined",
            "analyzed",
            "exported",
            "total_seconds",
        }

    assert [step[0] for step in call_trace].count("preprocess") == 2
    assert [step[0] for step in call_trace].count("validate") == 2


def test_run_batch_continues_after_per_image_failure_and_preserves_structured_error(tmp_path: Path) -> None:
    from src.pipeline.orchestrator_engine import run_batch

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True)
    for name in ("Image_A1.tif", "Image_B2.tif"):
        (input_dir / name).write_bytes(b"placeholder")

    segmentation_core = _StubSegmentationCore(behavior_by_image={})
    consolidated_counts: list[int] = []

    def preprocess(path: str | Path, **_: object) -> np.ndarray:
        segmentation_core.current_image_id = Path(path).stem
        return np.zeros((8, 8), dtype=np.uint8)

    def refine(masks: list[dict[str, object]], **_: object) -> np.ndarray:
        labels = np.zeros((8, 8), dtype=np.int32)
        labels[2:5, 2:5] = 1
        return labels

    def extract(mask_labels: np.ndarray, intensity_image: np.ndarray | None = None) -> pd.DataFrame:
        assert intensity_image is not None
        return _build_annotated_frame().drop(columns=["validation_status", "rejection_reason", "is_edge_particle"])

    def validate(metrics: pd.DataFrame, config: object | None = None, image_shape: tuple[int, int] | None = None) -> dict[str, object]:
        if segmentation_core.current_image_id == "Image_A1":
            raise PrdError(
                code="ERR_MASK_002",
                message=ERROR_CATALOG["ERR_MASK_002"],
                stage="validation",
                context={"valid_particles": 0, "rejection_stats": {"area_range": 1}},
            )
        annotated = _build_annotated_frame()
        return {
            "annotated_particles": annotated,
            "valid_particles": annotated.copy(deep=True),
            "rejection_stats": {"area_range": 0, "aspect_ratio_range": 0, "edge_particle": 0, "circularity_min": 0},
            "config": {},
        }

    def consolidate(per_image_metrics: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
        consolidated_counts.append(len(per_image_metrics))
        return pd.DataFrame(
            [
                {
                    "image_id": image_id,
                    "particle_id": 1,
                    "area_px": 120.0,
                    "area_um2": 0.8,
                    "perimeter_px": 44.0,
                    "circularity": 0.78,
                    "equivalent_diameter_px": 12.3,
                    "equivalent_diameter_um": 1.0,
                    "aspect_ratio": 1.0,
                    "area": 120.0,
                    "perimeter": 44.0,
                    "metric_error_code": None,
                    "metric_error_message": None,
                    "is_edge_particle": False,
                    "validation_status": "valid",
                    "rejection_reason": None,
                }
                for image_id, _ in per_image_metrics
            ]
        )

    summary = run_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        config={
            "dependencies": {
                "preprocess_tiff_for_model": preprocess,
                "segmentation_core": segmentation_core,
                "refine_masks_to_labels": refine,
                "extract_metrics": extract,
                "apply_validation_filters": validate,
                "consolidate_filtered_metrics_batch": consolidate,
            }
        },
    )

    assert summary["images_total"] == 2
    assert summary["images_succeeded"] == 1
    assert summary["images_failed"] == 1
    assert summary["status"] == "completed_with_errors"
    assert consolidated_counts == [1]

    failed = next(item for item in summary["images"] if item["image_id"] == "Image_A1")
    success = next(item for item in summary["images"] if item["image_id"] == "Image_B2")

    assert failed["status"] == "failed"
    assert failed["error"] == {
        "code": "ERR_MASK_002",
        "message": ERROR_CATALOG["ERR_MASK_002"],
        "stage": "validation",
        "context": {"valid_particles": 0, "rejection_stats": {"area_range": 1}},
    }
    assert failed["timings"]["total_seconds"] >= 0.0
    assert failed["state_transitions"] == [
        {"from": "raw", "to": "preprocessed"},
        {"from": "preprocessed", "to": "segmented"},
        {"from": "segmented", "to": "refined"},
        {"from": "refined", "to": "analyzed"},
    ]
    assert success["status"] == "success"
    assert Path(summary["exports"]["csv_path"]).exists()
