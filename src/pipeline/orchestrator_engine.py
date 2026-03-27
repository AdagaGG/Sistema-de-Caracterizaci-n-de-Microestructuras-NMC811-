from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable

import pandas as pd

from src.export.csv_export import consolidate_filtered_metrics_batch, write_analisis_gold_standard_csv
from src.metrics.geometry_extractor import extract_metrics
from src.postprocessing.morphology_ops import refine_masks_to_labels
from src.preprocessing.pipeline import preprocess_tiff_for_model
from src.segmentation.mobilesam_inference import MobileSamInferenceCore
from src.utils.error_codes import serialize_error
from src.validation.filters import ValidationConfig, apply_validation_filters
from src.visualization.heatmaps import generate_circularity_heatmap_artifacts, write_heatmap_batch_manifest

_STAGE_SEQUENCE = ["preprocessed", "segmented", "refined", "analyzed", "exported"]
_STATE_CHAIN = ["raw", "preprocessed", "segmented", "refined", "analyzed", "exported"]


def _list_input_images(input_dir: Path) -> list[Path]:
    return sorted([*input_dir.glob("*.tif"), *input_dir.glob("*.tiff")], key=lambda path: path.name.lower())


def _state_transitions_for(state: str) -> list[dict[str, str]]:
    if state == "raw":
        return []

    transitions: list[dict[str, str]] = []
    for idx in range(len(_STATE_CHAIN) - 1):
        transitions.append({"from": _STATE_CHAIN[idx], "to": _STATE_CHAIN[idx + 1]})
        if _STATE_CHAIN[idx + 1] == state:
            break
    return transitions


def _resolve_dependencies(config: dict[str, Any]) -> dict[str, Any]:
    deps = dict(config.get("dependencies", {}))

    segmentation_core = deps.get("segmentation_core")
    if segmentation_core is None:
        segmentation_core = config.get("segmentation_core")
    if segmentation_core is None:
        raise ValueError("segmentation_core dependency is required")

    return {
        "preprocess_tiff_for_model": deps.get("preprocess_tiff_for_model", preprocess_tiff_for_model),
        "segmentation_core": segmentation_core,
        "refine_masks_to_labels": deps.get("refine_masks_to_labels", refine_masks_to_labels),
        "extract_metrics": deps.get("extract_metrics", extract_metrics),
        "apply_validation_filters": deps.get("apply_validation_filters", apply_validation_filters),
        "generate_circularity_heatmap_artifacts": deps.get(
            "generate_circularity_heatmap_artifacts",
            generate_circularity_heatmap_artifacts,
        ),
        "write_heatmap_batch_manifest": deps.get("write_heatmap_batch_manifest", write_heatmap_batch_manifest),
        "consolidate_filtered_metrics_batch": deps.get(
            "consolidate_filtered_metrics_batch",
            consolidate_filtered_metrics_batch,
        ),
        "write_analisis_gold_standard_csv": deps.get(
            "write_analisis_gold_standard_csv",
            write_analisis_gold_standard_csv,
        ),
    }


def run_batch(input_dir: str | Path, output_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    started_at = time.perf_counter()
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    dependencies = _resolve_dependencies(config=config)
    preprocessing_fn: Callable[..., Any] = dependencies["preprocess_tiff_for_model"]
    segmentation_core: MobileSamInferenceCore = dependencies["segmentation_core"]
    refine_fn: Callable[..., Any] = dependencies["refine_masks_to_labels"]
    metrics_fn: Callable[..., pd.DataFrame] = dependencies["extract_metrics"]
    validation_fn: Callable[..., dict[str, Any]] = dependencies["apply_validation_filters"]
    heatmap_fn: Callable[..., dict[str, Any]] = dependencies["generate_circularity_heatmap_artifacts"]
    manifest_fn: Callable[..., Path] = dependencies["write_heatmap_batch_manifest"]
    consolidate_fn: Callable[..., pd.DataFrame] = dependencies["consolidate_filtered_metrics_batch"]
    write_csv_fn: Callable[..., Path] = dependencies["write_analisis_gold_standard_csv"]

    preprocessing_cfg = dict(config.get("preprocessing", {}))
    refinement_cfg = dict(config.get("refinement", {}))
    validation_cfg = config.get("validation", {})
    validation_config = ValidationConfig(**validation_cfg)
    vram_stress_limit_bytes = config.get("vram_stress_limit_bytes")

    image_summaries: list[dict[str, Any]] = []
    per_image_valid_metrics: list[tuple[str, pd.DataFrame]] = []
    heatmap_entries: list[dict[str, Any]] = []

    for image_path in _list_input_images(input_root):
        image_id = image_path.stem
        timings: dict[str, float] = {}
        image_started_at = time.perf_counter()
        current_state = "raw"

        try:
            stage_start = time.perf_counter()
            preprocessed_image = preprocessing_fn(image_path, **preprocessing_cfg)
            timings["preprocessed"] = time.perf_counter() - stage_start
            current_state = "preprocessed"

            stage_start = time.perf_counter()
            if vram_stress_limit_bytes is None:
                segmentation_result = segmentation_core.segment_image(preprocessed_image)
            else:
                segmentation_result = segmentation_core.segment_image(
                    preprocessed_image,
                    vram_stress_limit_bytes=vram_stress_limit_bytes,
                )
            timings["segmented"] = time.perf_counter() - stage_start
            current_state = "segmented"

            stage_start = time.perf_counter()
            refined_labels = refine_fn(segmentation_result["masks"], **refinement_cfg)
            timings["refined"] = time.perf_counter() - stage_start
            current_state = "refined"

            stage_start = time.perf_counter()
            current_state = "analyzed"
            metrics = metrics_fn(refined_labels)
            validation_payload = validation_fn(metrics, config=validation_config, image_shape=preprocessed_image.shape[:2])
            timings["analyzed"] = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            current_state = "exported"
            heatmap_entry = heatmap_fn(
                image=preprocessed_image,
                annotated_particles=validation_payload["annotated_particles"],
                image_stem=image_id,
                output_root=output_root,
            )
            timings["exported"] = time.perf_counter() - stage_start

            per_image_valid_metrics.append((image_id, validation_payload["valid_particles"]))
            heatmap_entries.append(heatmap_entry)

            total_seconds = time.perf_counter() - image_started_at
            timings["total_seconds"] = total_seconds
            image_summaries.append(
                {
                    "image_id": image_id,
                    "status": "success",
                    "state": current_state,
                    "state_transitions": _state_transitions_for(current_state),
                    "timings": timings,
                    "error": None,
                }
            )
        except Exception as error:  # noqa: BLE001
            total_seconds = time.perf_counter() - image_started_at
            timings["total_seconds"] = total_seconds
            image_summaries.append(
                {
                    "image_id": image_id,
                    "status": "failed",
                    "state": current_state,
                    "state_transitions": _state_transitions_for(current_state),
                    "timings": timings,
                    "error": serialize_error(error),
                }
            )

    merged_metrics = consolidate_fn(per_image_valid_metrics)
    csv_path = write_csv_fn(merged_metrics=merged_metrics, output_root=output_root)
    heatmap_manifest_path = manifest_fn(entries=heatmap_entries, output_root=output_root)

    success_count = sum(1 for item in image_summaries if item["status"] == "success")
    failed_count = sum(1 for item in image_summaries if item["status"] == "failed")

    return {
        "status": "completed" if failed_count == 0 else "completed_with_errors",
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "images_total": len(image_summaries),
        "images_succeeded": success_count,
        "images_failed": failed_count,
        "images": image_summaries,
        "exports": {
            "csv_path": str(csv_path),
            "heatmap_manifest_path": str(heatmap_manifest_path),
        },
        "timings": {
            "total_seconds": time.perf_counter() - started_at,
        },
    }


__all__ = ["run_batch"]
