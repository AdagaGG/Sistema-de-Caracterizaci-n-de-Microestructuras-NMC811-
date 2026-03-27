from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from src.pipeline import run_batch
from src.segmentation import MaskGeneratorConfig, MobileSamInferenceCore, ModelLoadConfig

_MICROMETERS_PER_PIXEL = 0.08431
_AREA_CONVERSION_UM2_TO_PX2 = 1.0 / (_MICROMETERS_PER_PIXEL**2)


def _normalize_batch_result(batch_result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(batch_result)
    normalized["total_images"] = int(batch_result.get("total_images", batch_result.get("images_total", 0)))
    normalized["completed_count"] = int(batch_result.get("completed_count", batch_result.get("images_succeeded", 0)))
    normalized["failed_count"] = int(batch_result.get("failed_count", batch_result.get("images_failed", 0)))
    normalized["results"] = list(batch_result.get("results", batch_result.get("images", [])))
    return normalized


def _resolve_path(candidate: str | Path, output_dir: Path) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    return output_dir / path


def _build_segmentation_core(ui_params: dict[str, Any]) -> MobileSamInferenceCore:
    checkpoint_path = str(ui_params.get("checkpoint_path", "")).strip()
    device = str(ui_params.get("device", "cpu")).strip() or "cpu"

    model_config = ModelLoadConfig(
        backend="mobilesam",
        checkpoint_path=checkpoint_path,
        model_type="vit_t",
        device=device,
        config_path=None,
    )
    generator_config = MaskGeneratorConfig()
    return MobileSamInferenceCore.from_config(
        model_config=model_config,
        generator_config=generator_config,
    )


def _build_pipeline_config(ui_params: dict[str, Any]) -> dict[str, Any]:
    tile = int(ui_params.get("clahe_tile_grid_size", 8))
    min_area_um2 = float(ui_params.get("min_area_um2", 5.0))
    max_area_um2 = float(ui_params.get("max_area_um2", 500.0))
    min_area_px2 = min_area_um2 * _AREA_CONVERSION_UM2_TO_PX2
    max_area_px2 = max_area_um2 * _AREA_CONVERSION_UM2_TO_PX2
    segmentation_core = _build_segmentation_core(ui_params)

    return {
        "dependencies": {
            "segmentation_core": segmentation_core,
        },
        "preprocessing": {
            "clip_limit": float(ui_params.get("clahe_clip_limit", 2.0)),
            "tile_grid_size": (tile, tile),
        },
        "validation": {
            "area_min_px": min_area_px2,
            "area_max_px": max_area_px2,
            "circularity_min": float(ui_params.get("min_circularity", 0.3)),
        },
    }


def get_artifact_paths(output_dir: Path) -> dict[str, Any]:
    resolved_output_dir = Path(output_dir)
    csv_path = resolved_output_dir / "analisis_gold_standard.csv"
    manifest_path = resolved_output_dir / "visualizations" / "batch__heatmap_circularity_manifest.json"

    manifest_data: dict[str, Any] | None = None
    heatmap_paths: list[str] = []
    mask_paths: list[str] = []
    legend_paths: list[str] = []
    meta_paths: list[str] = []

    if manifest_path.exists():
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest_data.get("files", []):
            overlay_path = item.get("overlay_path")
            mask_path = item.get("mask_path")
            legend_path = item.get("legend_path")
            meta_path = item.get("meta_path")

            if overlay_path:
                resolved = _resolve_path(overlay_path, resolved_output_dir)
                if resolved.exists():
                    heatmap_paths.append(str(resolved))
            if mask_path:
                resolved = _resolve_path(mask_path, resolved_output_dir)
                if resolved.exists():
                    mask_paths.append(str(resolved))
            if legend_path:
                resolved = _resolve_path(legend_path, resolved_output_dir)
                if resolved.exists():
                    legend_paths.append(str(resolved))
            if meta_path:
                resolved = _resolve_path(meta_path, resolved_output_dir)
                if resolved.exists():
                    meta_paths.append(str(resolved))

    return {
        "output_dir": str(resolved_output_dir),
        "csv_path": str(csv_path) if csv_path.exists() else None,
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "heatmap_paths": heatmap_paths,
        "mask_paths": mask_paths,
        "legend_paths": legend_paths,
        "meta_paths": meta_paths,
        "manifest": manifest_data,
    }


def run_pipeline_on_uploads(uploaded_files: list, ui_params: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="nmc811-ui-input-") as temp_input:
        input_dir = Path(temp_input)

        for file_payload in uploaded_files:
            filename = str(file_payload.get("filename", "")).strip()
            file_bytes = file_payload.get("bytes", b"")
            if not filename or not isinstance(file_bytes, (bytes, bytearray)):
                continue
            if len(file_bytes) == 0:
                continue
            (input_dir / filename).write_bytes(bytes(file_bytes))

        output_dir = Path(tempfile.mkdtemp(prefix="nmc811-ui-output-"))
        config = _build_pipeline_config(ui_params)
        batch_result = run_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            config=config,
        )

    normalized_batch = _normalize_batch_result(batch_result)
    artifact_paths = get_artifact_paths(output_dir)

    return {
        "batch_result": normalized_batch,
        "artifact_paths": artifact_paths,
        "output_dir": str(output_dir),
    }
