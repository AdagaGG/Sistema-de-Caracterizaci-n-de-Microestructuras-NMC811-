from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
import yaml

from .morphology_ops import apply_opening


@dataclass(frozen=True)
class MorphologyCandidate:
    kernel_shape: str
    kernel_size: int


def _normalize_uint16_to_uint8(image_uint16: np.ndarray) -> np.ndarray:
    normalized_uint8 = np.empty_like(image_uint16, dtype=np.uint8)
    cv2.normalize(
        image_uint16,
        normalized_uint8,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    return normalized_uint8


def _load_and_normalize(path: Path) -> np.ndarray:
    image = tifffile.imread(path)
    if image.dtype != np.uint16:
        image = image.astype(np.uint16, copy=False)
    return _normalize_uint16_to_uint8(image)


def _build_binary_particle_mask(image_uint8: np.ndarray) -> np.ndarray:
    _, threshold_binary = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, threshold_inverted = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mask_binary = threshold_binary > 0
    mask_inverted = threshold_inverted > 0

    ratio_binary = float(mask_binary.mean())
    ratio_inverted = float(mask_inverted.mean())
    if ratio_binary <= ratio_inverted:
        return mask_binary
    return mask_inverted


def _component_count(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return max(int(count) - 1, 0)


def _evaluate_candidate(masks: list[np.ndarray], candidate: MorphologyCandidate) -> dict[str, Any]:
    per_image: list[dict[str, float]] = []

    for mask in masks:
        baseline_components = _component_count(mask)
        baseline_area = int(mask.sum())

        opened = apply_opening(mask, kernel_shape=candidate.kernel_shape, kernel_size=candidate.kernel_size)
        opened_components = _component_count(opened)
        opened_area = int(opened.sum())

        separation_score = max(opened_components - baseline_components, 0) / float(baseline_components + 1)
        erosion_penalty = max(baseline_area - opened_area, 0) / float(max(baseline_area, 1))
        quality_score = 0.68 * min(separation_score, 1.0) + 0.32 * (1.0 - erosion_penalty)

        per_image.append(
            {
                "baseline_components": float(baseline_components),
                "opened_components": float(opened_components),
                "separation_score": separation_score,
                "erosion_penalty": erosion_penalty,
                "quality_score": quality_score,
            }
        )

    mean_separation = float(np.mean([entry["separation_score"] for entry in per_image]))
    mean_erosion_penalty = float(np.mean([entry["erosion_penalty"] for entry in per_image]))
    mean_quality = float(np.mean([entry["quality_score"] for entry in per_image]))
    quality_std = float(np.std([entry["quality_score"] for entry in per_image]))
    robust_quality = mean_quality - 0.35 * quality_std

    return {
        "kernel_shape": candidate.kernel_shape,
        "kernel_size": int(candidate.kernel_size),
        "quality_score": round(robust_quality, 6),
        "separation_score": round(mean_separation, 6),
        "erosion_penalty": round(mean_erosion_penalty, 6),
        "mean_quality": round(mean_quality, 6),
        "quality_std": round(quality_std, 6),
    }


def _select_representative_images(image_paths: list[Path], representative_count: int) -> list[Path]:
    if representative_count <= 0:
        raise ValueError("representative_count must be > 0")
    if not image_paths:
        raise ValueError("image_paths must not be empty")

    sorted_paths = sorted(image_paths)
    count = min(representative_count, len(sorted_paths))
    if count == len(sorted_paths):
        return sorted_paths

    indices = np.linspace(0, len(sorted_paths) - 1, num=count, dtype=int)
    return [sorted_paths[idx] for idx in indices]


def tune_morphology_parameters(
    image_paths: list[Path],
    kernel_shapes: list[str],
    kernel_sizes: list[int],
    representative_count: int = 4,
) -> dict[str, Any]:
    representative_images = _select_representative_images(image_paths, representative_count=representative_count)
    normalized_images = [_load_and_normalize(path) for path in representative_images]
    base_masks = [_build_binary_particle_mask(image) for image in normalized_images]

    candidates = [
        MorphologyCandidate(kernel_shape=kernel_shape, kernel_size=kernel_size)
        for kernel_shape in kernel_shapes
        for kernel_size in kernel_sizes
    ]
    if not candidates:
        raise ValueError("kernel_shapes and kernel_sizes must define at least one candidate")

    evaluated_variants = [_evaluate_candidate(base_masks, candidate) for candidate in candidates]
    evaluated_variants.sort(
        key=lambda variant: (
            variant["quality_score"],
            -variant["erosion_penalty"],
            1 if variant["kernel_shape"] == "ellipse" else 0,
            -abs(variant["kernel_size"] - 5),
        ),
        reverse=True,
    )

    selected = evaluated_variants[0]
    rationale = (
        "Selected highest robust quality score (separation benefit minus erosion risk and cross-image instability), "
        "with ellipse preference and mid-size tie-break for guarded defaults."
    )

    return {
        "search_space": {
            "kernel_shapes": [shape for shape in kernel_shapes],
            "kernel_sizes": [int(size) for size in kernel_sizes],
            "representative_count": int(representative_count),
        },
        "representative_images": [path.name for path in representative_images],
        "selected_parameters": {
            "kernel_shape": selected["kernel_shape"],
            "kernel_size": int(selected["kernel_size"]),
            "quality_score": selected["quality_score"],
            "separation_score": selected["separation_score"],
            "erosion_penalty": selected["erosion_penalty"],
        },
        "selection_rationale": rationale,
        "evaluated_variants": evaluated_variants,
    }


def persist_morphology_defaults(config_path: Path, kernel_shape: str, kernel_size: int) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}

    if config_path.exists():
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    postprocessing = payload.setdefault("postprocessing", {})
    morphology = postprocessing.setdefault("morphology", {})
    morphology["kernel_shape"] = str(kernel_shape)
    morphology["kernel_size"] = int(kernel_size)

    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def persist_tuning_record(record_path: Path, tuning_record: dict[str, Any]) -> None:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(yaml.safe_dump(tuning_record, sort_keys=False), encoding="utf-8")

