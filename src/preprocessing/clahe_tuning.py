from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
import yaml


@dataclass(frozen=True)
class ClaheCandidate:
    clip_limit: float
    tile_grid_size: tuple[int, int]


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


def _compute_quality_score(enhanced_uint8: np.ndarray) -> float:
    std_score = float(np.std(enhanced_uint8)) / 255.0
    lap_var = float(cv2.Laplacian(enhanced_uint8, cv2.CV_64F).var())
    lap_score = min(lap_var / 2000.0, 1.0)
    return 0.55 * std_score + 0.45 * lap_score


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


def _evaluate_candidate(normalized_images: list[np.ndarray], candidate: ClaheCandidate) -> dict[str, Any]:
    per_image_scores: list[float] = []
    clahe = cv2.createCLAHE(clipLimit=candidate.clip_limit, tileGridSize=candidate.tile_grid_size)

    for image_uint8 in normalized_images:
        enhanced = clahe.apply(image_uint8)
        per_image_scores.append(_compute_quality_score(enhanced))

    mean_score = float(np.mean(per_image_scores))
    std_score = float(np.std(per_image_scores))
    robust_score = mean_score - 0.35 * std_score

    return {
        "clip_limit": float(candidate.clip_limit),
        "tile_grid_size": [int(candidate.tile_grid_size[0]), int(candidate.tile_grid_size[1])],
        "mean_score": round(mean_score, 6),
        "std_score": round(std_score, 6),
        "robust_score": round(robust_score, 6),
        "per_image_scores": [round(score, 6) for score in per_image_scores],
    }


def tune_clahe_parameters(
    image_paths: list[Path],
    clip_limits: list[float],
    tile_grid_sizes: list[tuple[int, int]],
    representative_count: int = 4,
) -> dict[str, Any]:
    representative_images = _select_representative_images(image_paths, representative_count=representative_count)
    normalized_images = [_load_and_normalize(path) for path in representative_images]

    candidates = [
        ClaheCandidate(clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        for clip_limit in clip_limits
        for tile_grid_size in tile_grid_sizes
    ]

    evaluated_combinations = [_evaluate_candidate(normalized_images, candidate) for candidate in candidates]
    evaluated_combinations.sort(
        key=lambda x: (
            x["robust_score"],
            -abs(x["clip_limit"] - 2.5),
            -abs(x["tile_grid_size"][0] - 8),
        ),
        reverse=True,
    )
    selected = evaluated_combinations[0]

    rationale = (
        "Selected the highest robust_score (mean quality penalized by cross-image variability) "
        "across representative samples."
    )

    return {
        "search_space": {
            "clip_limits": [float(value) for value in clip_limits],
            "tile_grid_sizes": [[int(t[0]), int(t[1])] for t in tile_grid_sizes],
            "representative_count": representative_count,
        },
        "representative_images": [path.name for path in representative_images],
        "selected_parameters": {
            "clip_limit": selected["clip_limit"],
            "tile_grid_size": list(selected["tile_grid_size"]),
            "robust_score": selected["robust_score"],
        },
        "selection_rationale": rationale,
        "evaluated_combinations": evaluated_combinations,
    }


def persist_clahe_defaults(config_path: Path, clip_limit: float, tile_grid_size: tuple[int, int]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}

    if config_path.exists():
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    preprocessing = payload.setdefault("preprocessing", {})
    clahe = preprocessing.setdefault("clahe", {})
    clahe["clip_limit"] = float(clip_limit)
    clahe["tile_grid_size"] = [int(tile_grid_size[0]), int(tile_grid_size[1])]

    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def persist_tuning_record(record_path: Path, tuning_record: dict[str, Any]) -> None:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(yaml.safe_dump(tuning_record, sort_keys=False), encoding="utf-8")
