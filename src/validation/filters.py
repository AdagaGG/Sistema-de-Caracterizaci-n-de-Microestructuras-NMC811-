from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.utils.error_codes import raise_prd_error


@dataclass(frozen=True)
class ValidationConfig:
    area_min_px: float = 50.0
    area_max_px: float = 10000.0
    aspect_ratio_min: float = 0.25
    aspect_ratio_max: float = 4.0
    exclude_edge_particles: bool = True
    circularity_min: float | None = None


def _compute_is_edge_particle(frame: pd.DataFrame, image_shape: tuple[int, int] | None) -> pd.Series:
    if "is_edge_particle" in frame.columns:
        return frame["is_edge_particle"].map(lambda value: bool(value) if pd.notna(value) else False).astype(bool)

    if "bbox" in frame.columns and image_shape is not None:
        height, width = image_shape

        def _touches_edge(value: Any) -> bool:
            if value is None:
                return False
            try:
                min_row, min_col, max_row, max_col = value
            except (TypeError, ValueError):
                return False
            return bool(min_row <= 0 or min_col <= 0 or max_row >= height or max_col >= width)

        return frame["bbox"].map(_touches_edge).astype(bool)

    return pd.Series(False, index=frame.index, dtype=bool)


def apply_validation_filters(
    metrics: pd.DataFrame,
    config: ValidationConfig | None = None,
    image_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    cfg = config or ValidationConfig()
    annotated = metrics.copy(deep=True)

    if annotated.empty:
        raise_prd_error("ERR_MASK_002", stage="validation", context={"valid_particles": 0, "rejection_stats": {}})

    annotated["is_edge_particle"] = _compute_is_edge_particle(annotated, image_shape=image_shape)
    annotated["validation_status"] = "valid"
    annotated["rejection_reason"] = pd.Series([pd.NA] * len(annotated), index=annotated.index, dtype="object")

    rejection_stats: dict[str, int] = {
        "area_range": 0,
        "aspect_ratio_range": 0,
        "edge_particle": 0,
        "circularity_min": 0,
    }

    area_rejected = (annotated["area_px"] < cfg.area_min_px) | (annotated["area_px"] > cfg.area_max_px)
    area_rejected = area_rejected & (annotated["validation_status"] == "valid")
    annotated.loc[area_rejected, "validation_status"] = "rejected"
    annotated.loc[area_rejected, "rejection_reason"] = "area_range"
    rejection_stats["area_range"] = int(area_rejected.sum())

    aspect_rejected = (annotated["aspect_ratio"] < cfg.aspect_ratio_min) | (annotated["aspect_ratio"] > cfg.aspect_ratio_max)
    aspect_rejected = aspect_rejected & (annotated["validation_status"] == "valid")
    annotated.loc[aspect_rejected, "validation_status"] = "rejected"
    annotated.loc[aspect_rejected, "rejection_reason"] = "aspect_ratio_range"
    rejection_stats["aspect_ratio_range"] = int(aspect_rejected.sum())

    if cfg.exclude_edge_particles:
        edge_rejected = annotated["is_edge_particle"] & (annotated["validation_status"] == "valid")
    else:
        edge_rejected = pd.Series(False, index=annotated.index, dtype=bool)
    annotated.loc[edge_rejected, "validation_status"] = "rejected"
    annotated.loc[edge_rejected, "rejection_reason"] = "edge_particle"
    rejection_stats["edge_particle"] = int(edge_rejected.sum())

    if cfg.circularity_min is None:
        circularity_rejected = pd.Series(False, index=annotated.index, dtype=bool)
    else:
        circularity_rejected = (annotated["circularity"] < cfg.circularity_min) & (annotated["validation_status"] == "valid")
    annotated.loc[circularity_rejected, "validation_status"] = "rejected"
    annotated.loc[circularity_rejected, "rejection_reason"] = "circularity_min"
    rejection_stats["circularity_min"] = int(circularity_rejected.sum())

    valid_particles = annotated.loc[annotated["validation_status"] == "valid"].copy()

    if valid_particles.empty:
        raise_prd_error(
            "ERR_MASK_002",
            stage="validation",
            context={"valid_particles": 0, "rejection_stats": rejection_stats},
        )

    return {
        "annotated_particles": annotated.reset_index(drop=True),
        "valid_particles": valid_particles.reset_index(drop=True),
        "rejection_stats": rejection_stats,
        "config": {
            "area_range_pixels": [cfg.area_min_px, cfg.area_max_px],
            "aspect_ratio_range": [cfg.aspect_ratio_min, cfg.aspect_ratio_max],
            "exclude_edge_particles": cfg.exclude_edge_particles,
            "circularity_min": cfg.circularity_min,
        },
    }

