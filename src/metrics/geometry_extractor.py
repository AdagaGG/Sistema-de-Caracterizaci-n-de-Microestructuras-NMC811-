from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from skimage.measure import regionprops

from src.utils.error_codes import ERROR_CATALOG

PIXEL_SCALE_UM = 0.08431

METRICS_SCHEMA_COLUMNS = [
    "id",
    "area_px",
    "area_um2",
    "perimeter_px",
    "circularity",
    "equivalent_diameter_px",
    "equivalent_diameter_um",
    "aspect_ratio",
    "area",
    "perimeter",
    "metric_error_code",
    "metric_error_message",
]


def _empty_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=METRICS_SCHEMA_COLUMNS)


def _as_labeled_int32(mask_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(mask_labels)
    if labels.ndim != 2:
        raise ValueError("mask_labels must be a 2D labeled mask")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("mask_labels must use an integer dtype")
    if np.any(labels < 0):
        raise ValueError("mask_labels must contain non-negative labels")
    return labels.astype(np.int32, copy=False)


def _aspect_ratio_from_bbox(region: Any) -> float:
    min_row, min_col, max_row, max_col = region.bbox
    width = float(max_col - min_col)
    height = float(max_row - min_row)
    if height == 0.0:
        return 0.0
    return width / height


def extract_metrics(mask_labels: np.ndarray) -> pd.DataFrame:
    labels = _as_labeled_int32(mask_labels)
    if labels.size == 0 or int(labels.max()) == 0:
        return _empty_metrics_frame()

    rows: list[dict[str, Any]] = []

    for region in regionprops(labels):
        area_px = float(region.area)
        perimeter_px = float(region.perimeter)

        metric_error_code: str | None = None
        metric_error_message: str | None = None
        if perimeter_px <= 0.0:
            circularity = 0.0
            metric_error_code = "ERR_METRIC_004"
            metric_error_message = ERROR_CATALOG[metric_error_code]
        else:
            circularity = float(4.0 * np.pi * area_px / (perimeter_px**2))

        equivalent_diameter_px = float(np.sqrt(4.0 * area_px / np.pi)) if area_px > 0 else 0.0
        aspect_ratio = _aspect_ratio_from_bbox(region)

        rows.append(
            {
                "id": int(region.label),
                "area_px": area_px,
                "area_um2": area_px * (PIXEL_SCALE_UM**2),
                "perimeter_px": perimeter_px,
                "circularity": circularity,
                "equivalent_diameter_px": equivalent_diameter_px,
                "equivalent_diameter_um": equivalent_diameter_px * PIXEL_SCALE_UM,
                "aspect_ratio": aspect_ratio,
                "area": area_px,
                "perimeter": perimeter_px,
                "metric_error_code": metric_error_code,
                "metric_error_message": metric_error_message,
            }
        )

    frame = pd.DataFrame.from_records(rows, columns=METRICS_SCHEMA_COLUMNS)
    return frame.sort_values(by="id", kind="stable").reset_index(drop=True)
