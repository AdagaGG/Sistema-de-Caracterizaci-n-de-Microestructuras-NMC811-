from __future__ import annotations

from typing import Any

import cv2
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
    "circularity_effective",
    "equivalent_diameter_px",
    "equivalent_diameter_um",
    "aspect_ratio",
    "perimeter_um",
    "mean_intensity",
    "std_intensity",
    "dark_area_fraction",
    "crack_severity",
    "bbox",
    "contour_points",
    "centroid_xy",
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


def _contour_points_from_region(region: Any) -> list[list[int]] | None:
    """Build global contour points [x, y] from a labeled region."""
    region_mask = (region.image.astype(np.uint8, copy=False)) * 255
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if contour.shape[0] < 3:
        return None

    min_row, min_col, _, _ = region.bbox
    points: list[list[int]] = []
    for point in contour.reshape(-1, 2):
        x = int(point[0]) + int(min_col)
        y = int(point[1]) + int(min_row)
        points.append([x, y])
    return points


def _to_gray_image(intensity_image: np.ndarray | None, expected_shape: tuple[int, int]) -> np.ndarray | None:
    if intensity_image is None:
        return None
    image = np.asarray(intensity_image)
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image.astype(np.uint8, copy=False), cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("intensity_image must be grayscale (H, W) or BGR (H, W, 3)")
    if gray.shape != expected_shape:
        raise ValueError("intensity_image shape must match mask_labels shape")
    return gray.astype(np.float32, copy=False)


def _internal_defect_metrics(region: Any, gray_image: np.ndarray | None) -> tuple[float | None, float | None, float | None, float]:
    if gray_image is None:
        return None, None, None, 0.0

    min_row, min_col, max_row, max_col = region.bbox
    local_gray = gray_image[min_row:max_row, min_col:max_col]
    local_mask = region.image.astype(bool, copy=False)
    values = local_gray[local_mask]
    if values.size == 0:
        return None, None, None, 0.0

    mean_intensity = float(np.mean(values))
    std_intensity = float(np.std(values))
    dark_threshold = 0.65 * mean_intensity
    dark_area_fraction = float(np.mean(values < dark_threshold))
    p10_intensity = float(np.percentile(values, 10))
    contrast_drop = max(0.0, (mean_intensity - p10_intensity) / max(mean_intensity, 1e-6))

    # Blend dark-volume fraction with low-tail contrast drop to catch internal fissures/voids.
    crack_severity = float(
        np.clip(0.7 * (dark_area_fraction / 0.25) + 0.3 * (contrast_drop / 0.70), 0.0, 1.0)
    )
    return mean_intensity, std_intensity, dark_area_fraction, crack_severity


def extract_metrics(mask_labels: np.ndarray, intensity_image: np.ndarray | None = None) -> pd.DataFrame:
    labels = _as_labeled_int32(mask_labels)
    if labels.size == 0 or int(labels.max()) == 0:
        return _empty_metrics_frame()
    expected_shape = (int(labels.shape[0]), int(labels.shape[1]))
    gray_image = _to_gray_image(intensity_image, expected_shape=expected_shape)

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
        mean_intensity, std_intensity, dark_area_fraction, crack_severity = _internal_defect_metrics(region, gray_image)
        circularity_effective = float(np.clip(circularity * (1.0 - 0.60 * crack_severity), 0.0, 1.0))
        contour_points = _contour_points_from_region(region)
        min_row, min_col, max_row, max_col = region.bbox
        bbox = [int(min_row), int(min_col), int(max_row), int(max_col)]
        centroid_xy = [float(region.centroid[1]), float(region.centroid[0])]

        rows.append(
            {
                "id": int(region.label),
                "area_px": area_px,
                "area_um2": area_px * (PIXEL_SCALE_UM**2),
                "perimeter_px": perimeter_px,
                "perimeter_um": perimeter_px * PIXEL_SCALE_UM,
                "circularity": circularity,
                "circularity_effective": circularity_effective,
                "equivalent_diameter_px": equivalent_diameter_px,
                "equivalent_diameter_um": equivalent_diameter_px * PIXEL_SCALE_UM,
                "aspect_ratio": aspect_ratio,
                "mean_intensity": mean_intensity,
                "std_intensity": std_intensity,
                "dark_area_fraction": dark_area_fraction,
                "crack_severity": crack_severity,
                "bbox": bbox,
                "contour_points": contour_points,
                "centroid_xy": centroid_xy,
                "area": area_px,
                "perimeter": perimeter_px,
                "metric_error_code": metric_error_code,
                "metric_error_message": metric_error_message,
            }
        )

    frame = pd.DataFrame.from_records(rows, columns=METRICS_SCHEMA_COLUMNS)
    return frame.sort_values(by="id", kind="stable").reset_index(drop=True)
