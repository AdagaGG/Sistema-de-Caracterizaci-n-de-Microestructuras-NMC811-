from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from skimage.segmentation import clear_border


_KERNEL_SHAPES: dict[str, int] = {
    "rect": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS,
}


def _safe_mask_area(item: dict[str, Any]) -> int:
    raw_area = item.get("area")
    if raw_area is not None:
        if isinstance(raw_area, bool):
            raw_area = int(raw_area)
        if isinstance(raw_area, (int, np.integer)):
            return int(raw_area)
        if isinstance(raw_area, (float, np.floating)):
            if np.isfinite(raw_area):
                return int(raw_area)
        if isinstance(raw_area, str):
            trimmed = raw_area.strip()
            if trimmed:
                try:
                    parsed = float(trimmed)
                except ValueError:
                    pass
                else:
                    if np.isfinite(parsed):
                        return int(parsed)

    return int(np.asarray(item["segmentation"], dtype=bool).sum())


def build_structuring_element(kernel_shape: str = "ellipse", kernel_size: int = 3) -> np.ndarray:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    normalized_shape = kernel_shape.lower()
    if normalized_shape not in _KERNEL_SHAPES:
        raise ValueError(f"Unsupported kernel_shape: {kernel_shape}")

    return cv2.getStructuringElement(_KERNEL_SHAPES[normalized_shape], (kernel_size, kernel_size))


def apply_opening(
    mask: np.ndarray,
    kernel_shape: str = "ellipse",
    kernel_size: int = 3,
) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    kernel = build_structuring_element(kernel_shape=kernel_shape, kernel_size=kernel_size)

    eroded = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=1)
    opened = cv2.dilate(eroded, kernel, iterations=1)
    return opened.astype(bool)


def refine_masks_to_labels(
    masks: list[dict[str, Any]],
    kernel_shape: str = "ellipse",
    kernel_size: int = 3,
    exclude_edge_particles: bool = True,
) -> np.ndarray:
    if not masks:
        return np.zeros((0, 0), dtype=np.int32)

    ordered = sorted(
        masks,
        key=_safe_mask_area,
        reverse=True,
    )

    first_shape = np.asarray(ordered[0]["segmentation"], dtype=bool).shape
    label_map = np.zeros(first_shape, dtype=np.int32)
    next_label = 1

    for item in ordered:
        segmentation = np.asarray(item["segmentation"], dtype=bool)
        if segmentation.shape != first_shape:
            raise ValueError("All mask segmentations must have the same shape")

        opened = apply_opening(segmentation, kernel_shape=kernel_shape, kernel_size=kernel_size)
        if exclude_edge_particles:
            opened = clear_border(opened)

        component_count, components = cv2.connectedComponents(opened.astype(np.uint8), connectivity=8)
        for component_id in range(1, component_count):
            component_pixels = components == component_id
            write_region = component_pixels & (label_map == 0)
            if not np.any(write_region):
                continue
            label_map[write_region] = next_label
            next_label += 1

    return label_map


__all__ = [
    "build_structuring_element",
    "apply_opening",
    "refine_masks_to_labels",
]
