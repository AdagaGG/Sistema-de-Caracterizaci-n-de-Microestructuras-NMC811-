from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.postprocessing.morphology_ops import (
    apply_opening,
    build_structuring_element,
    refine_masks_to_labels,
)


def test_build_structuring_element_supports_kernel_variants() -> None:
    rect = build_structuring_element("rect", 5)
    ellipse = build_structuring_element("ellipse", 5)
    cross = build_structuring_element("cross", 5)

    assert rect.shape == (5, 5)
    assert ellipse.shape == (5, 5)
    assert cross.shape == (5, 5)
    assert rect.dtype == np.uint8
    assert ellipse.dtype == np.uint8
    assert cross.dtype == np.uint8
    assert int(rect.sum()) > int(cross.sum())
    assert int(ellipse.sum()) < int(rect.sum())


@pytest.mark.parametrize("kernel_size", [0, -3, 4])
def test_build_structuring_element_rejects_invalid_kernel_size(kernel_size: int) -> None:
    with pytest.raises(ValueError):
        build_structuring_element("ellipse", kernel_size)


def test_apply_opening_matches_explicit_erosion_then_dilation() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[3:17, 3:17] = True
    mask[0, 0] = True

    kernel = build_structuring_element("ellipse", 3)
    expected = cv2.dilate(cv2.erode(mask.astype(np.uint8), kernel, iterations=1), kernel, iterations=1) > 0

    opened = apply_opening(mask, kernel_shape="ellipse", kernel_size=3)

    assert opened.dtype == np.bool_
    assert np.array_equal(opened, expected)


def test_refine_masks_to_labels_separates_touching_particles() -> None:
    segmentation = np.zeros((30, 40), dtype=bool)
    segmentation[8:22, 4:14] = True
    segmentation[8:22, 24:34] = True
    segmentation[14:16, 14:24] = True

    labels = refine_masks_to_labels(
        [{"segmentation": segmentation, "area": int(segmentation.sum()), "bbox": (4, 8, 30, 14)}],
        kernel_shape="ellipse",
        kernel_size=3,
        exclude_edge_particles=False,
    )

    assert labels.dtype == np.int32
    assert labels.shape == segmentation.shape
    assert labels.min() == 0
    assert int(labels.max()) == 2
    assert set(np.unique(labels)) == {0, 1, 2}


def test_refine_masks_to_labels_excludes_border_touching_components() -> None:
    interior = np.zeros((24, 24), dtype=bool)
    interior[9:15, 9:15] = True

    border_touching = np.zeros((24, 24), dtype=bool)
    border_touching[1:7, 0:5] = True

    labels = refine_masks_to_labels(
        [
            {"segmentation": interior, "area": int(interior.sum()), "bbox": (9, 9, 6, 6)},
            {"segmentation": border_touching, "area": int(border_touching.sum()), "bbox": (0, 1, 5, 6)},
        ],
        kernel_shape="ellipse",
        kernel_size=3,
        exclude_edge_particles=True,
    )

    assert labels.dtype == np.int32
    assert int(labels.max()) == 1
    assert int(labels[11, 11]) == 1
    assert int(labels[2, 1]) == 0


def test_refine_masks_to_labels_handles_none_area_with_segmentation_fallback() -> None:
    larger = np.zeros((10, 10), dtype=bool)
    larger[1:7, 1:7] = True

    smaller = np.zeros((10, 10), dtype=bool)
    smaller[4:8, 4:8] = True

    labels = refine_masks_to_labels(
        [
            {"segmentation": larger, "area": None, "bbox": (1, 1, 6, 6)},
            {"segmentation": smaller, "area": int(smaller.sum()), "bbox": (4, 4, 4, 4)},
        ],
        kernel_shape="ellipse",
        kernel_size=1,
        exclude_edge_particles=False,
    )

    assert int(labels[2, 2]) == 1
    assert int(labels[7, 7]) == 2


def test_refine_masks_to_labels_handles_malformed_area_with_segmentation_fallback() -> None:
    larger = np.zeros((10, 10), dtype=bool)
    larger[1:7, 1:7] = True

    smaller = np.zeros((10, 10), dtype=bool)
    smaller[4:8, 4:8] = True

    labels = refine_masks_to_labels(
        [
            {"segmentation": larger, "area": "not-a-number", "bbox": (1, 1, 6, 6)},
            {"segmentation": smaller, "area": int(smaller.sum()), "bbox": (4, 4, 4, 4)},
        ],
        kernel_shape="ellipse",
        kernel_size=1,
        exclude_edge_particles=False,
    )

    assert int(labels[2, 2]) == 1
    assert int(labels[7, 7]) == 2
