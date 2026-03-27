from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

_RED_BRG = (39, 48, 215)  # #D73027 in BGR
_MID_BRG = (139, 224, 254)  # #FEE08B in BGR
_GREEN_BRG = (80, 152, 26)  # #1A9850 in BGR
_REJECTED_BRG = (128, 128, 128)  # #808080 in BGR

_OVERLAY_SUFFIX = "__heatmap_circularity_overlay.png"
_MASK_SUFFIX = "__heatmap_circularity_mask.png"
_LEGEND_SUFFIX = "__heatmap_circularity_legend.png"
_META_SUFFIX = "__heatmap_circularity_meta.json"
_BATCH_MANIFEST_NAME = "batch__heatmap_circularity_manifest.json"


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image.astype(np.uint8, copy=False), cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.astype(np.uint8, copy=False)
    raise ValueError("image must be grayscale (H, W) or BGR (H, W, 3)")


def _interpolate_bgr(start: tuple[int, int, int], end: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    tt = float(np.clip(t, 0.0, 1.0))
    return (
        int(round(start[0] + (end[0] - start[0]) * tt)),
        int(round(start[1] + (end[1] - start[1]) * tt)),
        int(round(start[2] + (end[2] - start[2]) * tt)),
    )


def normalize_circularity(value: Any) -> tuple[float, bool, bool]:
    numeric = float(value)
    if not np.isfinite(numeric):
        return 0.0, False, True
    clipped = numeric < 0.0 or numeric > 1.0
    norm = float(np.clip(numeric, 0.0, 1.0))
    return norm, clipped, False


def circularity_to_bgr(circularity: float) -> tuple[int, int, int]:
    norm, _, _ = normalize_circularity(circularity)
    if norm <= 0.5:
        return _interpolate_bgr(_RED_BRG, _MID_BRG, norm / 0.5)
    return _interpolate_bgr(_MID_BRG, _GREEN_BRG, (norm - 0.5) / 0.5)


def build_heatmap_output_paths(image_stem: str, output_root: str | Path) -> dict[str, Path]:
    visualizations_dir = Path(output_root) / "visualizations"
    return {
        "overlay_path": visualizations_dir / f"{image_stem}{_OVERLAY_SUFFIX}",
        "mask_path": visualizations_dir / f"{image_stem}{_MASK_SUFFIX}",
        "legend_path": visualizations_dir / f"{image_stem}{_LEGEND_SUFFIX}",
        "meta_path": visualizations_dir / f"{image_stem}{_META_SUFFIX}",
    }


def _build_legend_image(width: int = 340, height: int = 130) -> np.ndarray:
    legend = np.zeros((height, width, 3), dtype=np.uint8)
    legend[:, :] = (30, 30, 30)
    overlay = legend.copy()
    cv2.rectangle(overlay, (0, 0), (width - 1, height - 1), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.45, legend, 0.55, 0, dst=legend)

    bar_x1, bar_x2 = 16, width - 16
    bar_y1, bar_y2 = 22, 44
    for x in range(bar_x1, bar_x2 + 1):
        t = (x - bar_x1) / max(bar_x2 - bar_x1, 1)
        legend[bar_y1:bar_y2, x] = circularity_to_bgr(t)

    cv2.rectangle(legend, (bar_x1, bar_y1), (bar_x2, bar_y2), (255, 255, 255), thickness=1)
    cv2.putText(
        legend,
        "Circularity [0.00 -> 1.00]",
        (16, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        legend,
        "Fractured (Low Circularity)",
        (16, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        legend,
        "Intact (High Circularity)",
        (160, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(legend, (16, 84), (36, 104), _REJECTED_BRG, thickness=-1)
    cv2.rectangle(legend, (16, 84), (36, 104), (255, 255, 255), thickness=1)
    cv2.putText(legend, "Rejected/Filtered", (46, 99), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return legend


def _to_contour(points: Any) -> np.ndarray | None:
    if points is None:
        return None
    contour = np.asarray(points, dtype=np.int32)
    if contour.ndim != 2 or contour.shape[0] < 3 or contour.shape[1] != 2:
        return None
    return contour.reshape((-1, 1, 2))


def _coerce_optional_int(value: Any, default: int = 0) -> int:
    if value is None or pd.isna(value):
        return int(default)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return int(default)
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return int(default)
        try:
            return int(float(stripped))
        except ValueError:
            return int(default)
    return int(default)


def generate_circularity_heatmap_artifacts(
    image: np.ndarray,
    annotated_particles: pd.DataFrame,
    image_stem: str,
    output_root: str | Path,
    alpha: float = 0.45,
) -> dict[str, Any]:
    paths = build_heatmap_output_paths(image_stem=image_stem, output_root=output_root)
    paths["overlay_path"].parent.mkdir(parents=True, exist_ok=True)

    base_bgr = _to_bgr(image)
    mask_color = np.zeros_like(base_bgr, dtype=np.uint8)
    boundary = np.zeros_like(base_bgr, dtype=np.uint8)

    rows = annotated_particles.copy()
    sort_key = "particle_id" if "particle_id" in rows.columns else None
    if sort_key is not None:
        rows = rows.sort_values(by=sort_key, kind="stable")

    clipped_count = 0
    invalid_metric_count = 0
    valid_particles = 0
    rejected_particles = 0

    for row in rows.itertuples(index=False):
        contour = _to_contour(getattr(row, "contour_points", None))
        if contour is None:
            continue

        status = str(getattr(row, "validation_status", "valid"))
        if status == "rejected":
            color = _REJECTED_BRG
            rejected_particles += 1
        else:
            norm, clipped, invalid_metric = normalize_circularity(getattr(row, "circularity", 0.0))
            clipped_count += int(clipped)
            invalid_metric_count += int(invalid_metric)
            color = circularity_to_bgr(norm)
            valid_particles += 1

        cv2.fillPoly(mask_color, [contour], color=color)
        cv2.polylines(boundary, [contour], isClosed=True, color=(255, 255, 255), thickness=1)

    overlay = cv2.addWeighted(base_bgr, 1.0 - alpha, mask_color, alpha, 0.0)
    overlay = np.maximum(overlay, boundary)
    legend = _build_legend_image()
    mask_with_boundary = np.maximum(mask_color, boundary)

    cv2.imwrite(str(paths["overlay_path"]), overlay)
    cv2.imwrite(str(paths["mask_path"]), mask_with_boundary)
    cv2.imwrite(str(paths["legend_path"]), legend)

    meta = {
        "image_stem": image_stem,
        "normalization": {
            "strategy": "fixed_domain",
            "domain": [0.0, 1.0],
            "formula": "norm = clip((circularity - 0.0) / (1.0 - 0.0), 0.0, 1.0)",
        },
        "colormap": {
            "lut_name": "circularity_red_to_green_v1",
            "endpoints": {
                "fractured_low_hex": "#D73027",
                "neutral_mid_hex": "#FEE08B",
                "intact_high_hex": "#1A9850",
            },
            "interpolation": "linear_rgb",
        },
        "overlay": {
            "alpha": alpha,
            "boundary_color_hex": "#FFFFFF",
            "boundary_thickness_px": 1,
            "draw_order": "fill first, boundary second",
        },
        "rejected_style": {
            "fill_hex": "#808080",
            "fill_bgr": [128, 128, 128],
        },
        "valid_particles": valid_particles,
        "rejected_particles": rejected_particles,
        "clipped_count": clipped_count,
        "invalid_metric_count": invalid_metric_count,
    }

    paths["meta_path"].write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "image_stem": image_stem,
        "overlay_path": paths["overlay_path"],
        "mask_path": paths["mask_path"],
        "legend_path": paths["legend_path"],
        "meta_path": paths["meta_path"],
        "valid_particles": valid_particles,
        "rejected_particles": rejected_particles,
        "clipped_count": clipped_count,
    }


def write_heatmap_batch_manifest(entries: list[dict[str, Any]], output_root: str | Path) -> Path:
    manifest_path = Path(output_root) / "visualizations" / _BATCH_MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_entries = []
    for item in entries:
        normalized_entries.append(
            {
                "image_stem": item["image_stem"],
                "overlay_path": str(item["overlay_path"]),
                "mask_path": str(item["mask_path"]),
                "legend_path": str(item["legend_path"]),
                "meta_path": str(item["meta_path"]),
                "valid_particles": _coerce_optional_int(item.get("valid_particles")),
                "rejected_particles": _coerce_optional_int(item.get("rejected_particles")),
                "clipped_count": _coerce_optional_int(item.get("clipped_count")),
            }
        )
    manifest = {"files": normalized_entries}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


__all__ = [
    "build_heatmap_output_paths",
    "circularity_to_bgr",
    "generate_circularity_heatmap_artifacts",
    "normalize_circularity",
    "write_heatmap_batch_manifest",
]
