from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

CSV_EXPORT_FILENAME = "analisis_gold_standard.csv"

CSV_EXPORT_COLUMNS = [
    "image_id",
    "particle_id",
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
    "is_edge_particle",
    "validation_status",
    "rejection_reason",
]

_NUMERIC_COLUMNS = [
    "area_px",
    "area_um2",
    "perimeter_px",
    "circularity",
    "equivalent_diameter_px",
    "equivalent_diameter_um",
    "aspect_ratio",
    "area",
    "perimeter",
]

_INTEGER_COLUMNS = ["particle_id"]
_BOOLEAN_COLUMNS = ["is_edge_particle"]
_TEXT_COLUMNS = ["image_id", "validation_status", "rejection_reason", "metric_error_code", "metric_error_message"]


def _empty_export_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CSV_EXPORT_COLUMNS)


def _normalize_row_group(image_id: str, frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy(deep=True)
    normalized["image_id"] = image_id

    if "particle_id" not in normalized.columns and "id" in normalized.columns:
        normalized["particle_id"] = normalized["id"]

    for column in CSV_EXPORT_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA

    normalized = normalized.loc[:, CSV_EXPORT_COLUMNS]

    for column in _NUMERIC_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    for column in _INTEGER_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("Int64")
    for column in _BOOLEAN_COLUMNS:
        normalized[column] = normalized[column].map(lambda value: bool(value) if pd.notna(value) else False).astype(bool)
    for column in _TEXT_COLUMNS:
        normalized[column] = normalized[column].astype("string")

    return normalized


def consolidate_filtered_metrics_batch(
    per_image_metrics: Iterable[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    normalized_frames: list[pd.DataFrame] = []

    for image_id, frame in per_image_metrics:
        normalized_frames.append(_normalize_row_group(image_id=image_id, frame=frame))

    if not normalized_frames:
        return _empty_export_frame()

    merged = pd.concat(normalized_frames, axis=0, ignore_index=True, sort=False)
    merged = merged.sort_values(by=["image_id", "particle_id"], kind="stable", na_position="last").reset_index(drop=True)
    return merged.loc[:, CSV_EXPORT_COLUMNS]


def write_analisis_gold_standard_csv(
    merged_metrics: pd.DataFrame,
    output_root: str | Path,
    filename: str = CSV_EXPORT_FILENAME,
) -> Path:
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_frame = merged_metrics.copy(deep=True)
    for column in CSV_EXPORT_COLUMNS:
        if column not in export_frame.columns:
            export_frame[column] = pd.NA
    export_frame = export_frame.loc[:, CSV_EXPORT_COLUMNS]
    export_frame = export_frame.sort_values(by=["image_id", "particle_id"], kind="stable", na_position="last").reset_index(drop=True)

    output_path = output_dir / filename
    export_frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


__all__ = [
    "CSV_EXPORT_COLUMNS",
    "CSV_EXPORT_FILENAME",
    "consolidate_filtered_metrics_batch",
    "write_analisis_gold_standard_csv",
]

