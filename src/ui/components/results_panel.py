from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


def _pick_first(payload: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return default


def _read_json(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_name(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_candidates = {_normalize_name(name) for name in candidates}
    for column in frame.columns:
        if _normalize_name(str(column)) in normalized_candidates:
            return str(column)
    return None


def _coerce_numeric_series(frame: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    column = _find_column(frame, candidates)
    if column is None:
        return None
    return pd.to_numeric(frame[column], errors="coerce")


def _extract_image_stem_from_heatmap_path(heatmap_path: str) -> str:
    name = Path(heatmap_path).name
    return name.split("__heatmap", maxsplit=1)[0]


def _csv_summary_for_heatmap(frame: pd.DataFrame, heatmap_path: str) -> tuple[int | None, float | None, float | None]:
    if frame.empty:
        return None, None, None

    scoped = frame
    image_stem = _extract_image_stem_from_heatmap_path(heatmap_path)
    image_column = _find_column(frame, ["image_id", "image", "image_stem", "image_name", "Image ID"])
    if image_column is not None:
        scoped = frame.loc[frame[image_column].astype(str) == image_stem]
        if scoped.empty:
            scoped = frame

    particle_count = int(len(scoped))
    circularity_values = _coerce_numeric_series(scoped, ["circularity", "Circularity", "circularidad"])
    area_values = _coerce_numeric_series(scoped, ["area_um2", "Area_um2", "area (um2)", "area_um^2"])

    mean_circularity = float(circularity_values.mean()) if circularity_values is not None and circularity_values.notna().any() else None
    mean_area_um2 = float(area_values.mean()) if area_values is not None and area_values.notna().any() else None
    return particle_count, mean_circularity, mean_area_um2


def _best_effort_meta(meta: dict[str, Any]) -> tuple[int | None, float | None, float | None]:
    particle_count = _pick_first(meta, ["particle_count", "valid_particles", "total_particles", "num_particles"])
    mean_circularity = _pick_first(meta, ["mean_circularity", "avg_circularity", "circularity_mean"])
    mean_area_um2 = _pick_first(meta, ["mean_area_um2", "avg_area_um2", "area_um2_mean"])

    return (
        int(particle_count) if particle_count is not None else None,
        float(mean_circularity) if mean_circularity is not None else None,
        float(mean_area_um2) if mean_area_um2 is not None else None,
    )


def render_results_panel(batch_result: dict, artifact_paths: dict) -> None:
    total_images = int(_pick_first(batch_result, ["total_images", "images_total"], 0))
    completed_count = int(_pick_first(batch_result, ["completed_count", "images_succeeded"], 0))
    failed_count = int(_pick_first(batch_result, ["failed_count", "images_failed"], 0))
    results = _pick_first(batch_result, ["results", "images"], [])
    status_value = str(_pick_first(batch_result, ["status"], "unknown"))

    col_total, col_ok, col_failed, col_status = st.columns(4)
    col_total.metric("Images Processed", total_images)
    col_ok.metric("Completed", completed_count)
    col_failed.metric("Failed", failed_count)
    col_status.metric("Status", status_value)

    st.markdown("### Heatmaps")
    heatmap_paths = artifact_paths.get("heatmap_paths", [])
    meta_paths = artifact_paths.get("meta_paths", [])
    csv_path = artifact_paths.get("csv_path")
    csv_frame = pd.read_csv(csv_path) if csv_path and Path(csv_path).exists() else pd.DataFrame()

    if not heatmap_paths:
        st.info("No heatmaps generated yet.")
    else:
        for index, heatmap_path in enumerate(heatmap_paths):
            st.image(heatmap_path, caption=Path(heatmap_path).name, use_container_width=True)

            meta = _read_json(meta_paths[index]) if index < len(meta_paths) else {}
            meta_particle_count, meta_mean_circularity, meta_mean_area_um2 = _best_effort_meta(meta)
            csv_particle_count, csv_mean_circularity, csv_mean_area_um2 = _csv_summary_for_heatmap(csv_frame, heatmap_path)

            particle_count = meta_particle_count if meta_particle_count is not None else csv_particle_count
            mean_circularity = meta_mean_circularity if meta_mean_circularity is not None else csv_mean_circularity
            mean_area_um2 = meta_mean_area_um2 if meta_mean_area_um2 is not None else csv_mean_area_um2

            left_col, mid_col, right_col = st.columns(3)
            left_col.metric("particle_count", particle_count if particle_count is not None else "N/A")
            mid_col.metric(
                "mean_circularity",
                f"{mean_circularity:.4f}" if mean_circularity is not None else "N/A",
            )
            right_col.metric(
                "mean_area_um2",
                f"{mean_area_um2:.4f}" if mean_area_um2 is not None else "N/A",
            )

    st.markdown("### CSV Results")
    if not csv_frame.empty:
        frame = csv_frame
        table_columns = ["image_id", "particle_id", "area_um2", "circularity", "perimeter_um"]
        available_columns = [column for column in table_columns if column in frame.columns]
        preview_frame = frame.loc[:, available_columns] if available_columns else frame
        st.dataframe(preview_frame, use_container_width=True)
        st.caption(f"{len(frame)} valid particles detected")
    else:
        st.info("CSV artifact is not available.")

    if failed_count > 0:
        with st.expander("⚠️ Processing Errors", expanded=False):
            for item in results:
                if str(item.get("status", "")) != "failed":
                    continue
                image_id = item.get("image_id", "unknown")
                error = item.get("error") or {}
                st.write(
                    {
                        "image_id": image_id,
                        "code": error.get("code", "ERR_UNKNOWN"),
                        "message": error.get("message", "Unknown error"),
                        "context": error.get("context", {}),
                    }
                )
