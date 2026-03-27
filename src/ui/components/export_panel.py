from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import streamlit as st


def _build_heatmap_zip(artifact_paths: dict) -> bytes | None:
    heatmap_files = artifact_paths.get("heatmap_paths", [])
    existing = [Path(path_str) for path_str in heatmap_files if Path(path_str).exists()]

    if not existing:
        return None

    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        for file_path in existing:
            archive.write(file_path, arcname=file_path.name)

    buffer.seek(0)
    return buffer.getvalue()


def render_export_panel(artifact_paths: dict) -> None:
    csv_path = artifact_paths.get("csv_path")
    manifest_path = artifact_paths.get("manifest_path")

    has_any = bool(csv_path or manifest_path or artifact_paths.get("heatmap_paths"))
    if not has_any:
        st.info("No artifacts available for export.")
        return

    if csv_path and Path(csv_path).exists():
        csv_bytes = Path(csv_path).read_bytes()
        st.download_button(
            label="⬇ Download analisis_gold_standard.csv",
            data=csv_bytes,
            file_name="analisis_gold_standard.csv",
            mime="text/csv",
            use_container_width=True,
        )

    heatmap_zip = _build_heatmap_zip(artifact_paths)
    if heatmap_zip is not None and len(artifact_paths.get("heatmap_paths", [])) >= 2:
        st.download_button(
            label="⬇ Download nmc811_heatmaps.zip",
            data=heatmap_zip,
            file_name="nmc811_heatmaps.zip",
            mime="application/zip",
            use_container_width=True,
        )

    if manifest_path and Path(manifest_path).exists():
        manifest_bytes = Path(manifest_path).read_bytes()
        st.download_button(
            label="⬇ Download batch manifest JSON",
            data=manifest_bytes,
            file_name=Path(manifest_path).name,
            mime="application/json",
            use_container_width=True,
        )
