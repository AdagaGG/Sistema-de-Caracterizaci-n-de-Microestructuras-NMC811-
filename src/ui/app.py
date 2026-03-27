from __future__ import annotations

import streamlit as st

from src.ui.bridge import run_pipeline_on_uploads
from src.ui.components.export_panel import render_export_panel
from src.ui.components.params_panel import render_params_panel
from src.ui.components.results_panel import render_results_panel
from src.ui.components.upload_panel import render_upload_panel
from src.utils.error_codes import PrdError


def main() -> None:
    st.set_page_config(
        page_title="NMC811 Segmentation",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("NMC811 Particle Segmentation")
    st.caption("MobileSAM · ASTM-aligned geometry metrics · NMC811 cathode analysis")

    with st.sidebar:
        st.session_state["ui_params"] = render_params_panel()
        st.markdown("---")
        st.markdown("### About")
        st.caption("Interactive UI for batch TIFF segmentation and geometry extraction.")

    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.session_state["uploaded_files"] = render_upload_panel()
        run_clicked = st.button(
            "▶ Run Pipeline",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get("uploaded_files") is None,
        )

    if run_clicked:
        files = st.session_state.get("uploaded_files")
        params = st.session_state.get("ui_params", {})

        if not files:
            st.warning("Please upload at least one non-empty TIFF file.")
        else:
            with st.spinner("Processing batch..."):
                try:
                    run_output = run_pipeline_on_uploads(uploaded_files=files, ui_params=params)
                except PrdError as error:
                    st.error(f"Pipeline error [{error.code}]: {error.message}")
                else:
                    st.session_state["batch_result"] = run_output.get("batch_result", {})
                    st.session_state["artifact_paths"] = run_output.get("artifact_paths", {})
                    completed = int(st.session_state["batch_result"].get("completed_count", 0))
                    st.success(f"Batch complete — {completed} images processed")

    batch_result = st.session_state.get("batch_result")
    artifact_paths = st.session_state.get("artifact_paths")

    with right_col:
        if batch_result and artifact_paths:
            render_results_panel(batch_result=batch_result, artifact_paths=artifact_paths)
        else:
            st.info("Upload TIFF files and click Run Pipeline.")

    st.markdown("---")
    if artifact_paths:
        render_export_panel(artifact_paths=artifact_paths)


if __name__ == "__main__":
    main()
