from __future__ import annotations

import streamlit as st


def render_params_panel() -> dict:
    with st.expander("⚙️ Pipeline Parameters", expanded=False):
        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("**PREPROCESSING**")
            clahe_clip_limit = st.slider(
                "CLAHE Clip Limit",
                min_value=1.0,
                max_value=8.0,
                value=2.0,
                step=0.5,
            )
            clahe_tile_grid_size = st.slider(
                "Grid Size",
                min_value=4,
                max_value=16,
                value=8,
                step=2,
            )

            st.markdown("**VALIDATION FILTERS**")
            min_area_um2 = st.number_input("Min Area (µm²)", min_value=0.0, value=5.0)
            max_area_um2 = st.number_input("Max Area (µm²)", min_value=0.0, value=500.0)
            min_circularity = st.slider(
                "Min Circularity (0=any, 1=perfect circle)",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
            )

        with right_col:
            st.markdown("**RUNTIME**")
            checkpoint_path = st.text_input("Checkpoint Path", value="weights/mobile_sam.pt")
            device = st.selectbox("Device", options=["cuda:0", "cuda:1", "cpu"], index=0)

    return {
        "clahe_clip_limit": float(clahe_clip_limit),
        "clahe_tile_grid_size": int(clahe_tile_grid_size),
        "min_area_um2": float(min_area_um2),
        "max_area_um2": float(max_area_um2),
        "min_circularity": float(min_circularity),
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
    }
