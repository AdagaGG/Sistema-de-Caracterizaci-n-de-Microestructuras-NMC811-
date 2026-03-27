from __future__ import annotations

import streamlit as st


def render_upload_panel() -> list[dict] | None:
    files = st.file_uploader(
        "Upload NMC811 TIFF images",
        type=["tif", "tiff"],
        accept_multiple_files=True,
        help="Accepts 16-bit grayscale TIFF files. Multiple files processed as batch.",
    )

    if not files:
        return None

    uploaded_payloads: list[dict] = []
    for uploaded in files:
        raw_bytes = uploaded.getvalue()
        size_kb = len(raw_bytes) / 1024.0
        status = "✓" if len(raw_bytes) > 0 else "✗"
        st.write(f"{status} {uploaded.name} — {size_kb:.2f} KB")

        uploaded_payloads.append(
            {
                "filename": uploaded.name,
                "bytes": raw_bytes,
            }
        )

    st.caption(f"{len(uploaded_payloads)} files ready for processing")
    return uploaded_payloads
