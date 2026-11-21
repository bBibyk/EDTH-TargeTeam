from __future__ import annotations

import streamlit as st

from demo.pc.state import AppState


def render(state: AppState) -> None:
    st.subheader("Video")
    frame = state.get_frame()
    left, _, right = st.columns([5, 1, 4])
    with left:
        if frame is None:
            st.info("Waiting for RTP video stream...")
        else:
            st.image(frame, channels="BGR", width="stretch")
    with right, st.container(border=True):
        st.caption("Model info here")
