from __future__ import annotations

import streamlit as st

from demo import config
from demo.common.control import ControlClient
from demo.pc import tones
from demo.pc.state import AppState


def _send_command(
    state: AppState, client: ControlClient, label: str, command: str
) -> None:
    try:
        response = client.send(command)
    except ConnectionError as exc:
        state.set_command_result(command, error=str(exc))
        st.error(f"{label} command failed: {exc}")
        return

    state.set_command_result(command, response=response)
    if response.get("ack") == command:
        st.success(f"{label} command acknowledged")
    else:
        st.warning(f"{label} command response: {response}")


def render(state: AppState, client: ControlClient) -> None:
    container = st.sidebar
    container.subheader("Control panel")

    if container.button("Landing", type="primary", width="stretch"):
        _send_command(state, client, "Landing", "LANDING")

    if container.button("Strike", width="stretch"):
        _send_command(state, client, "Strike", "STRIKE")

    status = state.status()
    if status["last_error"]:
        container.error(f"Last command error: {status['last_error']}")
    elif status["last_command"]:
        container.caption(
            f"Last command: {status['last_command']} | response: {status['last_response']}"
        )
    else:
        container.caption(f"Control target: {config.RBP_HOST}:{config.CONTROL_PORT}")

    container.caption("Temporary demo controls")
    temporary_buttons = (
        ("Wake up", "wake_up"),
        ("Confirmation", "confirmation"),
        ("Capture", "capture"),
    )
    for label, tone_name in temporary_buttons:
        if container.button(label, width="stretch") and not tones.play_tone(tone_name):
            container.audio(
                tones.tone_wav_bytes(tone_name),
                format="audio/wav",
                autoplay=True,
            )
