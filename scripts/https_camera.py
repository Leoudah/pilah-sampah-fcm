"""Receive camera snapshots for the hosted Streamlit app."""

import base64
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image
from streamlit.components.v1 import declare_component


CAMERA_KEY = "https_camera_capture"
camera_capture = declare_component("https_camera", path=str(Path(__file__).with_name("https_camera")))


def camera_event():
    event = st.session_state.get(CAMERA_KEY)
    ack = event.get("id") if isinstance(event, dict) else None
    value = camera_capture(key=CAMERA_KEY, ack=ack, default=None)
    return value if isinstance(value, dict) else {}


def decode_camera_frame(payload):
    if not isinstance(payload, str) or len(payload) > 400_000:
        return None
    try:
        jpeg = base64.b64decode(payload, validate=True)
        with Image.open(BytesIO(jpeg), formats=["JPEG"]) as incoming:
            if max(incoming.size) <= 640:
                return incoming.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    return None
