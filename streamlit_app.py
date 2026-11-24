"""Classify photos and live camera frames with the scenario 1a model."""

from collections import deque
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import perf_counter
from urllib.parse import urlsplit

import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from scripts.https_camera import camera_event, decode_camera_frame
from scripts.predict_lightweight import load_model, prepare_image


MODEL_PATH = Path(__file__).resolve().parent / "artifacts/lightweight_fcm/mobilenet_fcm_knn_1a.onnx"
PREDICTIONS_PER_SECOND = 5
NAMA = {
    "cardboard": "Kardus",
    "glass": "Kaca",
    "metal": "Logam",
    "paper": "Kertas",
    "plastic": "Plastik",
    "trash": "Sampah lainnya",
}


@st.cache_resource(show_spinner=False)
def model_session(path, modified_ns):
    return load_model(path)


def predict(photo, model, config):
    tensor = prepare_image(photo, config)
    outputs = model.run(None, {model.get_inputs()[0].name: tensor})
    fuzzy = outputs[1][0] if len(outputs) > 1 else None
    return outputs[0][0], fuzzy


def review_photo(photo, model, config):
    bright = ImageEnhance.Brightness(photo).enhance(1.25)
    rotated = [bright, *(bright.transpose(angle) for angle in (
        Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270,
    ))]
    batch = np.concatenate([prepare_image(view, config) for view in rotated])
    outputs = model.run(None, {model.get_inputs()[0].name: batch})
    fuzzy = outputs[1].mean(axis=0) if len(outputs) > 1 else None
    return outputs[0].mean(axis=0), fuzzy


def render_prediction(labels, scores, fuzzy, elapsed, rate=None, reviewed=False):
    ranked = np.argsort(-scores, kind="stable")[:3]
    st.html(f'<p class="hasil-kelas">{NAMA[labels[ranked[0]]]}</p>')
    for i in ranked:
        st.progress(float(scores[i]), text=f"{NAMA[labels[i]]} · {scores[i]:.1%}")
    timing = f"Pemrosesan CPU {elapsed:.1f} ms"
    if rate is not None:
        timing += f" · {rate:.1f} prediksi/dtk"
    st.caption(f"Skor KNN · {timing}")

    with st.expander("Lihat proses FCM"):
        if fuzzy is None:
            st.info("Keluaran keanggotaan belum tersedia pada model ini.")
            return
        if reviewed:
            st.write("Panel ini menampilkan rata-rata keanggotaan FCM dari empat orientasi foto.")
        else:
            st.write(
                "FCM menghitung keanggotaan gambar terhadap enam klaster. "
                "KNN memakai 50 komponen PCA dan enam nilai keanggotaan untuk memperkirakan kelas sampah."
            )
        for cluster, score in enumerate(fuzzy, 1):
            st.progress(float(score), text=f"Klaster {cluster} · {score:.1%}")
        st.caption(f"Total keanggotaan FCM {float(fuzzy.sum()):.0%}")


class LivePrediction:
    def __init__(self, model, config, labels, model_updated_ns):
        self.model, self.config, self.labels = model, config, labels
        self.model_updated_ns = model_updated_ns
        self.lock = Lock()
        self.times = deque(maxlen=8)
        self.scores = None
        self.fuzzy = None
        self.elapsed = 0.0
        self.next_frame = 0.0
        self.generation = 0
        self.last_capture = None

    def receive(self, frame):
        start = perf_counter()
        with self.lock:
            if start < self.next_frame:
                return frame
            self.next_frame = start + 1 / PREDICTIONS_PER_SECOND
            generation = self.generation

        photo = Image.fromarray(frame.to_ndarray(format="rgb24"))
        self.update(photo, start, generation)
        return frame

    def update(self, photo, start, generation):
        scores, fuzzy = predict(photo, self.model, self.config)
        complete = perf_counter()
        with self.lock:
            if generation == self.generation:
                self.scores = scores if self.scores is None else 0.6 * scores + 0.4 * self.scores
                self.fuzzy = fuzzy
                self.elapsed = (complete - start) * 1000
                self.times.append(start)

    def receive_capture(self, event):
        capture = event.get("id")
        if not isinstance(capture, str) or capture == self.last_capture:
            return
        self.last_capture = capture
        if event.get("kind") == "stop":
            self.clear()
        elif event.get("kind") == "frame":
            photo = decode_camera_frame(event.get("jpeg"))
            if photo is not None:
                self.update(photo, perf_counter(), self.generation)

    def snapshot(self):
        with self.lock:
            if not self.times or perf_counter() - self.times[-1] > 1.5:
                return None
            rate = None
            if len(self.times) > 1:
                rate = (len(self.times) - 1) / (self.times[-1] - self.times[0])
            return self.scores, self.fuzzy, self.elapsed, rate

    def clear(self):
        with self.lock:
            self.generation += 1
            self.scores = None
            self.fuzzy = None
            self.next_frame = 0.0
            self.times.clear()


@st.fragment(run_every=0.2)
def render_live(stream):
    current = stream.snapshot()
    if current is None:
        st.info("Menunggu gambar dari kamera.")
    else:
        render_prediction(stream.labels, *current)


st.set_page_config(page_title="Pilah Sampah", page_icon="♻️", layout="wide")
st.html("""
<style>
  .stMainBlockContainer {max-width: 1100px; padding-top: 4rem;}
  .st-key-photo_panel, .st-key-result_panel {
    background: white;
    border: 1px solid #d8e5dd;
    border-radius: 18px;
    padding: 1.25rem;
  }
  .st-key-photo_panel [data-testid="stImage"] img {
    border-radius: 12px;
  }
  .hasil-kelas {
    margin: .1rem 0 .45rem;
    color: #1b5e47;
    font-size: clamp(1.8rem, 5vw, 2.5rem);
    line-height: 1.15;
    font-weight: 700;
    overflow-wrap: anywhere;
  }
  @media (max-width: 600px) {
    .stMainBlockContainer {padding-top: 3rem;}
    .st-key-photo_panel, .st-key-result_panel {padding: 1rem;}
  }
</style>
""")

st.title("Pilah sampah")
st.write("Unggah foto atau arahkan kamera ke satu benda untuk melihat perkiraan jenis materialnya.")
st.caption("Skenario 1a · MobileNetV3-Small + PCA 50 + FCM + KNN")

photo_column, result_column = st.columns([1.15, 1], gap="medium")
photo = None
camera = None
online_camera = False
model_ready = MODEL_PATH.is_file()
with photo_column, st.container(key="photo_panel"):
    st.subheader("Foto atau kamera")
    source = st.segmented_control(
        "Sumber gambar", ["Unggah foto", "Ambil foto", "Kamera realtime"],
        default="Unggah foto", required=True,
    )
    if source == "Kamera realtime":
        if model_ready:
            model_updated_ns = MODEL_PATH.stat().st_mtime_ns
            live = st.session_state.get("live_prediction")
            if getattr(live, "model_updated_ns", None) != model_updated_ns or not hasattr(live, "receive_capture"):
                model, config, labels = model_session(MODEL_PATH, model_updated_ns)
                live = LivePrediction(model, config, labels, model_updated_ns)
                st.session_state.live_prediction = live
            address = urlsplit(st.context.url or "")
            online_camera = address.scheme == "https" and address.hostname not in ("localhost", "127.0.0.1", "::1")
            if online_camera:
                live.receive_capture(camera_event())
            else:
                camera = webrtc_streamer(
                    key="kamera-realtime",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration={"iceServers": []},
                    media_stream_constraints={
                        "audio": False,
                        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 15, "max": 20}},
                    },
                    video_frame_callback=live.receive,
                    on_video_ended=live.clear,
                    video_html_attrs={
                        "autoPlay": True, "controls": False, "muted": True, "playsInline": True,
                        "style": {"width": "100%", "maxHeight": "340px", "borderRadius": "12px"},
                    },
                )
            st.caption("Letakkan satu benda di tengah. Untuk botol, beri jarak dari tepi agar leher dan dasar terlihat utuh.")
    else:
        st.session_state.pop("live_prediction", None)
        if source == "Ambil foto":
            incoming = st.camera_input("Ambil foto")
        else:
            incoming = st.file_uploader("Pilih foto", type=["jpg", "jpeg", "png", "webp"])
        if incoming is not None:
            try:
                with Image.open(BytesIO(incoming.getvalue())) as original:
                    photo = ImageOps.exif_transpose(original).convert("RGB")
            except (OSError, ValueError, Image.DecompressionBombError):
                st.error("Foto tidak bisa dibaca. Coba file JPG, PNG, atau WebP yang bisa dibuka.")
            else:
                preview = photo.copy()
                preview.thumbnail((640, 340))
                st.image(preview, caption=incoming.name)

with result_column, st.container(key="result_panel"):
    st.subheader("Hasil prediksi")
    if not model_ready:
        st.error("Model ONNX skenario 1a belum tersedia. Ekspor model lalu muat ulang halaman ini.")
    elif online_camera:
        current = live.snapshot()
        if current is None:
            st.info("Tekan Start dan izinkan akses kamera untuk melihat prediksi.")
        else:
            render_prediction(live.labels, *current)
    elif camera is not None:
        if camera.state.playing:
            render_live(live)
        elif camera.state.signalling:
            st.info("Menghubungkan kamera.")
        else:
            st.info("Tekan Start dan izinkan akses kamera untuk melihat prediksi.")
    elif photo is None:
        st.info("Hasil muncul setelah foto siap.")
    else:
        model, config, labels = model_session(MODEL_PATH, MODEL_PATH.stat().st_mtime_ns)
        reviewed = st.toggle("Cek ulang foto sulit", help="Coba empat orientasi dengan foto sedikit lebih terang.")
        start = perf_counter()
        scores, fuzzy = predict(photo, model, config)
        if reviewed:
            first = int(scores.argmax())
            st.caption(f"Prediksi awal · {NAMA[labels[first]]} {scores[first]:.1%}")
            scores, fuzzy = review_photo(photo, model, config)
        render_prediction(labels, scores, fuzzy, (perf_counter() - start) * 1000, reviewed=reviewed)
