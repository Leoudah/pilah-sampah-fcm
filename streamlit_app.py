"""Classify photos and live camera frames with the selected ONNX model."""

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


MODEL_DIR = Path(__file__).resolve().parent / "artifacts/lightweight_fcm"
MODEL_CHOICES = {
    "FCM + KNN tanpa LDA": {
        "file": "mobilenet_fcm_knn_1a.onnx",
        "caption": "Skenario 1a · MobileNetV3-Small + PCA 50 + FCM + KNN",
        "classifier": "KNN",
        "fuzzy_by_class": False,
        "fuzzy_caption": "FCM menghitung keanggotaan terhadap enam klaster menggunakan 50 komponen PCA.",
    },
    "LDA + FCM + KNN": {
        "file": "mobilenet_lda5_fcm_knn.onnx",
        "caption": "MobileNetV3-Small · PCA 50 dan LDA 5 + FCM · KNN",
        "classifier": "KNN",
        "fuzzy_by_class": False,
        # Material dominan dan jumlah citra pada klaster terkuat di data latih.
        "cluster_profiles": (
            ("glass", 715, 753),
            ("cardboard", 757, 866),
            ("metal", 755, 844),
            ("paper", 709, 759),
            ("trash", 736, 752),
            ("plastic", 705, 826),
        ),
        "fuzzy_caption": "FCM menghitung keanggotaan terhadap enam klaster menggunakan lima komponen LDA.",
    },
    "FCM + MLP 64": {
        "file": "mobilenet_fcm_mlp64.onnx",
        "caption": "MobileNetV3-Small + FCM + MLP 64 neuron",
        "classifier": "MLP",
        "fuzzy_by_class": True,
        "fuzzy_caption": "Panel ini menampilkan keanggotaan FCM yang telah dikelompokkan menurut material.",
    },
}
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


def render_prediction(labels, scores, fuzzy, elapsed, rate=None, *, preset, reviewed=False):
    ranked = np.argsort(-scores, kind="stable")[:3]
    st.html(f'<p class="hasil-kelas">{NAMA[labels[ranked[0]]]}</p>')
    for i in ranked:
        st.progress(float(scores[i]), text=f"{NAMA[labels[i]]} · {scores[i]:.1%}")
    timing = f"Pemrosesan CPU {elapsed:.1f} ms"
    if rate is not None:
        timing += f" · {rate:.1f} prediksi/dtk"
    st.caption(f"Skor {preset['classifier']} · {timing}")

    with st.expander("Lihat proses FCM"):
        if fuzzy is None:
            st.info("Keluaran keanggotaan belum tersedia pada model ini.")
            return
        if reviewed:
            st.write("Panel ini menampilkan rata-rata keanggotaan FCM dari empat orientasi foto.")
        else:
            st.write(preset["fuzzy_caption"])
        profiles = preset.get("cluster_profiles")
        if preset["fuzzy_by_class"]:
            fuzzy_names = [NAMA[label] for label in labels]
        elif profiles and len(profiles) == len(fuzzy):
            fuzzy_names = [
                f"Klaster {index} · dominan {NAMA[material]} ({count}/{total} citra latih klaster)"
                for index, (material, count, total) in enumerate(profiles, 1)
            ]
            st.caption(
                "Material dominan dihitung dari label citra latih pada klaster terkuatnya. "
                "Persentase pada tiap bar menunjukkan keanggotaan foto saat ini. Kelas prediksi ditentukan KNN."
            )
        else:
            fuzzy_names = [f"Klaster {index}" for index in range(1, len(fuzzy) + 1)]
            st.caption(
                "Pada model PCA, keanggotaan keenam klaster hampir merata. "
                "Pilih LDA + FCM + KNN untuk melihat material dominan tiap klaster."
            )
        for name, score in zip(fuzzy_names, fuzzy):
            st.progress(float(score), text=f"{name} · {score:.1%}")
        st.caption(f"Total keanggotaan FCM {float(fuzzy.sum()):.0%}")


class LivePrediction:
    def __init__(self, model, config, labels, model_revision):
        self.model, self.config, self.labels = model, config, labels
        self.model_revision = model_revision
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
        with self.lock:
            if generation != self.generation:
                return
            model, config = self.model, self.config
        scores, fuzzy = predict(photo, model, config)
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

    def set_model(self, model, config, labels, model_revision):
        with self.lock:
            self.model, self.config, self.labels = model, config, labels
            self.model_revision = model_revision
            self._reset_scores()

    def clear(self):
        with self.lock:
            self._reset_scores()

    def _reset_scores(self):
        self.generation += 1
        self.scores = None
        self.fuzzy = None
        self.next_frame = 0.0
        self.times.clear()


@st.fragment(run_every=0.2)
def render_live(stream, preset):
    current = stream.snapshot()
    if current is None:
        st.info("Menunggu gambar dari kamera.")
    else:
        render_prediction(stream.labels, *current, preset=preset)


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
selected_model = st.selectbox("Model", list(MODEL_CHOICES))
preset = MODEL_CHOICES[selected_model]
model_path = MODEL_DIR / preset["file"]
st.caption(preset["caption"])

photo_column, result_column = st.columns([1.15, 1], gap="medium")
photo = None
camera = None
online_camera = False
model_ready = model_path.is_file()
with photo_column, st.container(key="photo_panel"):
    st.subheader("Foto atau kamera")
    source = st.segmented_control(
        "Sumber gambar", ["Unggah foto", "Ambil foto", "Kamera realtime"],
        default="Unggah foto", required=True,
    )
    if source == "Kamera realtime":
        if model_ready:
            model_updated_ns = model_path.stat().st_mtime_ns
            revision = (model_path, model_updated_ns)
            live = st.session_state.get("live_prediction")
            if not hasattr(live, "set_model"):
                model, config, labels = model_session(*revision)
                live = LivePrediction(model, config, labels, revision)
                st.session_state.live_prediction = live
            elif live.model_revision != revision:
                model, config, labels = model_session(*revision)
                live.set_model(model, config, labels, revision)
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
        st.error(f"Model ONNX untuk {selected_model} belum tersedia.")
    elif online_camera:
        current = live.snapshot()
        if current is None:
            st.info("Tekan Start dan izinkan akses kamera untuk melihat prediksi.")
        else:
            render_prediction(live.labels, *current, preset=preset)
    elif camera is not None:
        if camera.state.playing:
            render_live(live, preset)
        elif camera.state.signalling:
            st.info("Menghubungkan kamera.")
        else:
            st.info("Tekan Start dan izinkan akses kamera untuk melihat prediksi.")
    elif photo is None:
        st.info("Hasil muncul setelah foto siap.")
    else:
        model, config, labels = model_session(model_path, model_path.stat().st_mtime_ns)
        reviewed = st.toggle("Cek ulang foto sulit", help="Coba empat orientasi dengan foto sedikit lebih terang.")
        start = perf_counter()
        scores, fuzzy = predict(photo, model, config)
        if reviewed:
            first = int(scores.argmax())
            st.caption(f"Prediksi awal · {NAMA[labels[first]]} {scores[first]:.1%}")
            scores, fuzzy = review_photo(photo, model, config)
        render_prediction(labels, scores, fuzzy, (perf_counter() - start) * 1000, preset=preset, reviewed=reviewed)
