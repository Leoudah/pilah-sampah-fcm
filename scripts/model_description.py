"""Model documentation and evaluation on the application's saved test sets."""

import json
from pathlib import Path

import streamlit as st


ASSETS = Path(__file__).resolve().parents[1] / "artifacts" / "model_description"
MLP = "mobilenet_fcm_mlp64.onnx"
LDA = "mobilenet_lda5_fcm_knn.onnx"
COLOR = ["#e9f4ed", "#145942"]


def heatmap(matrix, rows, columns, *, y_title, x_title, percent, alt):
    cells = []
    for row, counts in zip(rows, matrix, strict=True):
        total = sum(counts)
        for col, count in zip(columns, counts, strict=True):
            fraction = count / total if total else 0
            cells.append({
                "baris": row, "kolom": col, "jumlah": count,
                "nilai": fraction if percent else count,
                "teks": f"{fraction:.1%}" if percent else str(count),
            })
    maximum = 1 if percent else max(cell["nilai"] for cell in cells)
    axes = {
        "x": {
            "field": "kolom", "type": "nominal", "sort": columns, "title": x_title,
            "axis": {"labelAngle": -30, "labelLimit": 110},
        },
        "y": {
            "field": "baris", "type": "nominal", "sort": rows, "title": y_title,
            "axis": {"labelLimit": 115},
        },
    }
    st.vega_lite_chart({
        "data": {"values": cells}, "height": 250,
        "encoding": axes,
        "layer": [
            {
                "mark": {"type": "rect", "stroke": "white", "strokeWidth": 1},
                "encoding": {
                    "color": {"field": "nilai", "type": "quantitative",
                              "scale": {"domain": [0, maximum], "range": COLOR}, "legend": None},
                    "tooltip": [
                        {"field": "baris", "title": y_title},
                        {"field": "kolom", "title": x_title},
                        {"field": "jumlah", "type": "quantitative", "title": "Jumlah foto"},
                        {"field": "teks", "title": "Nilai sel"},
                    ],
                },
            },
            {
                "mark": {"type": "text", "fontSize": 11, "fontWeight": "bold"},
                "encoding": {
                    "text": {"field": "teks"},
                    "color": {"condition": {"test": f"datum.nilai > {maximum * 0.64}", "value": "white"},
                              "value": "#18352b"},
                },
            },
        ],
    }, width="stretch", alt=alt)


def matrix_table(matrix, rows, columns, row_header):
    st.dataframe([
        {row_header: row, **dict(zip(columns, counts, strict=True))}
        for row, counts in zip(rows, matrix, strict=True)
    ], hide_index=True, width="stretch")


def render_architecture(filename):
    with st.expander("Alur dan parameter model", expanded=True):
        st.markdown("**Foto RGB → MobileNetV3-Small pralatih ImageNet → 576 fitur**")
        if filename == MLP:
            st.write("StandardScaler menyiapkan 576 fitur dasar. Cabang FCM memakai PCA 48 dan normalisasi L2.")
            st.write("FCM mempelajari empat prototipe untuk setiap material (24 total, m = 1,35). "
                     "Keanggotaannya dijumlahkan per material dan distandardisasi.")
            st.markdown("**576 fitur standar + 6 skor FCM → MLP dengan 64 neuron tersembunyi → kelas sampah**")
        else:
            st.write("Fitur dasar KNN melalui StandardScaler dan PCA dengan 50 komponen.")
            if filename == LDA:
                st.write("Cabang fuzzy memakai LDA langsung dari 576 fitur MobileNet. Lima komponen LDA masuk ke FCM.")
            else:
                st.write("Cabang fuzzy memakai 50 komponen PCA yang sama sebagai masukan FCM.")
            st.write("FCM memakai enam klaster, m = 2 dan jarak Euclidean.")
            st.markdown("**50 komponen PCA + 6 keanggotaan FCM (α = 1) → KNN cosine (k = 7) → kelas sampah**")


def render_evaluation(data, names, *, uses_v2):
    if uses_v2:
        st.caption("Data uji gabungan V2 tanpa TACO, 1.637 foto. Pembagian ini berbeda dari uji dua model KNN.")
    else:
        st.caption("Data uji Kaggle Zlatan v5, 1.200 foto dengan 200 foto per kelas. Kedua model KNN memakai foto uji yang sama.")

    accuracy, f1 = st.columns(2)
    accuracy.metric("Akurasi", f"{data['accuracy']:.2%}")
    f1.metric("Macro-F1", f"{data['macro']['f1-score']:.4f}")
    st.caption(f"Precision makro {data['macro']['precision']:.4f} · Recall makro {data['macro']['recall']:.4f}")
    st.caption(f"{data['correct']:,} dari {data['test_count']:,} foto diprediksi dengan benar.".replace(",", "."))

    st.subheader("Heatmap prediksi")
    unit = st.radio("Tampilan sel", ("Jumlah foto", "Persentase per kelas asli"), horizontal=True)
    if unit == "Jumlah foto":
        st.caption("Angka menunjukkan jumlah foto. Sel diagonal adalah prediksi benar.")
    else:
        st.caption("Angka menunjukkan persentase dari total foto pada masing-masing kelas asli. Sel diagonal adalah recall per kelas.")
    labels = [names[label] for label in data["class_order"]]
    matrix = data["confusion_matrix"]
    heatmap(matrix, labels, labels, y_title="Kelas asli", x_title="Kelas prediksi",
            percent=unit == "Persentase per kelas asli",
            alt="Confusion matrix enam kelas. Baris menunjukkan kelas asli, kolom menunjukkan kelas prediksi.")
    with st.expander("Lihat rincian per kelas"):
        st.dataframe([
            {"Kelas": names[label], "Precision": f"{row['precision']:.3f}", "Recall": f"{row['recall']:.3f}",
             "F1": f"{row['f1-score']:.3f}", "Foto uji": int(row["support"])}
            for label, row in data["per_class"].items()
        ], hide_index=True, width="stretch")
        st.markdown("**Confusion matrix dalam jumlah foto**")
        matrix_table(matrix, labels, labels, "Kelas asli")


def render_membership_example():
    with st.expander("Coba fungsi keanggotaan pada dua klaster"):
        st.write("Simulasi satu dimensi dengan pusat di posisi 0 dan 3 serta m = 2. Geser posisi contoh untuk melihat perubahan keanggotaan.")
        position = st.slider("Posisi contoh", 0.0, 3.0, 1.0, step=0.1)
        first = (3 - position) ** 2 / (position ** 2 + (3 - position) ** 2)
        left, right = st.columns(2)
        left.metric("Klaster 1", f"{first:.1%}")
        right.metric("Klaster 2", f"{1 - first:.1%}")

        curve = []
        for i in range(61):
            x = i / 20
            mu = (3 - x) ** 2 / (x ** 2 + (3 - x) ** 2)
            curve.extend(({"x": x, "keanggotaan": mu, "klaster": "Klaster 1"},
                          {"x": x, "keanggotaan": 1 - mu, "klaster": "Klaster 2"}))
        st.vega_lite_chart({
            "height": 170, "layer": [
                {"data": {"values": curve}, "mark": {"type": "line", "strokeWidth": 2},
                 "encoding": {
                     "x": {"field": "x", "type": "quantitative", "title": "Posisi contoh", "scale": {"domain": [0, 3]}},
                     "y": {"field": "keanggotaan", "type": "quantitative", "title": "Keanggotaan",
                           "scale": {"domain": [0, 1]}, "axis": {"format": ".0%"}},
                     "color": {"field": "klaster", "type": "nominal", "title": None,
                               "scale": {"range": ["#20694f", "#365d99"]}},
                     "strokeDash": {"field": "klaster", "legend": None},
                 }},
                {"data": {"values": [{"x": position}]}, "mark": {"type": "rule", "color": "#4b504e", "opacity": 0.7},
                 "encoding": {"x": {"field": "x", "type": "quantitative"}}},
            ],
        }, width="stretch", alt="Simulasi dua kurva keanggotaan FCM. Kedua klaster bernilai 50 persen pada posisi 1,5.")
        st.caption("Eksperimen KNN memakai enam klaster. Ilustrasi ini memakai dua pusat agar hubungan dengan jarak mudah terlihat.")


def render_clusters(filename, data, preset, names):
    if filename == MLP:
        st.subheader("Prototipe FCM per material")
        st.write("Model MLP memakai 24 prototipe. Panel prediksi menjumlahkan keanggotaan empat prototipe untuk tiap material.")
    else:
        st.subheader("Posisi citra dan pusat FCM")
        st.metric("Rata-rata keanggotaan tertinggi per foto uji", f"{data['mean_highest_membership']:.2%}")
        st.image(str(ASSETS / "fcm_pca_vs_lda.png"), width="stretch")
        st.caption("Warna titik menunjukkan label asli, dan tanda silang menunjukkan pusat FCM. Grafik menampilkan "
                   "dua sumbu pertama. FCM menghitung jarak di 50 dimensi PCA atau lima dimensi LDA.")
        st.write("Pada PCA, enam pusat hampir berimpit. Pada LDA, pusat lebih tersebar dan nilai keanggotaan "
                 "lebih sering terkonsentrasi pada klaster tertentu.")
    if filename == LDA:
        st.subheader("Komposisi enam klaster LDA")
        st.caption("Setiap foto latih dikelompokkan menurut nilai keanggotaan tertingginya. Angka menunjukkan komposisi "
                   "data latih, sedangkan kelas foto baru dipilih oleh KNN.")
        rows = [f"Klaster {i}" for i in range(1, 7)]
        columns = [names[label] for label in data["class_order"]]
        counts = data["train_cluster_counts"]
        heatmap(counts, rows, columns, y_title="Klaster", x_title="Material pada data latih", percent=True,
                alt="Heatmap proporsi label material di enam klaster FCM. Setiap baris berjumlah 100 persen.")
        with st.expander("Lihat material dominan tiap klaster"):
            st.dataframe([
                {"Klaster": row, "Material dominan": names[material], "Citra": f"{count}/{total}",
                 "Proporsi": f"{count / total:.2%}"}
                for row, (material, count, total) in zip(rows, preset["cluster_profiles"], strict=True)
            ], hide_index=True, width="stretch")
            matrix_table(counts, rows, columns, "Klaster")
    render_membership_example()


def render_description(model_choices, names):
    st.title("Deskripsi model")
    st.write("Lihat cara model bekerja, hasil uji per kelas, dan cara membaca derajat keanggotaan fuzzy.")
    selected = st.selectbox("Model yang dijelaskan", list(model_choices), index=1)
    preset = model_choices[selected]
    filename = preset["file"]
    st.caption(preset["caption"])
    try:
        all_results = json.loads((ASSETS / "evaluation.json").read_text(encoding="utf-8"))
        data = all_results[filename]
    except (OSError, ValueError, KeyError):
        st.error("Data evaluasi untuk model ini belum tersedia.")
        render_architecture(filename)
        return

    overview, evaluation, clusters = st.tabs(["Cara kerja", "Evaluasi", "Klaster dan FCM"])
    with overview:
        render_architecture(filename)
        st.markdown("**Keluaran di web**")
        st.write("Skor klasifikasi menentukan label hasil prediksi. Derajat keanggotaan menjelaskan kedekatan "
                 "foto terhadap pusat klaster atau kelompok prototipe.")
    with evaluation:
        render_evaluation(data, names, uses_v2=filename == MLP)
    with clusters:
        render_clusters(filename, data, preset, names)
