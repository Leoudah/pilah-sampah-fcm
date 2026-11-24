# Pilah Sampah

Demo klasifikasi sampah dengan tiga model ONNX yang bisa dipilih di aplikasi.

    python -m pip install -r requirements.txt
    python -m streamlit run streamlit_app.py

Untuk Streamlit Community Cloud, pilih `streamlit_app.py` sebagai entrypoint. Ketiga artefak tersedia di `artifacts/lightweight_fcm/`.

| Pilihan di web | Artefak | Panel FCM |
| --- | --- | --- |
| FCM + KNN tanpa LDA | `mobilenet_fcm_knn_1a.onnx` | Enam keanggotaan klaster di ruang PCA 50 |
| LDA + FCM + KNN | `mobilenet_lda5_fcm_knn.onnx` | Enam keanggotaan klaster di ruang LDA 5 |
| FCM + MLP 64 | `mobilenet_fcm_mlp64.onnx` | Enam keanggotaan yang dikelompokkan menurut material |

Model LDA menerima fitur MobileNet langsung pada cabang FCM. KNN menggabungkan enam keanggotaan tersebut dengan 50 komponen PCA dari cabang lainnya. Hasil verifikasi ONNX pada 1.200 foto uji cocok dengan notebook, yaitu 1.027 prediksi benar (85,58%).

Pada URL HTTPS publik, **Kamera realtime** mengirim frame JPEG tanpa audio melalui koneksi Streamlit. Browser menunggu konfirmasi setiap frame, hingga 2,5 pengiriman per detik. Transport lokal menggunakan WebRTC.

## Ringkasan skenario 1a

1. **Dataset.** Eksperimen memakai 6.000 foto Zlatan v5, masing-masing 1.000 foto per kelas setelah menyaring duplikat identik dan konflik label. Split terstratifikasi dengan seed 42 menghasilkan 4.800 foto latih dan 1.200 foto uji.
2. **Ekstraksi fitur.** Orientasi foto dikoreksi, diubah ke RGB, di-resize dengan sisi terpendek 256 piksel, di-crop tengah ke 224 × 224, dan dinormalisasi dengan statistik ImageNet. MobileNetV3-Small menghasilkan 576 fitur yang kemudian distandardisasi dan direduksi menjadi 50 komponen PCA.
3. **Klasifikasi.** FCM dilatih dengan enam klaster, jarak Euclidean, `m = 2`, toleransi `10⁻⁵`, dan maksimal 200 iterasi. Enam keanggotaan klaster dikalikan `α = 1` dan digabungkan dengan 50 komponen PCA. KNN memakai `k = 7`, jarak cosine, dan pembobotan jarak. Aplikasi menampilkan skor kelas dari KNN dan derajat keanggotaan dari FCM.
4. **Evaluasi.** Model yang diekspor menghasilkan prediksi yang sama dengan skenario 1a pada seluruh 1.200 foto uji. Sebanyak 1.026 prediksi benar dengan akurasi 85,50%.

## Sumber data model

- [Garbage Dataset Classification · v5](https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification/versions/5)
- [Bobot MobileNetV3-Small ONNX](https://huggingface.co/onnxmodelzoo/mobilenet_v3_small_Opset17)
