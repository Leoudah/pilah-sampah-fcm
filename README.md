# Pilah Sampah

Demo pembelajaran berbasis skenario 1a dengan MobileNetV3-Small, PCA 50, Fuzzy C-Means, dan KNN.

    python -m pip install -r requirements.txt
    python -m streamlit run streamlit_app.py

Untuk Streamlit Community Cloud, pilih `streamlit_app.py` sebagai entrypoint. Aplikasi memuat `artifacts/lightweight_fcm/mobilenet_fcm_knn_1a.onnx`.

Pada URL HTTPS publik, **Kamera realtime** mengirim frame JPEG tanpa audio melalui koneksi Streamlit. Browser menunggu konfirmasi setiap frame, hingga 2,5 pengiriman per detik. Transport lokal menggunakan WebRTC.

## Ringkasan sistem

1. **Dataset.** Eksperimen memakai 6.000 foto Zlatan v5, masing-masing 1.000 foto per kelas setelah menyaring duplikat identik dan konflik label. Split terstratifikasi dengan seed 42 menghasilkan 4.800 foto latih dan 1.200 foto uji.
2. **Ekstraksi fitur.** Orientasi foto dikoreksi, diubah ke RGB, di-resize dengan sisi terpendek 256 piksel, di-crop tengah ke 224 × 224, dan dinormalisasi dengan statistik ImageNet. MobileNetV3-Small menghasilkan 576 fitur yang kemudian distandardisasi dan direduksi menjadi 50 komponen PCA.
3. **Klasifikasi.** FCM dilatih dengan enam klaster, jarak Euclidean, `m = 2`, toleransi `10⁻⁵`, dan maksimal 200 iterasi. Enam keanggotaan klaster dikalikan `α = 1` dan digabungkan dengan 50 komponen PCA. KNN memakai `k = 7`, jarak cosine, dan pembobotan jarak. Aplikasi menampilkan skor kelas dari KNN dan derajat keanggotaan dari FCM.
4. **Evaluasi.** Model yang diekspor menghasilkan prediksi yang sama dengan skenario 1a pada seluruh 1.200 foto uji. Sebanyak 1.026 prediksi benar dengan akurasi 85,50%.

## Sumber data model

- [Garbage Dataset Classification · v5](https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification/versions/5)
- [Bobot MobileNetV3-Small ONNX](https://huggingface.co/onnxmodelzoo/mobilenet_v3_small_Opset17)
