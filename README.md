# Pilah Sampah

Demo pembelajaran berbasis MobileNetV3-Small, FCM, dan MLP 64. Jalankan dengan Python 3.11.

    python -m pip install -r requirements.txt
    python -m streamlit run streamlit_app.py

Untuk Streamlit Community Cloud, pilih streamlit_app.py sebagai entrypoint. Model ONNX ada di artifacts/lightweight_fcm.

Pada URL HTTPS publik, **Kamera realtime** mengirim frame JPEG tanpa audio melalui koneksi Streamlit. Browser menunggu konfirmasi setiap frame, hingga 2,5 pengiriman per detik. Transport lokal menggunakan WebRTC.

## Ringkasan sistem

1. **Dataset dan pengolahan data.** V2 menggabungkan enam sumber Kaggle, yaitu Zlatan v5, RealWaste v1, Ionut Andrei Vaduva v4, Mostafa Mohamed v1, man fye v1, dan Vishal Lazrus v1. Label diseragamkan menjadi kardus, kaca, logam, kertas, plastik, dan sampah lainnya. Kami menyaring duplikat identik atau sangat mirip, konflik label, dan gambar V2 yang tumpang tindih dengan TrashNet. Data latih memakai 8.161 foto V2 dan 2.027 foto TrashNet v1. Evaluasi memakai 1.633 foto validasi V2, 1.637 foto test V2, dan 476 foto TrashNet yang disisihkan.
2. **Ekstraksi fitur.** Foto dikoreksi orientasinya, diubah ke RGB, di-resize bilinear dengan sisi terpendek 256 piksel, lalu di-crop tengah 224 × 224 dan dinormalisasi dengan nilai ImageNet. MobileNetV3-Small pralatih dengan bobot tetap menghasilkan 576 fitur per foto. Fitur distandardisasi. Cabang FCM memakai PCA 48 komponen dan normalisasi L2.
3. **Model dan parameter.** FCM melatih 4 prototipe per kelas dengan jarak Euclidean, m = 1,35, inisialisasi K-means++, toleransi 10⁻⁵, maksimal 250 iterasi, dan seed 42. Empat keanggotaan dijumlahkan per kelas. Enam skor FCM distandardisasi dan digabung dengan 576 fitur MobileNet menjadi 582 masukan. MLP memakai 64 neuron tersembunyi, ReLU, 6 keluaran softmax, Adam dengan learning rate 0,001, alpha 0,01, batch 128, maksimal 45 iterasi, dan seed 42.
4. **Evaluasi.** Akurasi mencapai 88,70% di test V2 (1.452/1.637) dan 72,27% pada foto TrashNet yang disisihkan (344/476). ONNX menyimpan skor MLP dan skor FCM agar keduanya bisa ditampilkan di Streamlit.

## Sumber data model

- [Garbage Dataset Classification · v5](https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification/versions/5)
- [RealWaste · v1](https://www.kaggle.com/datasets/joebeachcapital/realwaste/versions/1) · [sumber UCI](https://archive.ics.uci.edu/dataset/908/realwaste)
- [Garbage Classification · Mostafa · v1](https://www.kaggle.com/datasets/mostafaabla/garbage-classification/versions/1)
- [Urban Garbage Classification · v4](https://www.kaggle.com/datasets/ionutandreivaduva/garbage-classification/versions/4)
- [Single/Multi-Material Garbage Classification · v1](https://www.kaggle.com/datasets/manfye88/single-multi-material-garbage-classification/versions/1)
- [Multi-Class Garbage Classification · v1](https://www.kaggle.com/datasets/vishallazrus/multi-class-garbage-classification-dataset/versions/1)
- [TrashNet · v1](https://www.kaggle.com/datasets/feyzazkefe/trashnet/versions/1) · [sumber TrashNet](https://github.com/garythung/trashnet)
- [Bobot MobileNetV3-Small ONNX](https://huggingface.co/onnxmodelzoo/mobilenet_v3_small_Opset17)
