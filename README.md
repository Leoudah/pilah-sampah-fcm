# Pilah Sampah

Demo pembelajaran berbasis MobileNetV3-Small, FCM, dan MLP 64. Jalankan dengan Python 3.11.

    python -m pip install -r requirements.txt
    python -m streamlit run streamlit_app.py

Untuk Streamlit Community Cloud, pilih streamlit_app.py sebagai entrypoint. Model ONNX ada di artifacts/lightweight_fcm.

Pada URL HTTPS publik, **Kamera realtime** mengirim frame JPEG tanpa audio melalui koneksi Streamlit. Browser menunggu konfirmasi setiap frame, hingga 2,5 pengiriman per detik. Transport lokal menggunakan WebRTC.

## Sumber data model

- [Garbage Dataset Classification · v5](https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification/versions/5)
- [RealWaste · v1](https://www.kaggle.com/datasets/joebeachcapital/realwaste/versions/1) · [sumber UCI](https://archive.ics.uci.edu/dataset/908/realwaste)
- [Garbage Classification · Mostafa · v1](https://www.kaggle.com/datasets/mostafaabla/garbage-classification/versions/1)
- [Urban Garbage Classification · v4](https://www.kaggle.com/datasets/ionutandreivaduva/garbage-classification/versions/4)
- [Single/Multi-Material Garbage Classification · v1](https://www.kaggle.com/datasets/manfye88/single-multi-material-garbage-classification/versions/1)
- [Multi-Class Garbage Classification · v1](https://www.kaggle.com/datasets/vishallazrus/multi-class-garbage-classification-dataset/versions/1)
- [TrashNet · v1](https://www.kaggle.com/datasets/feyzazkefe/trashnet/versions/1) · [sumber TrashNet](https://github.com/garythung/trashnet)
- [Bobot MobileNetV3-Small ONNX](https://huggingface.co/onnxmodelzoo/mobilenet_v3_small_Opset17)
