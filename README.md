<div align="center">

# 🖼️ CIFAR-10 & General Image Classifier

**Aplikasi klasifikasi gambar ringan berbasis Streamlit & ONNX Runtime — akurasi 98,42% pada CIFAR-10 resmi dan 256 label sehari-hari dalam satu antarmuka.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.59-FF4B4B?logo=streamlit&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/Runtime-ONNX%20INT8-005CED?logo=onnx&logoColor=white)
![CIFAR-10 Accuracy](https://img.shields.io/badge/CIFAR--10%20Test-98.42%25-2EA043)
![Model Size](https://img.shields.io/badge/Model%20Size-22.4%20MB-6F42C1)

</div>

---

## ✨ Fitur Utama

- **🎯 Mode CIFAR-10 (98,42% Test Accuracy)** — Menggunakan **ViT-Small/16** terkuantisasi **INT8 ONNX** (`model_vit_cifar10.onnx`, 22,4 MB) untuk mengklasifikasikan 10 kelas: *Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship,* dan *Truck*.
- **🌍 Mode Umum (256 Label Zero-Shot)** — Menggunakan **CLIP ViT-B/32** (INT8 ONNX) dengan *precomputed text embeddings* untuk mengenali 256 benda sehari-hari (orang, rumah, batu, hewan, makanan, kendaraan, tempat, dll.) dalam bahasa Indonesia + deteksi otomatis **Tidak yakin (not sure)**.
- **⚡ Unified ONNX Runtime (Tanpa TensorFlow)** — Kedua mode berjalan penuh di `onnxruntime`, memangkas ukuran dependensi, menurunkan konsumsi RAM secara drastis, dan mencegah *out-of-memory* di Streamlit Community Cloud.
- **🛡️ Praproses Aman & Responsif** — Koreksi orientasi EXIF otomatis, batas piksel anti-*decompression bomb*, *center-square crop* agar foto asli tidak terdistorsi, serta proteksi unggahan hingga 5 MB.

---

## 📊 Peningkatan Performa & Optimasi

| Metrik | Sebelumnya (v6) | Sekarang (v7) | Perubahan |
| :--- | :---: | :---: | :---: |
| **Arsitektur CIFAR-10** | ResNet-18 (32×32) | **ViT-Small/16 (224×224)** | Upgrade ke Vision Transformer |
| **Akurasi Uji CIFAR-10 (10.000 gambar)** | 96,29% (9.629) | **98,42% (9.842)** | **+2,13%** (+213 gambar benar) |
| **Ukuran Berkas Model** | 45,0 MB (`.h5`) | **22,4 MB (`.onnx` INT8)** | **−50% lebih kecil** |
| **Mesin Inferensi** | TensorFlow CPU + ONNX | **Unified ONNX Runtime** | **Tanpa TensorFlow** |
| **Tampilan Inferensi** | 10-view TTA | **1-view (~88 ms/foto)** | Lebih cepat & efisien |

### Akurasi Uji per Kelas (CIFAR-10 Official Test Set)

| Kelas | Akurasi | Benar / 1.000 | Kelas | Akurasi | Benar / 1.000 |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **Frog** | 99,60% | 996 | **Bird** | 98,60% | 986 |
| **Airplane** | 99,50% | 995 | **Horse** | 98,50% | 985 |
| **Ship** | 99,40% | 994 | **Truck** | 97,90% | 979 |
| **Automobile** | 99,10% | 991 | **Dog** | 96,30% | 963 |
| **Deer** | 99,10% | 991 | **Cat** | 96,20% | 962 |

<details>
<summary>📈 Lihat Confusion Matrix (10.000 Gambar Uji Resmi)</summary>

<br>

![CIFAR-10 Confusion Matrix](results/confusion_matrix_tta.png)

</details>

### Reproduksibilitas & Bukti Evaluasi

Model `model_vit_cifar10.onnx` diekspor dari `MF21377197/vit-small-patch16-224-finetuned-Cifar10` (Apache 2.0, *fine-tuned* pada 45.000 data latih / 5.000 validasi CIFAR-10 tanpa menyentuh himpunan uji), lalu dikuantisasi dinamis ke `QUInt8`.

- 📄 Metrik lengkap JSON: [`results/metrics.json`](results/metrics.json)
- 🔐 Checksum SHA-256: [`results/model_sha256.txt`](results/model_sha256.txt)
- 🖼️ Confusion matrix: [`results/confusion_matrix_tta.png`](results/confusion_matrix_tta.png)

Verifikasi ulang angka uji kapan saja pada 10.000 gambar resmi CIFAR-10:

```bash
python scripts/eval_cifar_test.py
```

---

## 🌍 Cara Kerja Mode Umum

Mode Umum memakai **CLIP ViT-B/32** (INT8 ONNX, ~85 MB) yang diunduh sekali ke cache server dengan verifikasi SHA-256. Embedding teks untuk 256 label sudah tersedia di `data/general_label_embeddings.npz`, sehingga runtime hanya menjalankan *image encoder*.

- Menampilkan **5 tebakan teratas** lengkap dengan nama bahasa Indonesia dan padanan bahasa Inggris.
- Otomatis menandai **Tidak yakin (not sure)** bila kemiripan tidak jauh lebih tinggi dari *null prompts* (`random`, `abstract`, `blurry`) atau bila dua label teratas hampir seri.

---

## 🗂️ Struktur Proyek

```text
cifar10-app/
├── app.py                              # Antarmuka Streamlit & pengatur mode
├── classify.py                         # Praproses & inferensi ONNX CIFAR-10
├── general.py                          # Inferensi & unduh cache CLIP Mode Umum
├── general_labels.py                   # 256 label & prompt Mode Umum
├── model_vit_cifar10.onnx              # Model ViT-Small/16 INT8 ONNX (22,4 MB)
├── style.css                           # Gaya tampilan kartu hasil & progress bar
├── data/
│   └── general_label_embeddings.npz    # Precomputed text embeddings CLIP
├── results/
│   ├── metrics.json                    # Akurasi uji resmi & per kelas
│   ├── model_sha256.txt                # Checksum integritas model ONNX
│   └── confusion_matrix_tta.png        # Visualisasi confusion matrix
├── scripts/
│   └── eval_cifar_test.py              # Skrip evaluasi 10.000 gambar uji CIFAR-10
├── tests/                              # Unit test, smoke test & Streamlit AppTest
├── .streamlit/config.toml              # Tema terang & batas unggah 5 MB
├── .github/workflows/
│   ├── test.yml                        # CI pytest otomatis
│   └── verify-model.yml                # Verifikasi akurasi model di GitHub Actions
├── requirements.txt                    # Dependensi runtime ringan (tanpa TensorFlow)
├── requirements-dev.txt                # Dependensi runtime + pytest
└── runtime.txt                         # Python 3.11 (Streamlit Community Cloud)
```

---

## 🚀 Menjalankan Secara Lokal

Gunakan Python 3.11 (sesuai `runtime.txt`):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Buka URL yang tampil di terminal, pilih mode **CIFAR-10** atau **Umum**, lalu unggah gambar **JPG, PNG, atau WEBP** (maksimal 5 MB).

---

## 🧪 Pengujian

```bash
pip install -r requirements-dev.txt
pytest
```

Untuk menjalankan uji foto sungguhan Mode Umum (mengunduh model CLIP ~85 MB sekali):

```bash
GENERAL_LIVE=1 pytest tests/test_general_live.py
```

---

## ☁️ Deployment

Siap langsung di-deploy ke **Streamlit Community Cloud**:

- **Main file path**: `app.py`
- **Python version**: `3.11` (`runtime.txt`)
- **Dependencies**: `requirements.txt` (ringan, hanya `onnxruntime` — tanpa TensorFlow)
- **Upload limit**: 5 MB (`.streamlit/config.toml`)

---

## 🤝 Contributors

- **[@cendysaputra](https://github.com/cendysaputra)** — Project owner & maintainer
- **Antigravity (Google DeepMind)** — ViT-Small INT8 ONNX upgrade (98,42%), TensorFlow removal, codebase cleanup & documentation
- **[@claude](https://github.com/claude)** — Earlier CIFAR-10 SWA fine-tuning & Mode Umum implementation

