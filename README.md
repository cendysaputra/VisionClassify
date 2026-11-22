<div align="center">

# Aplikasi Klasifikasi Gambar

Aplikasi web ringan berbasis Streamlit dan ONNX Runtime untuk klasifikasi gambar dengan akurasi uji **98,42%** pada CIFAR 10 serta **256 label** benda sehari hari.

![Python](https://img.shields.io/badge/Python-3.11-3776AB)
![Streamlit](https://img.shields.io/badge/Streamlit-1.59-FF4B4B)
![ONNX Runtime](https://img.shields.io/badge/Runtime-ONNX%20INT8-005CED)
![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.42%25-2EA043)
![Model Size](https://img.shields.io/badge/Model%20Size-22.4%20MB-6F42C1)
![License](https://img.shields.io/badge/License-MIT-blue)

</div>

## Tampilan Aplikasi

<table>
  <tr>
    <td align="center" width="50%">
      <img src="results/screenshot_cifar10.png" alt="Tampilan Mode CIFAR 10" width="100%" /><br />
      <sub><b>Mode CIFAR 10</b>: Prediksi 10 kelas dengan ViT Small ONNX (224×224)</sub>
    </td>
    <td align="center" width="50%">
      <img src="results/screenshot_umum.png" alt="Tampilan Mode Umum" width="100%" /><br />
      <sub><b>Mode Umum</b>: Prediksi 256 label harian dengan CLIP ViT B/32 ONNX</sub>
    </td>
  </tr>
</table>

## Ringkasan Fitur

* **Mode CIFAR 10**: Mengklasifikasikan 10 kelas (*Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck*) dengan akurasi uji resmi **98,42%**.
* **Mode Umum**: Mengenali 256 benda sehari hari dalam bahasa Indonesia memakai CLIP zero shot beserta deteksi otomatis **Tidak yakin**.
* **Ringan tanpa TensorFlow**: Kedua mode berjalan di satu mesin **ONNX Runtime**, menghemat RAM dan mempercepat proses deploy.

## Ilustrasi Data & Performa

![Grafik Akurasi dan Ukuran Model](results/accuracy_overview.png)

<table>
  <thead>
    <tr>
      <th align="left">Metrik</th>
      <th align="center">Versi v6</th>
      <th align="center">Versi v7 (Sekarang)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Arsitektur</td>
      <td align="center">ResNet 18 (32×32)</td>
      <td align="center"><b>ViT Small/16 (224×224)</b></td>
    </tr>
    <tr>
      <td>Akurasi Uji (10.000 gambar)</td>
      <td align="center">96,29% (9.629)</td>
      <td align="center"><b>98,42% (9.842)</b></td>
    </tr>
    <tr>
      <td>Ukuran Berkas Model</td>
      <td align="center">45,0 MB (<code>.h5</code>)</td>
      <td align="center"><b>22,4 MB (<code>.onnx</code> INT8)</b></td>
    </tr>
    <tr>
      <td>Mesin Inferensi</td>
      <td align="center">TensorFlow + ONNX</td>
      <td align="center"><b>ONNX Runtime</b></td>
    </tr>
    <tr>
      <td>Kecepatan Inferensi</td>
      <td align="center">10 tampilan (~100 ms)</td>
      <td align="center"><b>1 tampilan (~88 ms)</b></td>
    </tr>
  </tbody>
</table>

<details>
<summary>Lihat Confusion Matrix & Bukti Evaluasi</summary>

<br />

![Confusion Matrix CIFAR 10](results/confusion_matrix_tta.png)

* Metrik JSON lengkap: [`results/metrics.json`](results/metrics.json)
* Checksum SHA 256: [`results/model_sha256.txt`](results/model_sha256.txt)
* Ukur ulang secara mandiri: `python scripts/eval_cifar_test.py`

</details>

## Menjalankan Secara Lokal

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Pengujian

```bash
pip install -r requirements-dev.txt
pytest
```

## Lisensi

Proyek ini dirilis di bawah [Lisensi MIT](LICENSE).
