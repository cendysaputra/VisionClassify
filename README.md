# cifar10-app

Aplikasi web untuk mengklasifikasikan satu foto. Ada dua mode:

- **CIFAR-10** — model CNN yang sudah ada, 10 kelas: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, dan Truck.
- **Umum** — model zero-shot untuk benda sehari-hari. Daftarnya 256 label, termasuk orang, rumah, batu, hewan, makanan, kendaraan, dan pemandangan.

## Mode CIFAR-10

Modelnya ResNet-18 untuk CIFAR (`model_cnn_cifar10_v6.h5`, 11,17 juta parameter, 45 MB). Pada 10.000 gambar uji CIFAR-10 resmi:

| Tampilan saat prediksi | v5 (sebelumnya) | v6 (sekarang) |
|---|---|---|
| Satu gambar | 95,71% (9.571) | **95,86%** (9.586) |
| Gambar + cermin horizontal | 96,06% (9.606) | **96,18%** (9.618) |
| Cermin + geseran 1 piksel ke empat arah, 10 tampilan. Aplikasi memakai ini. | tidak diukur | **96,29%** (9.629) |

Jadi angka aplikasi naik dari 96,06% ke 96,29%, yaitu 23 gambar lebih banyak yang benar dari 10.000.

Foto biasa tidak akan setepat itu: sebelum diprediksi, gambar ditegakkan menurut data EXIF, di-crop persegi dari tengah jika tidak persegi, lalu diperkecil menjadi 32×32 piksel dengan nilai warna 0–1. Model menormalisasi warna di dalamnya. Prediksi merata-ratakan 10 tampilan: gambar asli, empat geseran satu piksel (tepi yang kosong diisi hitam), dan cermin horizontal dari kelimanya. Jika probabilitas tertinggi di bawah 50%, halaman menampilkan peringatan.

Sepuluh tampilan untuk satu gambar memakan sekitar 100 ms di mesin uji 4 inti, dibanding sekitar 60 ms untuk dua tampilan.

### Cara angka uji didapat

Himpunan latih resmi 50.000 gambar dibagi dengan seed 42: 45.000 untuk pelatihan dan 5.000 untuk validasi (500 gambar per kelas). Himpunan uji tidak dipakai untuk melatih atau memilih apa pun.

1. **Model dasar v5.** Resepnya: pad 4 lalu crop acak, flip horizontal, Cutout 16, SGD Nesterov momentum 0,9, weight decay 5e-4, batch 128, 200 epoch, learning rate 0,1 yang turun 5× setelah epoch 60, 120, dan 160. Checkpoint terbaik adalah epoch 196 menurut validasi. Log lengkapnya di [results/train_log.txt](results/train_log.txt).
2. **Lanjutan SWA menjadi v6.** `scripts/finetune_swa.py` meneruskan SGD dari bobot v5 dengan learning rate tetap 0,01 dan augmentasi yang sama selama 32 epoch, merata-ratakan bobot di akhir epoch 5 sampai 32 (Stochastic Weight Averaging), lalu menghitung ulang statistik BatchNorm dari himpunan latih. Skrip memeriksa bahwa pembagian datanya sama dengan v5, sehingga 5.000 gambar validasi tetap tidak pernah dilatih.
3. **Pemilihan dari validasi.** Dua model (v5 dan rata-rata SWA) dan empat kebijakan tampilan dibandingkan pada 5.000 gambar validasi. Yang tertinggi adalah SWA dengan cermin + geseran 1 piksel: 4.836 benar (96,72%), dibanding 4.824 (96,48%) untuk v5 dengan cermin. Validasi v5 sudah dipakai untuk memilih checkpoint-nya, jadi angka validasi ini bukan akurasi uji.
4. **Uji.** Setelah pilihan itu beku, himpunan uji dibaca sekali untuk model terpilih.

Lanjutan SWA berjalan di runner GitHub (4 inti CPU, bfloat16) selama 17.954 detik (4,99 jam).

Bukti yang di-commit:

- [Log lanjutan SWA](results/swa_log.txt) dan [metriknya](results/swa_metrics.json): skor validasi semua kandidat, pilihan, dan angka uji
- [Metrik uji JSON](results/metrics.json): akurasi uji, per kelas, matriks konfusi
- [Matriks konfusi uji dengan tampilan aplikasi](results/confusion_matrix_tta.png)
- [SHA-256 model](results/model_sha256.txt)

Mengukur ulang berkas yang sama, tanpa melatih:

```bash
python scripts/eval_cifar_test.py
```

Perintah itu memakai fungsi tampilan yang sama dengan aplikasi dan mencetak 9.586, 9.618, dan 9.629 dari 10.000. `--write-results` menulis ulang JSON, gambar, dan checksum. Workflow **Actions → Verify model accuracy** menjalankan pengukuran yang sama di GitHub, ditambah satu pengukuran dengan kode yang terpisah dari aplikasi.

Mode ini hanya memuat TensorFlow saat dipilih.

## Mode Umum

Mode Umum memakai CLIP ViT-B/32 dari OpenAI, dalam berkas ONNX terkuantisasi (sekitar 85 MB) lewat ONNX Runtime. Bobot tidak disimpan di git. Kunjungan pertama mengunduhnya ke cache server, lalu `st.cache_resource` menahan session-nya. Embedding teks untuk 256 label sudah dihitung dan ikut di repositori (`data/general_label_embeddings.npz`, sekitar 575 KB), jadi runtime hanya menjalankan encoder gambar.

Label ditulis dalam bahasa Inggris untuk CLIP, lalu ditampilkan dalam bahasa Indonesia. Contoh: `house` menjadi Rumah, `rock` menjadi Batu (prompt-nya mencakup batu, stone, dan boulder), `man` menjadi Pria.

Hasilnya 5 tebakan teratas beserta probabilitas. Probabilitas itu softmax atas ratusan label, jadi tebakan yang benar sering berada di bawah 50%. Halaman menulis **Tidak yakin (not sure)** jika foto tidak lebih mirip label mana pun dibanding kalimat generik ("gambar acak", "gambar abstrak", "foto buram"), atau jika dua label teratas hampir seri.

Batasan yang memang ada:

- Ini bukan detektor yang dilatih pada rumah atau orang. Foto yang jelas biasanya kena, foto yang sibuk bisa meleset.
- Batu yang memenuhi bingkai terbaca sebagai Batu. Foto yang separuh air bisa terbaca sebagai danau, atau ditandai tidak yakin.
- Orang dengan aksesori yang sangat menonjol, misalnya kacamata di cahaya latar, bisa ditandai tidak yakin.
- Daftar label tertutup. Benda di luar daftar akan menempel ke label terdekat atau ditandai tidak yakin.

Pada mesin uji, session CLIP memakai sekitar 200 MB RAM setelah Streamlit dimuat. Memuat model CIFAR-10 memakai sekitar 600 MB karena TensorFlow. Aplikasi hanya memuat model mode yang sedang aktif. Di paket gratis Streamlit Community Cloud (sekitar 1 GB RAM), muat ulang halaman saat berganti mode: TensorFlow sering tidak mengembalikan memorinya ke sistem operasi.

## Struktur Proyek

- `app.py` — antarmuka Streamlit dan sakelar mode
- `classify.py` — pembacaan gambar, praproses CIFAR-10, dan ringkasan prediksi
- `general.py` — unduh model Umum, praproses CLIP, dan skor label
- `general_labels.py` — 256 label, prompt bahasa Inggris, nama tampilan bahasa Indonesia
- `data/general_label_embeddings.npz` — embedding teks yang sudah dihitung
- `scripts/build_general_embeddings.py` — bangun ulang embedding itu (bukan dependensi runtime)
- `style.css` — tampilan hasil prediksi
- `model_cnn_cifar10_v6.h5` — model CIFAR-10 yang dipakai aplikasi
- `train.py` — pelatihan model dasar (v5)
- `scripts/finetune_swa.py` — lanjutan SWA dari model dasar menjadi v6
- `requirements-train.txt` — PyTorch CPU, hanya untuk pelatihan
- `scripts/eval_cifar_test.py` — ukur ulang akurasi uji pada berkas model
- `results/` — log pelatihan, metrik, matriks konfusi, dan SHA-256 model
- `requirements.txt` — dependensi aplikasi
- `requirements-dev.txt` — dependensi aplikasi plus pytest
- `runtime.txt` — Python 3.11 untuk Streamlit Community Cloud
- `.streamlit/config.toml` — batas unggah 5 MB dan tema terang
- `tests/` — uji praproses, aplikasi, dan keluaran model
- `.github/workflows/test.yml` — uji otomatis pada pull request
- `.github/workflows/train-model.yml` — pelatihan model dasar di GitHub Actions
- `.github/workflows/finetune-swa.yml` — lanjutan SWA di GitHub Actions (manual)
- `.github/workflows/verify-model.yml` — ukur ulang akurasi uji di GitHub Actions

## Menjalankan Secara Lokal

Gunakan Python 3.11, sesuai `runtime.txt`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Buka alamat yang dicetak di terminal, lalu pilih mode dan unggah file JPG, PNG, atau WEBP (maksimal 5 MB). Mode CIFAR-10 langsung memakai berkas `.h5`. Mode Umum mengunduh model CLIP sekali, sekitar 85 MB, lalu memakainya dari cache.

## Uji

```bash
pip install -r requirements-dev.txt
pytest
```

Uji model CIFAR memuat berkas `.h5` yang ada di repositori dan memeriksa bahwa keluarannya 10 probabilitas. Uji ini tidak melatih ulang model.

Uji mode Umum yang biasa tidak mengunduh apa pun: skornya diuji dengan embedding palsu dan dengan vektor yang sudah disimpan. Foto sungguhan (orang, rumah, batu) diuji terpisah, karena itu mengunduh atau membaca model CLIP:

```bash
GENERAL_LIVE=1 pytest tests/test_general_live.py
```

Untuk membangun ulang embedding setelah mengubah `general_labels.py`:

```bash
pip install onnxruntime tokenizers numpy
python scripts/build_general_embeddings.py
```

## Melatih Ulang Model

Model dibuat dalam dua tahap.

**Tahap 1, model dasar.** Pelatihan 200 epoch tidak muat di batas waktu runner GitHub, jadi workflow **Actions → Train model → Run workflow** hanya dijalankan manual dan berhenti sesuai batas waktu. Model dasar v5 dilatih dengan skrip yang sama di luar runner.

```bash
pip install -r requirements.txt -r requirements-train.txt
python train.py --epochs 200
```

Perintah itu mengunduh CIFAR-10, menahan 5.000 gambar latih sebagai validasi, dan menulis `model_cnn_cifar10_v5.h5`. Berkas v5 tidak lagi ada di pohon kerja; versi yang dipakai tersimpan di riwayat git.

**Tahap 2, lanjutan SWA.** Muat di satu job runner GitHub (sekitar 5 jam): **Actions → Fine-tune model (SWA) → Run workflow** dari branch selain `main`. Hasilnya di-commit ke branch itu. Secara lokal:

```bash
git show 72895f7:model_cnn_cifar10_v5.h5 > model_cnn_cifar10_v5.h5
python scripts/finetune_swa.py --time-budget-minutes 270
```

Skrip menulis `model_cnn_cifar10_v6.h5` hanya jika model SWA menang di validasi, dan mencatat kebijakan tampilan terpilih di `results/swa_metrics.json`. Kebijakan itu disetel di `TTA_POLICY` pada `classify.py`.

## Deployment

Aplikasi di-deploy lewat Streamlit Community Cloud dari repositori ini.

- Berkas utama: `app.py`
- Versi Python: `runtime.txt` (3.11)
- Dependensi: `requirements.txt` (TensorFlow untuk CIFAR-10, ONNX Runtime untuk mode Umum)
- Unggahan dibatasi 5 MB di `.streamlit/config.toml`

Berkas `model_cnn_cifar10_v6.h5` harus tetap ada di repositori agar mode CIFAR-10 dapat memuatnya. Repositori ini tidak memakai Git LFS; berkas model 45 MB masih di bawah batas 100 MB GitHub. Bobot CLIP tidak di-commit; server mengunduhnya dari Hugging Face saat mode Umum pertama kali dipakai. Jangan menyimpan rahasia di kode; jika suatu saat diperlukan, gunakan `.streamlit/secrets.toml`, yang tidak di-commit.
