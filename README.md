# Scene Classification: Custom CNN from Scratch vs Transfer Learning (MobileNetV2)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<USERNAME>/<REPO_NAME>/blob/main/notebooks/DL_Assignment_CNN_vs_TransferLearning.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: TensorFlow 2.15+](https://img.shields.io/badge/Framework-TensorFlow%202.15%2B-orange.svg)](https://tensorflow.org)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)

Proyek ini merupakan tugas mata kuliah **Deep Learning** yang mengimplementasikan dan membandingkan secara komparatif performa dua paradigma pemodelan *Computer Vision*: **Custom Convolutional Neural Network (CNN from scratch)** dan **Transfer Learning (MobileNetV2 pre-trained on ImageNet)** pada tugas klasifikasi pemandangan alam (*Scene Classification*).

---

## 📌 Ringkasan Tugas & Dataset

- **Domain**: Computer Vision — Scene Classification
- **Dataset**: [Intel Image Classification (Kaggle: `puneet6060/intel-image-classification`)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Jumlah Kelas (6)**:
  1. `buildings`
  2. `forest`
  3. `glacier`
  4. `mountain`
  5. `sea`
  6. `street`
- **Dimensi Input**: $150 \times 150 \times 3$ (sesuai resolusi asli dataset; efisien untuk akselerator GPU Colab)
- **Batch Size**: 32
- **Maksimum Epoch**: 30 (dilengkapi `EarlyStopping(patience=5)` dan `ReduceLROnPlateau(patience=2)`)
- **Global Reproducibility Seed**: 42

---

## 📂 Struktur Repositori

```text
dl-assignment-cnn-vs-transfer-learning/
├── README.md                                       # Dokumentasi lengkap proyek
├── requirements.txt                                # Daftar dependensi Python
├── .gitignore                                      # Menolak commit dataset, bobot model, & artefak besar
├── LICENSE                                         # Lisensi MIT
├── notebooks/
│   └── DL_Assignment_CNN_vs_TransferLearning.ipynb # Notebook utama (Standalone, siap dieksekusi di Colab T4)
├── src/                                            # Modul Python modular & reusable
│   ├── __init__.py                                 # Package init
│   ├── data_loader.py                              # Pengunduh dataset, resolver rekursif double-nested, & pipeline tf.data
│   ├── models.py                                   # Definisi Custom CNN & MobileNetV2 Transfer Learning
│   ├── train.py                                    # Callback builder, training loop, & logger riwayat
│   ├── evaluate.py                                 # Metrik evaluasi, classification report, & confusion matrix
│   └── gradcam.py                                  # Visualisasi interpretasi model Grad-CAM
└── outputs/
    └── .gitkeep                                    # Folder artefak grafik, report TXT, & tabel CSV
```

> **Catatan Keamanan & Integritas Repositori:**  
> Berkas berukuran besar seperti dataset gambar, arsip `.zip`, *checkpoint* bobot model (`.keras`, `.h5`), berkas kredensial (`kaggle.json`, `.env`), serta grafik hasil training diabaikan oleh Git via `.gitignore`. Folder `outputs/` hanya melacak `.gitkeep`.

---

## 🧠 Perbandingan Arsitektur Model

### 1. Custom CNN from Scratch
Arsitektur konvolusional 4 blok hierarkis yang dirancang khusus:
- **Blok 1 & 2:** `Conv2D(32/64, 3x3)` + `BatchNormalization` + `MaxPooling2D(2x2)` + `Spatial Dropout(0.2 - 0.25)`.
- **Blok 3 & 4:** `Conv2D(128, 3x3)` + `BatchNormalization` + `MaxPooling2D(2x2)` + `Spatial Dropout(0.3)`.
- **Classification Head:** `GlobalAveragePooling2D()` $\rightarrow$ `Dense(128, activation='relu')` $\rightarrow$ `BatchNormalization` $\rightarrow$ `Dropout(0.4)` $\rightarrow$ `Dense(6, activation='softmax')`.
- Dilatih dari nol (*random initialization*) menggunakan optimizer Adam ($lr=10^{-3}$).

### 2. Transfer Learning MobileNetV2 (ImageNet)
Memanfaatkan representasi visual kaya dari model pre-trained MobileNetV2 pada 1.4 juta citra ImageNet:
- **Tahap 1 (Feature Extraction):** Seluruh *backbone* dibekukan (`base_model.trainable = False`), hanya *custom head* (`GlobalAveragePooling2D` + `Dropout(0.3)` + `Dense(6, softmax)`) yang dilatih ($lr=10^{-3}$).
- **Tahap 2 (Fine-Tuning Terarah):** Membuka pembekuan pada 30 layer terakhir *backbone* dengan $lr=10^{-5}$ yang sangat kecil, menjaga layer `BatchNormalization` tetap dalam mode inferensi (`training=False`) untuk menghindari distorsi statistik running mean & variance.

---

## 🚀 Cara Menjalankan di Google Colab

Notebook `notebooks/DL_Assignment_CNN_vs_TransferLearning.ipynb` bersifat **standalone** (seluruh fungsi modular di-inline di notebook) sehingga dapat langsung diunggah atau dibuka di Google Colab dan dijalankan via **Runtime $\rightarrow$ Run all**.

### Penanganan Struktur Dataset Double-Nested
Kaggle dataset versi ini menghasilkan struktur direktori bertingkat ganda (*double-nested*):
```text
seg_train/seg_train/<class>/
seg_test/seg_test/<class>/
seg_pred/seg_pred/ (diabaikan)
```
Notebook dan modul `src/data_loader.py` dilengkapi fungsi `resolve_dataset_dirs()` yang secara rekursif menelusuri struktur direktori dan secara otomatis menemukan path `seg_train` dan `seg_test` terdalam tanpa hardcoding path atau nomor versi.

### 3 Opsi Pengunduhan Dataset:
1. **Opsi A — Kagglehub (Rekomendasi & Default):**
   ```python
   import kagglehub
   raw_path = kagglehub.dataset_download("puneet6060/intel-image-classification")
   train_dir, test_dir = resolve_dataset_dirs(raw_path)
   ```
   Langsung berjalan tanpa setup kredensial jika dataset bersifat publik.

2. **Opsi B — Kaggle CLI via Colab Secrets:**
   - Masukkan secret `KAGGLE_USERNAME` dan `KAGGLE_KEY` pada tab 🔑 **Secrets** di panel kiri Google Colab.
   - Aktifkan toggle akses notebook, lalu jalankan cell Opsi B.

3. **Opsi C — Google Drive ZIP:**
   - Unggah berkas `intel-image-classification.zip` ke Google Drive Anda.
   - Mount Google Drive pada notebook:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Jalankan fungsi ekstraksi `extract_from_drive_zip()`.

---

## 📊 Hasil Evaluasi & Benchmark

> *Bagian ini diisi dengan hasil eksperimen riil setelah mengeksekusi pelatihan di Google Colab.*

### Tabel Perbandingan Metrik (Placeholder)

| Model | Test Accuracy | Test Loss | Macro Precision | Macro Recall | Macro F1-Score | Training Time (s) | Trainable Params | Total Params |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN from Scratch** | *[0.XX]* | *[X.XXXX]* | *[0.XX]* | *[0.XX]* | *[0.XX]* | *[XX.X s]* | ~450,000 | ~450,000 |
| **MobileNetV2 (Transfer Learning)** | *[0.XX]* | *[X.XXXX]* | *[0.XX]* | *[0.XX]* | *[0.XX]* | *[XX.X s]* | ~7,700 (T1) / ~800k (T2) | ~2,260,000 |

### Artefak Evaluasi yang Dihasilkan
Seluruh artefak tersimpan otomatis pada direktori `/content/outputs/`:
- `custom_cnn_learning_curves.png` & `mobilenetv2_learning_curves.png`: Grafik Loss dan Akurasi.
- `custom_cnn_classification_report.txt` & `mobilenetv2_classification_report.txt`: Laporan metrik per-kelas.
- `custom_cnn_confusion_matrix.png` & `mobilenetv2_confusion_matrix.png`: Heatmap matriks kebingungan.
- `model_comparison_charts.png`: Bar chart perbandingan Accuracy, Loss, Macro F1, dan Durasi Komputasi.
- `benchmark_results.csv`: Tabel rekap metrik format CSV.
- `gradcam_samples.png`: Visualisasi peta atensi Grad-CAM pada 4 sampel citra uji.

---

## 🔍 Analisis & Kesimpulan (Placeholder Analisis)

1. **Efektivitas Transfer Learning:**  
   *[Tuliskan analisis perbandingan performa: bagaimana representasi fitur pre-trained ImageNet membantu MobileNetV2 mengungguli Custom CNN terutama pada kelas yang memiliki tekstur kompleks seperti 'sea' dan 'forest'].*
2. **Diagnosis Kebingungan Antar-Kelas:**  
   *[Tuliskan observasi Confusion Matrix mengenai misklasifikasi antara kelas 'glacier' vs 'mountain' yang secara visual sama-sama memiliki kontur bebatuan dan salju putih].*
3. **Interpretasi Visual Grad-CAM:**  
   *[Jelaskan apakah aktivasi Grad-CAM terfokus pada bagian semantik citra yang benar atau terdapat bias latar belakang].*
4. **Trade-Off Komputasi vs Akurasi:**  
   *[Diskusikan efisiensi memori, kecepatan inferensi, dan waktu konvergensi kedua model].*

---

## 📄 Lisensi

Proyek ini didistribusikan di bawah lisensi **MIT License**. Lihat berkas [LICENSE](LICENSE) untuk rincian selengkapnya.
