# 🥥 Aplikasi Klasifikasi Kematangan Kelapa Lokal

Aplikasi web sederhana (Streamlit) untuk mengklasifikasikan tingkat kematangan
buah kelapa lokal dari foto, menggunakan model CNN (backbone ResNet-18) yang
dilatih dengan pendekatan hybrid GA-PSO-Simulated Annealing + Attention Mechanism.

## Kelas Output
- **Mature** — Matang
- **Potential** — Berpotensi / menuju matang
- **Premature** — Belum matang / muda

## Cara Menjalankan (lokal)

1. Pastikan Python 3.9+ terinstal.
2. Install dependency:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```
4. Buka browser ke alamat yang muncul di terminal (biasanya `http://localhost:8501`).

## Struktur File
- `app.py` — kode aplikasi Streamlit (UI + inference).
- `model.pth` — bobot model (state_dict) hasil pelatihan.
- `requirements.txt` — daftar library yang dibutuhkan.

## Catatan
- File `.pth` yang disediakan hanya berisi bobot (state_dict), bukan kode arsitektur.
  Arsitektur (ResNet-18 + classifier head kustom 3 kelas) sudah direkonstruksi ulang
  di dalam `app.py` agar cocok dan bisa dimuat dengan benar.
- **Deteksi objek "apakah ini kelapa" menggunakan CLIP** (`openai/clip-vit-base-patch32`, ±600MB,
  diunduh otomatis saat aplikasi pertama kali start). CLIP dipakai sebagai gatekeeper open-vocabulary
  sebelum gambar masuk ke model klasifikasi kematangan, karena model kematangan sendiri hanya tahu
  3 kelas (Mature/Potential/Premature) dan tidak pernah diajari seperti apa "bukan kelapa" itu.
- ⚠️ **Peringatan resource:** menambahkan CLIP membuat aplikasi jauh lebih berat (RAM & waktu loading
  pertama kali). Di **Streamlit Community Cloud gratis (±1GB RAM)**, ada risiko aplikasi kehabisan
  memori terutama saat torch + torchvision + CLIP dimuat bersamaan. Kalau ini terjadi, pertimbangkan:
  - Ganti ke model CLIP yang lebih kecil (MobileCLIP/TinyCLIP)
  - Panggil CLIP lewat Hugging Face Inference API (bukan load lokal)
  - Upgrade ke tingkatan Streamlit Cloud berbayar, atau pindah host (Hugging Face Spaces, Railway, dll)
- Untuk deploy online, bisa memakai Streamlit Community Cloud, Hugging Face Spaces,
  atau platform lain yang mendukung Python (Railway, Render, dll).
