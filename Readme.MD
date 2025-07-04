# 🧠 Employee Face Detection System

Sistem pendeteksi kehadiran karyawan menggunakan deteksi wajah real-time. Dibangun dengan Python dan dirancang untuk lingkungan kerja sederhana, sistem ini secara otomatis menampilkan status kehadiran karyawan serta memberikan notifikasi suara tanpa menggunakan file suara eksternal.

## 🔍 Fitur Utama

- Deteksi wajah real-time via webcam menggunakan MediaPipe + OpenCV
- Status tampilan:
  - ✅ **EMPLOYEE DETECTED** (warna hijau + suara via text-to-speech)
  - ❌ **DETECTING...** lalu *"Employee not detected"* jika tidak ditemukan selama 3 detik
- Notifikasi suara otomatis via `pyttsx3` (TTS, tanpa file .wav)
- Pencatatan ke file `attendance_log.csv`:
  - Status: ARRIVED / ABSENT
  - Durasi absen dalam detik
- Informasi tambahan pada layar:
  - FPS
  - Jam saat ini
  - Lama absen (jika ada)

## 🛠 Teknologi yang Digunakan

- Python 3.x
- [OpenCV](https://opencv.org/) – kamera & tampilan real-time
- [MediaPipe](https://google.github.io/mediapipe/) – deteksi wajah
- [pyttsx3](https://pypi.org/project/pyttsx3/) – text-to-speech engine
- [NumPy](https://numpy.org/) – smoothing deteksi
- Modul Python standar: `os`, `time`, `datetime`, `threading`

## 🚀 Cara Menjalankan

1. **Instalasi dependensi**

```bash
pip install opencv-python mediapipe pyttsx3 numpy

    Untuk Linux, pastikan espeak dan aplay terinstall agar pyttsx3 bisa berbunyi.
    Untuk Windows, tidak perlu file audio tambahan.

    Jalankan program

python main.py

    Tekan q untuk keluar dari jendela kamera.

📁 Struktur Proyek

employee-detector/
├── main.py
├── attendance_log.csv   # dibuat otomatis
└── README.md

📝 Format Log

Program akan menyimpan riwayat deteksi ke file attendance_log.csv dalam format berikut:

Timestamp,Status,Duration
2025-07-04 15:21:12,ARRIVED,0
2025-07-04 15:25:33,ABSENT,4

📌 Catatan Tambahan

    Program hanya menggunakan satu file Python (main.py)

    Tidak menggunakan GUI eksternal seperti Tkinter — hanya jendela OpenCV

    Semua suara dihasilkan otomatis lewat Text-to-Speech, tanpa file .wav

    Bisa digunakan untuk keperluan pribadi, startup kecil, atau demonstrasi AI

💡 Rencana Pengembangan

    Mode multi-user (deteksi lebih dari satu wajah)

    Dashboard laporan bulanan

    Penyimpanan data ke cloud atau server lokal

    Support kamera eksternal (USB / IP Camera)
