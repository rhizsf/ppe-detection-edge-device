#!/bin/bash
# =============================================================================
# PROTMIND — run_headless.sh
# Script untuk menjalankan inferensi mode headless (tanpa GUI) secara manual.
# Mendukung deteksi virtual environment secara dinamis.
# =============================================================================

# Dapatkan absolute path ke root directory proyek
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
CURRENT_USER=$USER

cd "$PROJECT_DIR"

# Aktifkan virtual environment secara dinamis
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Menggunakan virtual environment yang sedang aktif: $VIRTUAL_ENV"
elif [ -f "venv/bin/activate" ]; then
    echo "Mengaktifkan virtual environment lokal..."
    source venv/bin/activate
elif [ -f "/home/$CURRENT_USER/protmind/protmind-env/bin/activate" ]; then
    echo "Mengaktifkan virtual environment $CURRENT_USER (protmind-env)..."
    source "/home/$CURRENT_USER/protmind/protmind-env/bin/activate"
else
    echo "Peringatan: Virtual environment tidak ditemukan. Mencoba menjalankan dengan Python default."
fi

# Jalankan inferensi mode headless dengan deteksi platform otomatis
if [ -f "/usr/bin/python3.6" ] && [ -f "/usr/lib/aarch64-linux-gnu/libgomp.so.1" ]; then
    echo "Jetson Nano terdeteksi. Menjalankan main_jetson.py pada Python 3.6 dengan akselerasi GPU..."
    LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 /usr/bin/python3.6 src/deployment/main_jetson.py --source 0 --headless --skip-frames 2
else
    echo "Bukan Jetson Nano. Menjalankan main.py dev fallback..."
    python src/main.py --source 0 --headless
fi
