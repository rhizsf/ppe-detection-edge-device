#!/bin/bash
# =============================================================================
# PROTMIND — run_headless.sh
# Script untuk menjalankan inferensi mode headless (tanpa GUI) secara manual.
# =============================================================================

# Dapatkan absolute path ke root directory proyek
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_DIR"

# Aktifkan virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment (venv) tidak ditemukan!"
    exit 1
fi

# Jalankan inferensi dengan parameter input 0 dan mode headless
python src/main.py --source 0 --headless
