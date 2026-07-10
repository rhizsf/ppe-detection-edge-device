#!/bin/bash
# =============================================================================
# PROTMIND — setup_jetson.sh
# Script untuk otomatisasi penyiapan lingkungan dan servis APD pada Jetson Nano.
# =============================================================================

# Warna untuk output log
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}   PROTMIND APD SYSTEM — JETSON NANO DEPLOYMENT SETUP       ${NC}"
echo -e "${GREEN}============================================================${NC}"

# 1. Pastikan script dijalankan di root project directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: Jalankan script ini dari root project directory (di luar folder src/deployment)${NC}"
    exit 1
fi

# 2. Buat virtual environment jika belum ada
if [ ! -d "venv" ]; then
    echo -e "${GREEN}[1/5] Membuat Virtual Environment Python...${NC}"
    python3 -m venv venv
fi

# 3. Aktifkan venv dan pasang dependensi
echo -e "${GREEN}[2/5] Menginstal dependensi dari requirements.txt...${NC}"
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Konversi model PyTorch (.pt) ke TensorRT (.engine) untuk akselerasi GPU Jetson
echo -e "${GREEN}[3/5] Mengonversi model kustom ke TensorRT (FP16)...${NC}"
if [ -f "models/best_person.pt" ]; then
    echo "Mengekspor model Person (Stage 1)..."
    yolo export model=models/best_person.pt format=engine half=True device=0
else
    echo -e "${RED}Peringatan: models/best_person.pt tidak ditemukan. Melewati ekspor.${NC}"
fi

if [ -f "models/best_ppe.pt" ]; then
    echo "Mengekspor model APD (Stage 2)..."
    yolo export model=models/best_ppe.pt format=engine half=True device=0
else
    echo -e "${RED}Peringatan: models/best_ppe.pt tidak ditemukan. Melewati ekspor.${NC}"
fi

# 5. Konfigurasi Systemd Service untuk auto-boot di Jetson
echo -e "${GREEN}[4/5] Memasang systemd service di sistem Linux...${NC}"
if [ -f "src/deployment/ppe_detection.service" ]; then
    sudo cp src/deployment/ppe_detection.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable ppe_detection.service
    echo -e "${GREEN}✓ Systemd service berhasil dipasang dan diaktifkan (enable).${NC}"
else
    echo -e "${RED}Error: File src/deployment/ppe_detection.service tidak ditemukan!${NC}"
fi

echo -e "${GREEN}[5/5] Memulai layanan APD...${NC}"
sudo systemctl start ppe_detection.service

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}   SELESAI! Sistem APD berjalan di latar belakang (Headless) ${NC}"
echo -e "${GREEN}   Gunakan perintah: sudo systemctl status ppe_detection      ${NC}"
echo -e "${GREEN}   Untuk memantau log: journalctl -u ppe_detection -f       ${NC}"
echo -e "${GREEN}============================================================${NC}"
