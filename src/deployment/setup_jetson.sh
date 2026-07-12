#!/bin/bash
# =============================================================================
# PROTMIND — setup_jetson.sh
# Script untuk otomatisasi penyiapan lingkungan dan servis APD pada Jetson Nano.
# Menangani deteksi environment dan path secara dinamis untuk akun pengguna.
# =============================================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

PROJECT_DIR=$(pwd)
CURRENT_USER=$USER

# 2. Cek apakah virtual environment sedang aktif
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${GREEN}Virtual Environment terdeteksi aktif: $VIRTUAL_ENV${NC}"
    PYTHON_EXEC="$VIRTUAL_ENV/bin/python"
else
    # Jika tidak ada venv aktif, coba cari path standar milik pengguna
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        PYTHON_EXEC="$PROJECT_DIR/venv/bin/python"
    elif [ -f "/home/$CURRENT_USER/protmind/protmind-env/bin/activate" ]; then
        source "/home/$CURRENT_USER/protmind/protmind-env/bin/activate"
        PYTHON_EXEC="/home/$CURRENT_USER/protmind/protmind-env/bin/python"
    else
        echo -e "${YELLOW}Peringatan: Tidak ada virtual environment aktif.${NC}"
        echo -e "Membuat virtual environment lokal './venv' menggunakan Python 3..."
        python3 -m venv venv
        source venv/bin/activate
        PYTHON_EXEC="$PROJECT_DIR/venv/bin/python"
    fi
fi

# Cetak info versi python yang aktif
echo -e "Menggunakan Python: $($PYTHON_EXEC --version) di ($PYTHON_EXEC)"

# 3. Verifikasi instalasi PyTorch Jetson Nano (sangat penting untuk akselerasi GPU)
echo -e "${GREEN}[1/5] Memverifikasi instalasi PyTorch & CUDA...${NC}"
$PYTHON_EXEC -c "
import torch
print('  -> PyTorch Version:', torch.__version__)
print('  -> CUDA Available :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  -> CUDA Device    :', torch.cuda.get_device_name(0))
else:
    print('  [WARNING] PyTorch tidak mendeteksi GPU CUDA. Pelatihan/inferensi akan sangat lambat!')
"

# 4. Pasang dependensi tambahan tanpa merusak PyTorch Jetson kustom
echo -e "${GREEN}[2/5] Menginstal dependensi tambahan...${NC}"
$PYTHON_EXEC -m pip install --upgrade pip
# Ultralytics dipasang. Pip akan mendeteksi jika torch sudah ada dan tidak akan mendownload ulang.
$PYTHON_EXEC -m pip install -r requirements.txt

# 5. Konversi model PyTorch (.pt) ke TensorRT (.engine) untuk akselerasi GPU Jetson
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
echo -e "${GREEN}[3/5] Mengonversi model kustom ke TensorRT (FP16)...${NC}"

# Mengekspor Stage 1 (Person)
if [ -f "models/best_person.pt" ]; then
    echo "Mengekspor model Person (Stage 1)..."
    # Ekspor ke ONNX menggunakan Python virtual environment (untuk menghindari Pickle error di Python 3.6)
    $PYTHON_EXEC -c "from ultralytics import YOLO; YOLO('models/best_person.pt').export(format='onnx', device='cpu')"
    
    if [ -f "$TRTEXEC" ] && [ -f "models/best_person.onnx" ]; then
        echo "Mengompilasi ONNX ke TensorRT Engine..."
        $TRTEXEC --onnx=models/best_person.onnx --saveEngine=models/best_person.engine --fp16
    fi
else
    echo -e "${YELLOW}Info: models/best_person.pt tidak ditemukan. Ekspor dilewati.${NC}"
fi

# Mengekspor Stage 2 (PPE)
TARGET_PPE_PT=""
if [ -f "models/best_ppe.pt" ]; then
    TARGET_PPE_PT="models/best_ppe.pt"
elif [ -f "models/best_ppe_20260710_022641.pt" ]; then
    TARGET_PPE_PT="models/best_ppe_20260710_022641.pt"
    ln -sf best_ppe_20260710_022641.pt models/best_ppe.pt
fi

if [ -n "$TARGET_PPE_PT" ]; then
    echo "Mengekspor model APD ($TARGET_PPE_PT)..."
    # Ekspor ke ONNX menggunakan Python virtual environment
    $PYTHON_EXEC -c "from ultralytics import YOLO; YOLO('$TARGET_PPE_PT').export(format='onnx', device='cpu')"
    
    # Dapatkan nama berkas onnx
    PPE_ONNX="${TARGET_PPE_PT%.pt}.onnx"
    PPE_ENGINE="${TARGET_PPE_PT%.pt}.engine"
    
    if [ -f "$TRTEXEC" ] && [ -f "$PPE_ONNX" ]; then
        echo "Mengompilasi ONNX ke TensorRT Engine..."
        $TRTEXEC --onnx="$PPE_ONNX" --saveEngine="$PPE_ENGINE" --fp16
        
        if [ "$TARGET_PPE_PT" = "models/best_ppe_20260710_022641.pt" ]; then
            ln -sf best_ppe_20260710_022641.engine models/best_ppe.engine
        fi
    fi
else
    echo -e "${YELLOW}Info: Berkas model APD (.pt) tidak ditemukan. Ekspor dilewati.${NC}"
fi

# 6. Generasi secara dinamis Systemd Service untuk auto-boot di Jetson
echo -e "${GREEN}[4/5] Membuat berkas Systemd Service secara dinamis...${NC}"
cat <<EOF > src/deployment/ppe_detection.service
[Unit]
Description=Protmind Real-Time APD Edge Detection Service
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$PROJECT_DIR
Environment="LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1"
ExecStart=/usr/bin/python3.6 src/deployment/main_jetson.py --source 0 --headless --skip-frames 2
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Salin service ke systemd
sudo cp src/deployment/ppe_detection.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ppe_detection.service
echo -e "${GREEN}✓ Systemd service dipasang pada /etc/systemd/system/ppe_detection.service${NC}"

# 7. Memulai layanan APD
echo -e "${GREEN}[5/5] Memulai layanan APD...${NC}"
sudo systemctl start ppe_detection.service

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}   SELESAI! Sistem APD berjalan di latar belakang (Headless) ${NC}"
echo -e "${GREEN}   Pengguna: $CURRENT_USER | Folder: $PROJECT_DIR           ${NC}"
echo -e "${GREEN}   Gunakan perintah: sudo systemctl status ppe_detection      ${NC}"
echo -e "${GREEN}   Untuk memantau log: journalctl -u ppe_detection -f       ${NC}"
echo -e "${GREEN}============================================================${NC}"
