# Real-Time PPE/APD Safety Compliance Detection System
### Deployed on NVIDIA Jetson Nano Developer Kit with ALSA Audio Warnings & Telegram Bot Integration

This repository hosts a production-grade, end-to-end edge AI safety monitoring system designed to run on the **NVIDIA Jetson Nano**. It automatically detects Personal Protective Equipment (PPE) / Alat Pelindung Diri (APD) compliance in real-time, plays high-fidelity audio warning messages via a connected **USB Sound Card and speaker**, and securely dispatches warning alerts and photo evidence to a **Telegram Bot** using a local `.env` configuration file.

---

## 1. System Architecture: Two-Stage Detection Pipeline

Unlike traditional single-stage detectors which get easily confused by background noise (such as blue buckets mistaken for helmets or orange traffic cones for safety vests), this system implements a robust **Two-Stage Detection & Relocalization Pipeline**:

```
      [ CAMERA STREAM / VIDEO INPUT ]
                     │
                     ▼
    STAGE 1: Person Detection (yolov8n.pt) ──► Bounding Box (Blue)
                     │
                     ▼ (Automatic BBox Crop)
    STAGE 2: PPE Detection (best_ppe.pt) ────► Evaluates cropped worker space
                     │
                     ▼ (Relative-to-Full Mapping)
    [ Cooldown (10s) & Queue Manager (3s) ]
           /                        \
          /                          \
         ▼                            ▼
  [ Audio Warnings ]          [ Telegram Bot Reports ]
  - Sequential playback       - Rich HTML Text Summary
  - Winsound / Aplay          - Color-Coded Annotated Image
```

1. **Stage 1 (Person Detection)**: Utilizes the pre-trained `yolov8n.pt` model (COCO class 0: `person`) to detect workers.
2. **Automatic Cropping**: The detected person coordinates are cropped from the frame in real-time.
3. **Stage 2 (PPE Detection)**: A second YOLOv8n model (`models/best_ppe.pt`) fine-tuned specifically on cropped person images is executed. It evaluates the worker's body to detect 6 classes:
   - **Safe/Compliant**: `helmet`, `vest`, `safety shoes`
   - **Violations**: `no-helmet`, `no-vest`, `no-safety shoes`
4. **Coordinate Remapping**: The local bounding box coordinates predicted on the cropped image are mapped back to absolute coordinates on the original full image for visual annotation.

---

## 2. Core Features

### 🔊 Asynchronous Audio Queue (Round-Robin)
To ensure the real-time camera loop maintains maximum frame rate, audio warning playback runs in an **asynchronous background thread** using a thread-safe task queue (`queue.Queue`). 
If a frame detects multiple violations (e.g., `no-helmet` and `no-vest`), the background worker plays them sequentially with a **3-second delay**:
*Memutar Audio Helm -> Wait 3s -> Memutar Audio Rompi*.

### ⏱️ 10-Second Cooldown Manager
Prevents repetitive and annoying warning voice spam. If a violation (e.g., `no-helmet`) is detected on consecutive frames, the system enforces a minimum **10-second silence lock** for that specific class before it can be voiced or reported again.

### 🚨 Rich Telegram Bot Alerts & Color-Coding
When a new violation passes the cooldown check, a warning is sent to the Telegram channel containing:
- **Location Identification**: Real-time camera location name.
- **Accurate Timestamp**: Real-time event date and time.
- **Deteksi Summary**: Total people present and a structured violation list.
- **Visual Evidence**: Attached frame capture with professional **Color-Coded** annotations:
  - 🔵 **Blue Box**: Overall worker (`person`) bounding box.
  - 🟢 **Green Box**: Compliant safety gear (`helmet`, `vest`, `safety shoes`).
  - 🔴 **Red Box**: Active violations (`no-helmet`, `no-vest`, `no-safety shoes`).
  
### ⚖️ Programmatic Class Balancing & Transfer Learning
To overcome severe class imbalance common in safety datasets (where compliant workers far outnumber violations), the training pipeline automatically applies deep learning optimization techniques:
- **Dynamic Crop Oversampling**: During the cropping stage, the script dynamically duplicates training images containing violations (`no-helmet` by 3x, `no-safety shoes` by 3x, and `no-vest` by 2x). This is strictly restricted to the `train` split to prevent validation/testing data leakage.
- **Classification Loss Scale (`cls=1.5`)**: Elevates the classification loss weight factor (increased from YOLOv8 default `0.5` to `1.5`) to heavily penalize APD misclassifications.
- **Backbone Layer Freezing (`freeze=10`)**: Freezes the first 10 layers of the pretrained YOLOv8n backbone to preserve visual features, accelerate convergence, and prevent overfitting.

### 💻 Cross-Platform Compatibility
- **Windows (Laptop)**: Native async playback via built-in `winsound` / PowerShell fallbacks.
- **Linux (Jetson Nano)**: Low-latency playback via native `aplay` (ALSA) / PulseAudio utilities.

---

## 3. Installation & Setup

### Prerequisite: Set up Virtual Environment & Requirements
Ensure you are in the workspace root and run the following in your terminal:
```powershell
# 1. Create a clean virtual environment
python -m venv venv

# 2. Activate the virtual environment
# Windows Powershell (adjust policy if blocked):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install CUDA-Enabled PyTorch (Recommended for RTX GPU Laptops)
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Configure Settings (`.env`)
Create a [.env](file:///d:/Developer/AI%20Developer%20Project/ppe-detection-edge-device/.env) file at the root directory of the workspace and set your credentials:
```env
# Telegram Bot Token & Chat ID
TELEGRAM_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Edge Device Settings
LOKASI_KAMERA=Gate 1 - Depan Proyek A
CAMERA_SOURCE=0
```

---

## 4. How to Run the Pipeline

### Step 1: Pre-process & Crop Dataset (Dry-Run)
Test the automatic cropping dataset builder by running:
```powershell
python src/train.py --dry-run
```
This scans your original dataset folder (`assets/dataset/`), detects persons using `yolov8n.pt`, crops them, translates the PPE annotations, and writes a new cropped dataset config at `assets/dataset_cropped/data_cropped.yaml`.

### Step 2: Run Training (Stage 2 Model)
To train your custom PPE detector model on the newly prepared cropped person dataset:
```powershell
python src/train.py
```
This runs for 100 epochs (fully configurable). Once finished, the best weights are copied to `models/best_ppe.pt` and training evaluation dashboards are generated at `models/training_graphs/`.

### Step 3: Run Inference (Edge Device Loop)
Start real-time monitoring on your webcam or a video file by executing:
```powershell
python src/main.py
```
Press **'Q'** on the camera window to safely close the inference loop.
