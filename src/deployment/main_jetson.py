"""
=============================================================================
PROTMIND — main_jetson.py
Inference Engine untuk Pengujian & Produksi khusus NVIDIA Jetson Nano
=============================================================================
"""

import os
import sys
import time
import queue
import logging
import platform
import threading
import subprocess
import gc
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO
from dotenv import load_dotenv

# Impor pustaka native TensorRT & PyCUDA secara aman (defensive imports)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError as e:
    TRT_AVAILABLE = False
    TRT_ERROR = str(e)

# Muat variabel lingkungan
load_dotenv()

# Bersihkan cache VRAM GPU pada startup untuk arsitektur RAM bersama Jetson Nano
try:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 'expandable_segments' hanya didukung di PyTorch >= 2.0
        torch_ver = torch.__version__.split('.')
        if len(torch_ver) > 0 and torch_ver[0].isdigit() and int(torch_ver[0]) >= 2:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
except Exception:
    pass

# Helper classes untuk meniru struktur output Ultralytics YOLO API
class Detection(object):
    def __init__(self, class_name, confidence, x1, y1, x2, y2):
        self.class_name = class_name
        self.confidence = confidence
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

class NumpyWrapper(object):
    def __init__(self, val):
        self.val = val
    def cpu(self):
        return self
    def numpy(self):
        return self.val

class MockBoxPPE(object):
    def __init__(self, c, cf, xywhn_val):
        self.cls = NumpyWrapper(np.array([c]))
        self.conf = NumpyWrapper(np.array([cf]))
        self.xywhn = NumpyWrapper(np.array([xywhn_val]))

class MockBoxesPerson(object):
    def __init__(self, xyxy_arr):
        self.xyxy = self
        self.arr = xyxy_arr
    def cpu(self):
        return self
    def numpy(self):
        return self.arr

class MockResult(object):
    def __init__(self, boxes):
        self.boxes = boxes

class YoloTRT(object):
    def __init__(self, engine_path, names):
        self.names = names
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding)})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': self.engine.get_binding_shape(binding)})

    def predict(self, frame, imgsz, conf_thres, iou_thres):
        img, r, dw, dh = self._preprocess(frame, (imgsz, imgsz))
        
        np.copyto(self.inputs[0]['host'], img.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        self.stream.synchronize()
        
        out_shape = self.outputs[0]['shape']
        output = self.outputs[0]['host'].reshape(out_shape)
        
        return self._postprocess(output, r, dw, dh, conf_thres, iou_thres)

    def _preprocess(self, img, input_size):
        shape = img.shape[:2]
        r = min(input_size[0] / shape[0], input_size[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = input_size[0] - new_unpad[0], input_size[1] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        return np.expand_dims(img, axis=0), r, left, top

    def _postprocess(self, output, r, dw, dh, conf_threshold, iou_threshold):
        output = output[0].T
        boxes = output[:, :4]
        scores = output[:, 4:]
        
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        
        mask = confidences > conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []
            
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        x1 = (x1 - dw) / r
        y1 = (y1 - dh) / r
        x2 = (x2 - dw) / r
        y2 = (y2 - dh) / r
        
        boxes_xywh = np.stack([x1, y1, x2-x1, y2-y1], axis=1)
        indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), confidences.tolist(), conf_threshold, iou_threshold)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append(Detection(
                    class_name=self.names.get(int(class_ids[i]), "unknown"),
                    confidence=float(confidences[i]),
                    x1=int(x1[i]), y1=int(y1[i]), x2=int(x2[i]), y2=int(y2[i])
                ))
        return results

class YoloTRTWrapper(object):
    def __init__(self, engine_path, names, is_person=True):
        self.trt_model = YoloTRT(engine_path, names)
        self.is_person = is_person
        self.names = names
        self.names_reverse = {v: k for k, v in names.items()}

    def __call__(self, frame, conf=0.25, iou=0.45, classes=None, verbose=False):
        detections = self.trt_model.predict(frame, 640, conf, iou)
        
        if self.is_person:
            xyxy_list = []
            for det in detections:
                if classes is not None and det.class_name != "person":
                    continue
                xyxy_list.append([det.x1, det.y1, det.x2, det.y2, det.confidence, 0])
            if not xyxy_list:
                xyxy_array = np.zeros((0, 6), dtype=np.float32)
            else:
                xyxy_array = np.array(xyxy_list, dtype=np.float32)
            return [MockResult(MockBoxesPerson(xyxy_array))]
        else:
            boxes = []
            h, w = frame.shape[:2]
            for det in detections:
                cls_id = self.names_reverse.get(det.class_name, 0)
                box_w = det.x2 - det.x1
                box_h = det.y2 - det.y1
                cx = det.x1 + box_w / 2.0
                cy = det.y1 + box_h / 2.0
                
                cx_n = cx / w if w > 0 else 0.0
                cy_n = cy / h if h > 0 else 0.0
                w_n = box_w / w if w > 0 else 0.0
                h_n = box_h / h if h > 0 else 0.0
                
                boxes.append(MockBoxPPE(cls_id, det.confidence, [cx_n, cy_n, w_n, h_n]))
            return [MockResult(boxes)]

def load_yolo_model(model_path, names, is_person=True):
    if model_path.endswith(".engine") and TRT_AVAILABLE:
        log.info(f"  [Detector] Memuat native TensorRT engine: {model_path}")
        return YoloTRTWrapper(model_path, names, is_person=is_person)
    else:
        log.info(f"  [Detector] Memuat standard YOLO model: {model_path}")
        return YOLO(model_path)

# Tentukan model secara dinamis (memprioritaskan TensorRT .engine jika CUDA/TRT aktif, lalu .pt)
is_python_36 = sys.version_info < (3, 7)
use_trt_engine = TRT_AVAILABLE and torch.cuda.is_available()

if use_trt_engine:
    model_person_path = "models/best_person.engine" if Path("models/best_person.engine").exists() else ("models/best_person.pt" if Path("models/best_person.pt").exists() else "yolov8n.pt")
    model_ppe_path = "models/best_ppe.engine" if Path("models/best_ppe.engine").exists() else ("models/best_ppe.pt" if Path("models/best_ppe.pt").exists() else "models/best_ppe_20260710_022641.pt")
else:
    # Fallback jika tidak ada CUDA/TRT:
    # Di Python 3.6, model .pt baru mengalami pickle error, jadi gunakan .onnx jika ada
    if is_python_36:
        model_person_path = "models/best_person.onnx" if Path("models/best_person.onnx").exists() else "models/best_person.pt"
        model_ppe_path = "models/best_ppe.onnx" if Path("models/best_ppe.onnx").exists() else "models/best_ppe.pt"
    else:
        model_person_path = "models/best_person.pt" if Path("models/best_person.pt").exists() else "yolov8n.pt"
        model_ppe_path = "models/best_ppe.pt" if Path("models/best_ppe.pt").exists() else "models/best_ppe_20260710_022641.pt"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "telegram_token"    : os.getenv("TELEGRAM_TOKEN", ""),
    "telegram_chat_id"  : os.getenv("TELEGRAM_CHAT_ID", ""),
    "camera_source"     : int(os.getenv("CAMERA_SOURCE", "0")) if os.getenv("CAMERA_SOURCE", "0").isdigit() else os.getenv("CAMERA_SOURCE", "0"),
    "lokasi_kamera"     : os.getenv("LOKASI_KAMERA", "Gate 1 - Jetson Nano Test"),

    "model_person"      : model_person_path,
    "model_ppe"         : model_ppe_path,

    "conf_person"       : 0.50,
    "conf_ppe"          : 0.50,
    "cooldown_time"     : int(os.getenv("COOLDOWN_TIME", "10")),
    "audio_delay"       : int(os.getenv("AUDIO_DELAY", "3")),
    
    "camera_vfov"       : float(os.getenv("CAMERA_VFOV", "48.0")),
    "audio_helm"        : "assets/audio/peringatan_helm.wav",
    "audio_rompi"       : "assets/audio/peringatan_rompi.wav",
    "audio_sepatu"      : "assets/audio/peringatan_sepatu.wav",
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "[%(asctime)s] %(levelname)s — %(message)s",
    datefmt = "%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("inference_jetson.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("PROTMIND-JETSON")

# ─────────────────────────────────────────────────────────────────────────────
# DETEKSI KARTU SUARA USB (LINUX ALSA)
# ─────────────────────────────────────────────────────────────────────────────
def find_usb_audio_card():
    """Mencari index kartu suara USB terbaik dari /proc/asound/cards."""
    cards_file = Path("/proc/asound/cards")
    if not cards_file.exists():
        return None
    try:
        with open(cards_file, "r") as f:
            content = f.read()

        cards = {}
        current_card_idx = None

        for line in content.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            # Jika diawali angka indeks (ALSA format), inisialisasi kartu baru
            if parts[0].isdigit():
                current_card_idx = int(parts[0])
                cards[current_card_idx] = line.lower()
            elif current_card_idx is not None:
                # Gabungkan deskripsi baris berikutnya ke kartu suara aktif
                cards[current_card_idx] += " " + line.lower()

        best_card_idx = None
        best_score = -9999

        for card_idx, info in cards.items():
            score = 0
            if "usb audio" in info:
                score += 100
            if "ab13x" in info:
                score += 100
            if "essager" in info:
                score += 100
            if "generic" in info:
                score += 50
            if "usb-audio" in info or "usb" in info or "audio" in info:
                score += 10

            # Berikan penalti berat jika perangkat merupakan webcam/mic kamera (seperti JETE-W7)
            if any(k in info for k in ["jete", "camera", "webcam", "mic", "microphone"]):
                score -= 150

            if score > best_score and score > 0:
                best_score = score
                best_card_idx = card_idx

        return best_card_idx
    except Exception:
        pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
# MAINKAN AUDIO LINTAS PLATFORM
# ─────────────────────────────────────────────────────────────────────────────
def play_audio(wav_path: str) -> None:
    """Memutar file audio secara asinkron tanpa memblokir stream kamera."""
    wav_file = Path(wav_path)
    if not wav_file.exists():
        log.warning(f"  [Audio] File suara tidak ditemukan: {wav_file.resolve()}")
        return

    system_name = platform.system()
    if system_name == "Windows":
        try:
            import winsound
            winsound.PlaySound(str(wav_file), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            subprocess.Popen(
                ["powershell", "-c", f"(New-Object Media.SoundPlayer '{wav_file.resolve()}').Play()"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    elif system_name == "Linux":
        card_idx = find_usb_audio_card()
        if card_idx is not None:
            # Menggunakan plughw agar format & sample rate disesuaikan otomatis oleh ALSA
            cmd = ["aplay", "-D", f"plughw:{card_idx},0", str(wav_file.resolve())]
            log.info(f"  [Audio] Menjalankan aplay ke USB Sound Card (hw:{card_idx})")
        else:
            cmd = ["aplay", str(wav_file.resolve())]
            log.info(f"  [Audio] Menjalankan aplay default")

        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            try:
                subprocess.Popen(["paplay", str(wav_file.resolve())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                log.error(f"  [Audio] Gagal memutar audio di Linux: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC DISTANCE ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────
def estimate_distance(px1: int, py1: int, px2: int, py2: int, frame_width: float, frame_height: float) -> float:
    """
    Mengestimasi jarak monokular ke pekerja secara dinamis:
    - Jika pekerja terpotong bingkai atas/bawah (sangat dekat), gunakan lebar bahu (width-based).
    - Jika pekerja terlihat utuh di dalam bingkai, gunakan tinggi badan (height-based).
    """
    try:
        if frame_width <= 0 or frame_height <= 0:
            return 0.0
            
        p_w = px2 - px1
        p_h = py2 - py1
        
        if p_w <= 0 or p_h <= 0:
            return 0.0

        h_actual = 1.7   # Tinggi rata-rata (meter)
        w_actual = 0.45  # Lebar bahu rata-rata (meter)
        
        hfov = 60.0
        vfov = float(CONFIG.get("camera_vfov", 48.0))
        
        is_cut_off = (py1 <= 5) or (py2 >= frame_height - 5)
        
        if is_cut_off:
            # Gunakan lebar bahu (width-based)
            hfov_rad = np.radians(hfov)
            focal_length_hx = frame_width / (2.0 * np.tan(hfov_rad / 2.0))
            distance = (focal_length_hx * w_actual) / p_w
        else:
            # Gunakan tinggi badan (height-based)
            vfov_rad = np.radians(vfov)
            focal_length_px = frame_height / (2.0 * np.tan(vfov_rad / 2.0))
            distance = (focal_length_px * h_actual) / p_h
            
        return round(distance, 2)
    except Exception as e:
        log.error(f"  [Distance] Gagal mengestimasi jarak: {e}")
        return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM METRICS
# ─────────────────────────────────────────────────────────────────────────────
def get_system_metrics():
    import psutil
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    gpu_usage = 0.0
    for path in [
        "/sys/devices/gpu.0/load",
        "/sys/class/devfreq/gpu.0/device/load",
        "/sys/devices/platform/host1x/17000000.gp10b/load"
    ]:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    val = int(f.read().strip())
                    gpu_usage = val / 10.0 if val > 100 else float(val)
                    break
            except Exception:
                pass
    return cpu_usage, gpu_usage, ram_usage

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND ALARM WORKER
# ─────────────────────────────────────────────────────────────────────────────
warning_queue = queue.Queue()
exit_event = threading.Event()

def get_combination_audio(violations):
    has_helmet = "no-helmet" in violations
    has_vest = "no-vest" in violations
    has_shoes = "no-safety shoes" in violations
    
    bitmask = (1 if has_helmet else 0) << 2 | (1 if has_vest else 0) << 1 | (1 if has_shoes else 0)
    if bitmask == 0:
        return None
        
    mapping = {
        0b100: CONFIG["audio_helm"],
        0b010: CONFIG["audio_rompi"],
        0b001: CONFIG["audio_sepatu"],
        0b110: "assets/audio/peringatan_helm_rompi.wav",
        0b101: "assets/audio/peringatan_helm_sepatu.wav",
        0b011: "assets/audio/peringatan_rompi_sepatu.wav",
        0b111: "assets/audio/peringatan_lengkap.wav",
    }
    
    audio_path = mapping.get(bitmask)
    if audio_path and Path(audio_path).exists():
        return audio_path
        
    fallback_path = "assets/audio/peringatan_umum.wav"
    return fallback_path if Path(fallback_path).exists() else None

def warning_worker():
    while not exit_event.is_set():
        try:
            task = warning_queue.get(timeout=0.5)
        except queue.Empty:
            continue
            
        if task is None:
            break
            
        violations_to_play, text_report, annotated_frame = task
        
        # 1. Kirim Laporan ke Telegram
        if CONFIG["telegram_token"] and CONFIG["telegram_chat_id"]:
            log.info("  [Telegram] Mengirim notifikasi alarm pelanggaran APD...")
            token = CONFIG["telegram_token"]
            chat_id = CONFIG["telegram_chat_id"]
            url = f"https://api.telegram.org/bot{token}/sendPhoto"
            
            success, img_encoded = cv2.imencode(".jpg", annotated_frame)
            if success:
                files = {"photo": ("alarm.jpg", img_encoded.tobytes(), "image/jpeg")}
                data = {"chat_id": chat_id, "caption": text_report, "parse_mode": "HTML"}
                try:
                    res = requests.post(url, data=data, files=files, timeout=10)
                    if res.status_code == 200:
                        log.info("  [Telegram] Berhasil mengirim laporan alarm!")
                    else:
                        log.error(f"  [Telegram] Gagal: {res.text}")
                except Exception as e:
                    log.error(f"  [Telegram] Error koneksi: {e}")
        else:
            log.warning("  [Telegram] Kredensial Telegram dilewati (tidak diset).")

        # 2. Putar Suara Warning Lintas Platform
        audio_file = get_combination_audio(violations_to_play)
        if audio_file:
            log.info(f"  [Audio Warning] Memutar peringatan kombinasi: {Path(audio_file).name}...")
            play_audio(audio_file)
            delay = CONFIG["audio_delay"]
            steps = int(delay * 10)
            for _ in range(steps):
                if exit_event.is_set():
                    break
                time.sleep(0.1)
            
        warning_queue.task_done()

worker_thread = threading.Thread(target=warning_worker, daemon=True)
worker_thread.start()

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="PROTMIND APD System — Inference Engine khusus Jetson Nano")
    parser.add_argument("--source", type=str, default=None, help="Sumber kamera (angka/path video).")
    parser.add_argument("--csi", action="store_true", help="Gunakan kamera CSI Jetson Nano via GStreamer nvarguscamerasrc.")
    parser.add_argument("--skip-frames", type=int, default=2, help="Interval skip frame inferensi untuk hemat CPU/GPU (default: 2).")
    parser.add_argument("--headless", action="store_true", help="Jalankan inferensi tanpa menampilkan window GUI (cv2.imshow).")
    parser.add_argument("--conf-person", type=float, default=None, help="Conf threshold untuk model Person.")
    parser.add_argument("--conf-ppe", type=float, default=None, help="Conf threshold untuk model PPE.")
    parser.add_argument("--test-audio", action="store_true", help="Uji coba pemutaran audio pada kartu suara USB.")
    args = parser.parse_args()

    # Uji coba audio cepat tanpa meload model/kamera
    if args.test_audio:
        log.info("=" * 60)
        log.info("  PROTMIND APD SYSTEM — UJI COBA AUDIO SPEAKER")
        log.info("=" * 60)
        test_file = CONFIG["audio_helm"]
        log.info(f"Mencoba memutar audio tes: {test_file}")
        play_audio(test_file)
        # Tunggu beberapa detik agar aplay (asinkron) selesai memutar suara sebelum program exit
        time.sleep(5)
        log.info("Uji coba audio selesai.")
        sys.exit(0)

    # Terapkan argumen CLI ke CONFIG
    if args.conf_person is not None: CONFIG["conf_person"] = args.conf_person
    if args.conf_ppe is not None: CONFIG["conf_ppe"] = args.conf_ppe
    if args.headless: CONFIG["headless"] = True

    log.info("=" * 60)
    log.info("  PROTMIND APD SYSTEM — JETSON INFERENCE TEST ENGINE")
    log.info("=" * 60)
    log.info(f"  Stage 1 (Person) : {CONFIG['model_person']}")
    log.info(f"  Stage 2 (PPE)    : {CONFIG['model_ppe']}")
    
    # Deteksi kesiapan model
    if not Path(CONFIG["model_person"]).exists():
        log.warning(f"  [Warning] Model Person '{CONFIG['model_person']}' tidak ditemukan. Menggunakan fallback yolov8n.pt")
        CONFIG["model_person"] = "yolov8n.pt"
    if not Path(CONFIG["model_ppe"]).exists():
        log.error(f"  File model PPE kustom '{CONFIG['model_ppe']}' tidak ditemukan.")
        sys.exit(1)

    # Inisialisasi model secara dinamis (mendukung YOLO biasa dan Native TRT Engine)
    model_p = load_yolo_model(CONFIG["model_person"], {0: "person"}, is_person=True)
    model_ppe = load_yolo_model(CONFIG["model_ppe"], {
        0: "helmet",
        1: "no-helmet",
        2: "no-safety shoes",
        3: "no-vest",
        4: "safety shoes",
        5: "vest"
    }, is_person=False)
    
    # Class map untuk APD
    ppe_classes = {
        0: ("helmet", True),
        1: ("no-helmet", False),
        2: ("no-safety shoes", False),
        3: ("no-vest", False),
        4: ("safety shoes", True),
        5: ("vest", True)
    }

    last_played = {"no-helmet": 0.0, "no-vest": 0.0, "no-safety shoes": 0.0}
    
    # Konfigurasi capture source
    source = args.source if args.source is not None else CONFIG["camera_source"]
    if args.source is not None and args.source.isdigit():
        source = int(args.source)

    # Inisialisasi kamera dengan opsi GStreamer Jetson
    if args.csi:
        # Pipeline optimal untuk kamera CSI Jetson Nano (resolusi 640x480)
        gstreamer_pipeline = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
            "nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false"
        )
        log.info(f"  [Camera] Mengaktifkan Jetson CSI Camera GStreamer Pipeline: {gstreamer_pipeline}")
        cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
    else:
        opened = False
        if platform.system() == "Linux" and isinstance(source, int):
            try:
                # Pipeline optimal untuk kamera USB di Jetson Nano Linux
                gstreamer_pipeline = (
                    f"v4l2src device=/dev/video{source} ! "
                    "video/x-raw, width=640, height=480, format=YUY2, framerate=30/1 ! "
                    "videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false"
                )
                log.info(f"  [Camera] Mencoba USB GStreamer Pipeline: {gstreamer_pipeline}")
                cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    opened = True
            except Exception:
                pass
                
        if not opened:
            log.info(f"  [Camera] Menggunakan OpenCV standard backend untuk source: {source}")
            cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        log.error(f"  Gagal membuka kamera / video source: {source}")
        sys.exit(1)

    log.info(f"  Kamera/Video berhasil terbuka. Sumber: {source}")
    log.info(f"  Frame Skipping: tiap {args.skip_frames} frame sekali.")
    log.info("-" * 60)

    frame_count = 0
    prev_frame_time = time.time()
    current_fps_frame = 0.0
    current_fps_inf = 0.0
    
    # Untuk perhitungan FPS Frame metode Window (akurat akademik)
    fps_window_counter = 0
    t_window_start = time.time()
    
    # Inisialisasi visual cache deteksi
    cached_detections = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                log.info("  Selesai memproses video (EOF) atau kamera terputus.")
                break
                
            frame_count += 1
            t_inf_start = time.time()
            annotated_frame = frame.copy()
            H, W, _ = frame.shape
            
            # Cek apakah harus menjalankan inferensi YOLO di frame ini
            run_inference = (frame_count % args.skip_frames == 0) or (frame_count == 1)
            
            # Reset penampung pelanggaran frame saat inferensi dijalankan
            frame_violations = []
            melanggar_count = 0
            lengkap_count = 0
            
            no_helmet_count = 0
            no_vest_count = 0
            no_shoes_count = 0
            
            t_pure_inf = 0.0
            if run_inference:
                t_model_start = time.time()
                # Bersihkan cache deteksi lama
                cached_detections = []
                
                # Jalankan Stage 1: Person Detection
                results_p = model_p(frame, classes=0, conf=CONFIG["conf_person"], verbose=False)
                person_boxes = results_p[0].boxes.xyxy.cpu().numpy()
                
                for p_idx, p_box in enumerate(person_boxes):
                    px1, py1, px2, py2 = map(int, p_box[:4])
                    px1, py1 = max(0, px1), max(0, py1)
                    px2, py2 = min(W, px2), min(H, py2)
                    p_w, p_h = px2 - px1, py2 - py1
                    
                    if p_w < 20 or p_h < 20:
                        continue
                        
                    # Crop person dari frame asli
                    person_crop = frame[py1:py2, px1:px2]
                    
                    # Jalankan Stage 2: APD Classifier/Detector
                    results_ppe = model_ppe(person_crop, conf=CONFIG["conf_ppe"], verbose=False)
                    ppe_boxes = results_ppe[0].boxes
                    
                    person_has_violation = False
                    person_violations = []
                    ppe_draw_list = []
                    
                    for ppe_box in ppe_boxes:
                        cls_id = int(ppe_box.cls.cpu().numpy()[0])
                        cls_conf = float(ppe_box.conf.cpu().numpy()[0])
                        cx_c, cy_c, w_c, h_c = ppe_box.xywhn.cpu().numpy()[0]
                        
                        cls_name, is_compliant = ppe_classes.get(cls_id, ("unknown", True))
                        
                        px_c_abs = cx_c * p_w + px1
                        py_c_abs = cy_c * p_h + py1
                        pw_abs = w_c * p_w
                        ph_abs = h_c * p_h
                        
                        cx1, cy1 = int(px_c_abs - pw_abs/2), int(py_c_abs - ph_abs/2)
                        cx2, cy2 = int(px_c_abs + pw_abs/2), int(py_c_abs + ph_abs/2)
                        
                        if not is_compliant:
                            person_has_violation = True
                            person_violations.append(cls_name)
                            if cls_name not in frame_violations:
                                frame_violations.append(cls_name)
                                
                            if cls_name == "no-helmet": no_helmet_count += 1
                            elif cls_name == "no-vest": no_vest_count += 1
                            elif cls_name == "no-safety shoes": no_shoes_count += 1
                            
                        ppe_draw_list.append((cx1, cy1, cx2, cy2, cls_name, cls_conf, is_compliant))
                        
                    # Estimasi Jarak secara dinamis (self-calibrating)
                    worker_distance = estimate_distance(px1, py1, px2, py2, W, H)
                    
                    if person_has_violation:
                        melanggar_count += 1
                    else:
                        lengkap_count += 1
                        
                    # Simpan hasil deteksi ke cache
                    cached_detections.append({
                        "p_box": (px1, py1, px2, py2),
                        "violations": person_violations,
                        "has_violation": person_has_violation,
                        "distance": worker_distance,
                        "ppe_draw": ppe_draw_list
                    })
                t_pure_inf = time.time() - t_model_start
                    
            # Terapkan Anotasi Visual dari Bounding Box Cache (Dukungan Frame Skip)
            for det in cached_detections:
                px1, py1, px2, py2 = det["p_box"]
                p_color = (0, 0, 255) if det["has_violation"] else (0, 255, 0)
                
                cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), p_color, 3)
                
                status_text = f"Pekerja ({det['distance']}m): "
                if det["has_violation"]:
                    status_text += f"MELANGGAR ({', '.join(det['violations'])})"
                else:
                    status_text += "LENGKAP"
                    
                cv2.putText(
                    annotated_frame, status_text, (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, p_color, 2
                )
                
                # Gambar box APD kustom
                for cx1, cy1, cx2, cy2, cls_name, cls_conf, is_compliant in det["ppe_draw"]:
                    color = (0, 255, 0) if is_compliant else (0, 0, 255)
                    cv2.rectangle(annotated_frame, (cx1, cy1), (cx2, cy2), color, 2)
                    cv2.putText(
                        annotated_frame, f"{cls_name} {cls_conf:.2f}", (cx1, cy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
                    )

            # Simpan gambar hasil deteksi jika ada pekerja terdeteksi
            if run_inference and len(person_boxes) > 0:
                save_dir = Path("detections")
                save_dir.mkdir(parents=True, exist_ok=True)
                
                has_any_violation = any(det["has_violation"] for det in cached_detections)
                status_str = "melanggar" if has_any_violation else "lengkap"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                
                save_path = save_dir / f"deteksi_{timestamp}_{status_str}.jpg"
                cv2.imwrite(str(save_path), annotated_frame)

            # Hitung FPS Frame menggunakan metode Window 30 frame (sangat akurat secara akademik)
            fps_window_counter += 1
            if fps_window_counter >= 30:
                t_window_end = time.time()
                t_elapsed = t_window_end - t_window_start
                current_fps_frame = 30.0 / t_elapsed if t_elapsed > 0 else 0.0
                
                # Reset window
                fps_window_counter = 0
                t_window_start = t_window_end
            elif frame_count == 1:
                t_frame_interval = time.time() - prev_frame_time
                current_fps_frame = 1.0 / t_frame_interval if t_frame_interval > 0 else 0.0
                current_fps_inf = 0.0
                
            # Update FPS Inferensi HANYA jika inferensi murni berjalan pada frame ini
            alpha = 0.1
            if run_inference and t_pure_inf > 0.0:
                fps_inf = 1.0 / t_pure_inf
                if current_fps_inf == 0.0:
                    current_fps_inf = fps_inf
                else:
                    current_fps_inf = alpha * fps_inf + (1.0 - alpha) * current_fps_inf
                
            # Ambil metrik resource sistem
            cpu_pct, gpu_pct, ram_pct = get_system_metrics()
            
            if frame_count % 30 == 0:
                log.info(
                    f"  [Jetson Perf] FPS Frame: {current_fps_frame:.2f} | FPS Inf: {current_fps_inf:.2f} | "
                    f"CPU: {cpu_pct:.1f}% | GPU: {gpu_pct:.1f}% | RAM: {ram_pct:.1f}%"
                )

            # ─────────────────────────────────────────────────────────────────────
            # ALARM MANAGER (Hanya dipicu saat inferensi baru dijalankan)
            # ─────────────────────────────────────────────────────────────────────
            if run_inference:
                current_time = time.time()
                violations_to_alert = []
                
                # Kumpulkan pelanggaran yang sudah melewati masa cooldown 10 detik
                for violation in frame_violations:
                    if current_time - last_played[violation] >= CONFIG["cooldown_time"]:
                        violations_to_alert.append(violation)
                        last_played[violation] = current_time
                        
                if violations_to_alert:
                    log.info(f"  [Alarm!] Terdeteksi Pelanggaran Baru: {violations_to_alert}")
                    waktu_str = datetime.now().strftime("%d %B %Y, %H:%M:%S") + " WIB"
                    
                    warning_text = (
                        f"🚨 <b>ALARM APD JETSON NANO</b> 🚨\n\n"
                        f"📍 <b>Kamera:</b> {CONFIG['lokasi_kamera']}\n"
                        f"🕒 <b>Waktu:</b> {waktu_str}\n\n"
                        f"👥 Pekerja Terdeteksi: {len(cached_detections)} Orang\n"
                        f"✅ Lengkap: {lengkap_count} Orang\n"
                        f"❌ Melanggar: {melanggar_count} Orang\n\n"
                        f"⚠️ <b>Rincian Pelanggaran:</b>\n"
                    )
                    
                    if no_helmet_count > 0:
                        warning_text += f"• {no_helmet_count} Orang : Tanpa Helm\n"
                    if no_vest_count > 0:
                        warning_text += f"• {no_vest_count} Orang : Tanpa Rompi\n"
                    if no_shoes_count > 0:
                        warning_text += f"• {no_shoes_count} Orang : Tanpa Sepatu Safety\n"
                        
                    warning_text += f"\nSegera lakukan inspeksi visual di lokasi konstruksi."
                    warning_queue.put((violations_to_alert, warning_text, annotated_frame.copy()))



            # Render overlay teks visual
            cv2.putText(annotated_frame, f"Lokasi: {CONFIG['lokasi_kamera']}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Pekerja: {len(cached_detections)} | Skip: {args.skip_frames}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"FPS Frame: {current_fps_frame:.1f} | FPS Inf: {current_fps_inf:.1f}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"CPU: {cpu_pct:.1f}% | GPU: {gpu_pct:.1f}% | RAM: {ram_pct:.1f}%", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # GUI window rendering
            if not CONFIG.get("headless", False):
                try:
                    cv2.imshow("PROTMIND APD Detection - Jetson Nano Test", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        log.info("  Keluar atas permintaan pengguna.")
                        break
                except cv2.error:
                    log.warning("  [GUI Warning] Jendela tampilan tidak didukung. Beralih ke headless.")
                    CONFIG["headless"] = True
                    time.sleep(0.001)
            else:
                time.sleep(0.001)
                
    except KeyboardInterrupt:
        log.info("  Program dihentikan oleh pengguna (Ctrl+C).")
    finally:
        exit_event.set()
        try: cap.release()
        except Exception: pass
        try: cv2.destroyAllWindows()
        except Exception: pass
        
        try: warning_queue.put(None)
        except Exception: pass
        worker_thread.join(timeout=1.0)
        log.info("  Sistem dinonaktifkan dengan sukses.")

if __name__ == "__main__":
    main()
