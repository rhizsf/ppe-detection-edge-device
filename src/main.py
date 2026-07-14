"""
=============================================================================
PROTMIND — main.py
Sistem Deteksi APD Real-Time & Peringatan Edge Device | Jetson Nano / Laptop
=============================================================================
Fungsi   : Mengambil video feed, mendeteksi person, memotong gambar person,
           mendeteksi kelengkapan APD (Tahap 2), memutar suara peringatan 
           Round-Robin jika melanggar dengan sistem cooldown, dan mengirim 
           laporan alarm lengkap dengan visual ke Telegram.
Hardware : NVIDIA Jetson Nano / Laptop dengan Webcam/Video File
Output   : Pemutaran audio + Notifikasi Telegram + Window Tampilan Frame
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
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import requests
from ultralytics import YOLO
from dotenv import load_dotenv

# Muat variabel lingkungan dari file .env
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Kredensial Telegram Bot (diambil dari file .env atau fallback ke string kosong)
    "telegram_token"    : os.getenv("TELEGRAM_TOKEN", ""),
    "telegram_chat_id"  : os.getenv("TELEGRAM_CHAT_ID", ""),

    # Parameter Lokasi & Kamera (diambil dari .env)
    "camera_source"     : int(os.getenv("CAMERA_SOURCE", "0")) if os.getenv("CAMERA_SOURCE", "0").isdigit() else os.getenv("CAMERA_SOURCE", "0"),
    "lokasi_kamera"     : os.getenv("LOKASI_KAMERA", "Gate 1 - Area Proyek Alpha"),

    # Konfigurasi Model
    "model_person"      : "models/best_person.pt" if Path("models/best_person.pt").exists() else "yolov8n.pt",         # Model Stage 1 kustom (fallback ke pre-trained YOLOv8n)
    "model_ppe"         : "models/best_ppe.pt" if Path("models/best_ppe.pt").exists() else "models/best_ppe_20260607_223158.pt", # Model Stage 2 kustom (fallback ke checkpoint lama)

    # Nilai Ambang Batas Kepercayaan (Confidence Threshold)
    "conf_person"       : 0.50,
    "conf_ppe"          : 0.50,

    # Manajemen Alarm & Audio
    "cooldown_time"     : int(os.getenv("COOLDOWN_TIME", "10")), # Jeda waktu minimum (detik) alarm sejenis tidak boleh berulang-ulang
    "audio_delay"       : int(os.getenv("AUDIO_DELAY", "3")),   # Jeda waktu (detik) pemutaran audio bergiliran (Round-Robin)

    # Kalibrasi Kamera untuk Jarak
    "camera_height"     : float(os.getenv("CAMERA_HEIGHT", "1.0")),
    "camera_tilt"       : float(os.getenv("CAMERA_TILT", "0.0")),
    "camera_vfov"       : float(os.getenv("CAMERA_VFOV", "48.0")),

    # Path Audio Peringatan
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
        logging.FileHandler("inference.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("PROTMIND-MAIN")

# ─────────────────────────────────────────────────────────────────────────────
# MAINKAN AUDIO LINTAS PLATFORM (Windows & Linux / Jetson Nano)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# FUNGSI PENDUKUNG ESTIMASI JARAK & METRIK SISTEM & KOMBINASI AUDIO
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

        # Parameter fisik rata-rata manusia
        h_actual = 1.7   # Tinggi rata-rata (meter)
        w_actual = 0.45  # Lebar bahu rata-rata (meter)
        
        # Spesifikasi FOV kamera (default: HFOV 60, VFOV 48)
        hfov = 60.0
        vfov = float(CONFIG.get("camera_vfov", 48.0))
        
        # Cek apakah bounding box menyentuh batas atas atau bawah frame (indikasi terpotong karena dekat)
        # Berikan margin toleransi 5 piksel
        is_cut_off = (py1 <= 5) or (py2 >= frame_height - 5)
        
        if is_cut_off:
            # Gunakan estimasi berbasis LEBAR (width-based)
            hfov_rad = np.radians(hfov)
            focal_length_hx = frame_width / (2.0 * np.tan(hfov_rad / 2.0))
            distance = (focal_length_hx * w_actual) / p_w
        else:
            # Gunakan estimasi berbasis TINGGI (height-based)
            vfov_rad = np.radians(vfov)
            focal_length_px = frame_height / (2.0 * np.tan(vfov_rad / 2.0))
            distance = (focal_length_px * h_actual) / p_h
            
        return round(distance, 2)
    except Exception as e:
        log.error(f"  [Distance] Gagal mengestimasi jarak: {e}")
        return 0.0


def get_system_metrics():
    """
    Mendapatkan pemakaian CPU, GPU, dan RAM secara real-time.
    Mendukung NVIDIA Jetson Nano dan PC standar.
    """
    import psutil
    
    # 1. CPU Usage (%)
    cpu_usage = psutil.cpu_percent()
    
    # 2. RAM Usage (%)
    ram_usage = psutil.virtual_memory().percent
    
    # 3. GPU Usage (%)
    gpu_usage = 0.0
    # Coba baca tegra GPU load untuk Jetson Nano
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


def get_combination_audio(violations):
    """
    Memetakan list pelanggaran APD ke berkas audio .wav kombinasi terintegrasi (8 variasi).
    """
    has_helmet = "no-helmet" in violations
    has_vest = "no-vest" in violations
    has_shoes = "no-safety shoes" in violations
    
    # Buat bitmask: [Helm, Rompi, Sepatu]
    bitmask = (1 if has_helmet else 0) << 2 | (1 if has_vest else 0) << 1 | (1 if has_shoes else 0)
    
    if bitmask == 0:
        return None
        
    mapping = {
        0b100: "assets/audio/peringatan_helm.wav",
        0b010: "assets/audio/peringatan_rompi.wav",
        0b001: "assets/audio/peringatan_sepatu.wav",
        0b110: "assets/audio/peringatan_helm_rompi.wav",
        0b101: "assets/audio/peringatan_helm_sepatu.wav",
        0b011: "assets/audio/peringatan_rompi_sepatu.wav",
        0b111: "assets/audio/peringatan_lengkap.wav",
    }
    
    audio_path = mapping.get(bitmask)
    if audio_path and Path(audio_path).exists():
        return audio_path
        
    fallback_path = "assets/audio/peringatan_umum.wav"
    if Path(fallback_path).exists():
        return fallback_path
        
    return None


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


def play_audio(wav_path: str) -> None:
    """Memutar file audio secara asinkron tanpa menghentikan frame-rate video."""
    wav_file = Path(wav_path)
    if not wav_file.exists():
        log.warning(f"  [Audio] File suara tidak ditemukan: {wav_file.resolve()}")
        return

    system_name = platform.system()
    if system_name == "Windows":
        try:
            import winsound
            # SND_ASYNC agar tidak memblokir thread yang sedang memutar audio
            winsound.PlaySound(str(wav_file), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            # Fallback jika winsound bermasalah
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
# BACKGROUND QUEUE WORKER
# ─────────────────────────────────────────────────────────────────────────────
# Menggunakan antrean asinkron agar proses pemutaran suara jeda 3 detik 
# dan pengiriman Telegram tidak menyebabkan lag atau stuttering pada stream kamera.
telegram_queue = queue.Queue()
audio_lock = threading.Lock()
latest_audio_task = None
exit_event = threading.Event()

def play_audio_blocking(wav_path: str) -> None:
    """Memutar file audio secara sinkron (blocking) pada thread audio worker."""
    wav_file = Path(wav_path)
    if not wav_file.exists():
        log.warning(f"  [Audio] File suara tidak ditemukan: {wav_file.resolve()}")
        return

    system_name = platform.system()
    if system_name == "Windows":
        try:
            import winsound
            winsound.PlaySound(str(wav_file), winsound.SND_FILENAME)
        except Exception:
            subprocess.run(
                ["powershell", "-c", f"(New-Object Media.SoundPlayer '{wav_file.resolve()}').PlaySync()"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    elif system_name == "Linux":
        card_idx = find_usb_audio_card()
        if card_idx is not None:
            cmd = ["aplay", "-D", f"plughw:{card_idx},0", str(wav_file.resolve())]
            log.info(f"  [Audio] Menjalankan aplay ke USB Sound Card (hw:{card_idx})")
        else:
            cmd = ["aplay", str(wav_file.resolve())]
            log.info(f"  [Audio] Menjalankan aplay default")

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            try:
                subprocess.run(["paplay", str(wav_file.resolve())], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                log.error(f"  [Audio] Gagal memutar audio di Linux: {e}")

def telegram_worker():
    """Worker background khusus untuk pengiriman pesan Telegram secara real-time (non-blocking untuk audio)."""
    while not exit_event.is_set():
        try:
            task = telegram_queue.get(timeout=0.5)
        except queue.Empty:
            continue
            
        if task is None:
            break
            
        text_report, annotated_frame, t_detect = task
        
        # Simpan gambar hasil deteksi pelanggaran ke folder terpisah secara lokal (diabaikan oleh Git)
        try:
            Path("detections").mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = Path("detections") / f"violation_{timestamp}.jpg"
            cv2.imwrite(str(img_path), annotated_frame)
            log.info(f"  [Storage] Gambar pelanggaran berhasil disimpan ke: {img_path}")
        except Exception as e:
            log.error(f"  [Storage] Gagal menyimpan gambar pelanggaran: {e}")
            
        # Cek jika aplikasi akan keluar sebelum mengirim Telegram
        if exit_event.is_set():
            telegram_queue.task_done()
            break
            
        # Kirim Laporan ke Telegram
        if CONFIG["telegram_token"] and CONFIG["telegram_chat_id"]:
            log.info("  [Telegram] Mengirim notifikasi alarm pelanggaran APD...")
            send_telegram_photo(text_report, annotated_frame, t_detect)
        else:
            log.warning("  [Telegram] Kredensial tidak dikonfigurasi. Laporan Telegram dilewati.")
            print(f"\n{text_report}\n")

        telegram_queue.task_done()

def audio_worker():
    """Worker background untuk memutar audio peringatan. Selalu memperbarui ke pelanggaran terbaru saat audio selesai."""
    global latest_audio_task
    while not exit_event.is_set():
        task = None
        with audio_lock:
            if latest_audio_task is not None:
                task = latest_audio_task
                latest_audio_task = None
        
        if task is None:
            time.sleep(0.1)
            continue
            
        violations_to_play = task
        audio_file = get_combination_audio(violations_to_play)
        if audio_file:
            log.info(f"  [Audio Warning] Memutar peringatan kombinasi: {Path(audio_file).name}...")
            play_audio_blocking(audio_file)

# Jalankan thread background baru
telegram_thread = threading.Thread(target=telegram_worker, daemon=True)
telegram_thread.start()

audio_thread = threading.Thread(target=audio_worker, daemon=True)
audio_thread.start()

# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────
def send_telegram_photo(caption: str, frame: np.ndarray, t_detect: float) -> None:
    """Mengirim pesan alarm teks lengkap dengan gambar hasil deteksi (color-coded) ke Telegram."""
    token = CONFIG["telegram_token"]
    chat_id = CONFIG["telegram_chat_id"]
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    
    # Hitung tundaan waktu pengiriman dari deteksi kamera ke bot Telegram
    t_delay_send = time.time() - t_detect
    caption_with_delay = caption + f"\n⏱️ <b>Delay Pengiriman:</b> {t_delay_send:.2f} detik"
    
    # Encode frame gambar langsung di memori untuk efisiensi
    success, img_encoded = cv2.imencode(".jpg", frame)
    if not success:
        log.error("  [Telegram] Gagal mengompresi frame gambar.")
        return
        
    files = {"photo": ("alarm.jpg", img_encoded.tobytes(), "image/jpeg")}
    data = {"chat_id": chat_id, "caption": caption_with_delay, "parse_mode": "HTML"}
    
    try:
        t_post_start = time.time()
        response = requests.post(url, data=data, files=files, timeout=12)
        t_post_duration = time.time() - t_post_start
        if response.status_code == 200:
            log.info(f"  [Telegram] Berhasil mengirim laporan alarm! (API Delay: {t_post_duration:.2f}s)")
        else:
            log.error(f"  [Telegram] Gagal mengirim: {response.text}")
    except Exception as e:
        log.error(f"  [Telegram] Kesalahan koneksi ke bot Telegram: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# DETEKSI INDEKS KAMERA
# ─────────────────────────────────────────────────────────────────────────────
def list_available_cameras():
    """Mendeteksi indeks kamera USB yang aktif di sistem."""
    log.info("  [Camera] Mendeteksi kamera yang tersedia di sistem...")
    available_indices = []
    # Coba indeks 0 hingga 5
    for index in range(6):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_indices.append(index)
            cap.release()
    if available_indices:
        log.info(f"  [Camera] Kamera aktif ditemukan pada indeks: {available_indices}")
    else:
        log.warning("  [Camera] Tidak ditemukan kamera aktif pada indeks 0-5.")
    return available_indices

# ─────────────────────────────────────────────────────────────────────────────
# DETEKSI UTAMA (INFERENCE ENGINE)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="PROTMIND APD System — Inference Engine")
    parser.add_argument("--source", type=str, default=None, help="Sumber kamera/video (indeks angka atau path video file).")
    parser.add_argument("--list-cameras", action="store_true", help="Menampilkan daftar indeks kamera USB yang tersedia di laptop.")
    parser.add_argument("--conf-person", type=float, default=None, help="Confidence threshold untuk model Person (default: 0.50).")
    parser.add_argument("--conf-ppe", type=float, default=None, help="Confidence threshold untuk model PPE (default: 0.50).")
    parser.add_argument("--headless", action="store_true", help="Jalankan inferensi tanpa menampilkan window GUI (cv2.imshow).")
    parser.add_argument("--test-audio", action="store_true", help="Uji coba pemutaran audio pada kartu suara USB.")
    args = parser.parse_args()

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

    if args.list_cameras:
        list_available_cameras()
        sys.exit(0)

    # Perbarui konfigurasi dinamis berdasarkan argumen CLI
    if args.source is not None:
        if args.source.isdigit():
            source = int(args.source)
        else:
            source = args.source
        CONFIG["camera_source"] = source
        log.info(f"  [CLI Override] Sumber kamera diatur ke: {source}")
    else:
        source = CONFIG["camera_source"]

    if args.conf_person is not None:
        CONFIG["conf_person"] = args.conf_person
        log.info(f"  [CLI Override] Conf Person diatur ke: {args.conf_person}")

    if args.conf_ppe is not None:
        CONFIG["conf_ppe"] = args.conf_ppe
        log.info(f"  [CLI Override] Conf PPE diatur ke: {args.conf_ppe}")

    if args.headless:
        CONFIG["headless"] = True
        log.info("  [CLI Override] Mode Headless diaktifkan (GUI Window dinonaktifkan).")

    # Load Models
    log.info("=" * 60)
    log.info("  PROTMIND APD SYSTEM — INFERENCE MODE")
    log.info("=" * 60)
    log.info(f"  Stage 1 (Person) : {CONFIG['model_person']}")
    log.info(f"  Stage 2 (PPE)    : {CONFIG['model_ppe']}")
    
    if not Path(CONFIG["model_ppe"]).exists():
        log.error(f"  File model PPE '{CONFIG['model_ppe']}' tidak ditemukan.")
        log.error("  Pastikan Anda telah menjalankan pelatihan menggunakan 'train.py' terlebih dahulu.")
        sys.exit(1)
        
    model_p = YOLO(CONFIG["model_person"])
    model_ppe = YOLO(CONFIG["model_ppe"])
    
    # Class maps untuk model PPE
    # 0: helmet (compliant), 1: no-helmet (violation)
    # 2: no-safety shoes (violation), 3: no-vest (violation)
    # 4: safety shoes (compliant), 5: vest (compliant)
    ppe_classes = {
        0: ("helmet", True),
        1: ("no-helmet", False),
        2: ("no-safety shoes", False),
        3: ("no-vest", False),
        4: ("safety shoes", True),
        5: ("vest", True)
    }

    # Inisialisasi status Cooldown (terakhir kali diputar)
    last_played = {
        "no-helmet": 0.0,
        "no-vest": 0.0,
        "no-safety shoes": 0.0
    }
    
    # Buka source kamera/video
    source = CONFIG["camera_source"]
    
    opened = False
    if platform.system() == "Linux" and isinstance(source, int):
        try:
            # Gunakan v4l2src GStreamer pipeline untuk hardware-accelerated BGR decoding
            gstreamer_pipeline = (
                f"v4l2src device=/dev/video{source} ! "
                "video/x-raw, width=640, height=480, format=YUY2, framerate=30/1 ! "
                "videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false"
            )
            log.info(f"  [Camera] Mencoba GStreamer pipeline: {gstreamer_pipeline}")
            cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                opened = True
        except Exception:
            pass
            
    if not opened:
        log.info(f"  [Camera] Menggunakan OpenCV standard backend untuk source: {source}")
        cap = cv2.VideoCapture(source)
        
    if not cap.isOpened():
        log.error(f"  Gagal membuka kamera atau video source: {source}")
        sys.exit(1)
        
    log.info(f"  Kamera/Video berhasil terbuka. Sumber: {source}")
    log.info("  Tekan 'Q' pada jendela video untuk keluar.")
    log.info("-" * 60)

    # Frame skipping untuk performa Jetson Nano
    frame_count = 0
    prev_frame_time = time.time()
    current_fps_frame = 0.0
    current_fps_inf = 0.0
    
    # Untuk perhitungan FPS Frame metode Window (akurat akademik)
    fps_window_counter = 0
    t_window_start = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                log.info("  Selesai memproses video (EOF) atau kamera terputus.")
                break
                
            frame_count += 1
            
            # Catat waktu mulai inferensi
            t_inf_start = time.time()
            
            # Salinan frame untuk anotasi visual
            annotated_frame = frame.copy()
            H, W, _ = frame.shape
            
            t_model_start = time.time()
            # Jalankan Deteksi Stage 1: Temukan Person (class 0 pada model COCO bawaan)
            results_p = model_p(frame, classes=0, conf=CONFIG["conf_person"], verbose=False)
            person_boxes = results_p[0].boxes.xyxy.cpu().numpy()
            
            total_people = len(person_boxes)
            
            # Koleksi pelanggaran yang terdeteksi di frame ini
            frame_violations = []
            
            # Variabel ringkasan kepatuhan dan pelanggaran per individu untuk laporan
            melanggar_count = 0
            lengkap_count = 0
            
            no_helmet_count = 0
            no_vest_count = 0
            no_shoes_count = 0
            
            for p_idx, p_box in enumerate(person_boxes):
                px1, py1, px2, py2 = map(int, p_box[:4])
                
                # Batasi koordinat agar berada dalam frame gambar
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(W, px2), min(H, py2)
                
                p_w = px2 - px1
                p_h = py2 - py1
                
                if p_w < 20 or p_h < 20:
                    continue
                    
                # Potong (Crop) area person dari frame
                person_crop = frame[py1:py2, px1:px2]
                
                # Jalankan Deteksi Stage 2: Cek kelengkapan APD pada potongan person tersebut
                results_ppe = model_ppe(person_crop, conf=CONFIG["conf_ppe"], verbose=False)
                ppe_boxes = results_ppe[0].boxes
                
                person_has_violation = False
                person_violations = []
                y_shoes_bottom = None
                
                # Loop setiap objek APD terdeteksi di dalam potongan person
                for ppe_box in ppe_boxes:
                    cls_id = int(ppe_box.cls.cpu().numpy()[0])
                    cls_conf = float(ppe_box.conf.cpu().numpy()[0])
                    cx_c, cy_c, w_c, h_c = ppe_box.xywhn.cpu().numpy()[0] # normalized coords on crop
                    
                    # Pemetaan kelas dan status kepatuhan
                    cls_name, is_compliant = ppe_classes.get(cls_id, ("unknown", True))
                    
                    # Peta koordinat bounding box kembali ke koordinat gambar penuh
                    px_c_abs = cx_c * p_w + px1
                    py_c_abs = cy_c * p_h + py1
                    pw_abs = w_c * p_w
                    ph_abs = h_c * p_h
                    
                    cx1 = int(px_c_abs - pw_abs/2)
                    cy1 = int(py_c_abs - ph_abs/2)
                    cx2 = int(px_c_abs + pw_abs/2)
                    cy2 = int(py_c_abs + ph_abs/2)
                    
                    # Simpan koordinat Y terbawah untuk safety shoes
                    if cls_name in ["safety shoes", "no-safety shoes", "safety-shoes", "no_safety-shoes"]:
                        if y_shoes_bottom is None or cy2 > y_shoes_bottom:
                            y_shoes_bottom = cy2
                    
                    # Warna bounding box objek APD
                    color = (0, 255, 0) if is_compliant else (0, 0, 255)
                    
                    # Gambar bounding box atribut APD
                    cv2.rectangle(annotated_frame, (cx1, cy1), (cx2, cy2), color, 2)
                    cv2.putText(
                        annotated_frame, f"{cls_name} {cls_conf:.2f}", (cx1, cy1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
                    )
                    
                    # Catat jika terdeteksi pelanggaran
                    if not is_compliant:
                        person_has_violation = True
                        person_violations.append(cls_name)
                        if cls_name not in frame_violations:
                            frame_violations.append(cls_name)
                            
                        # Akumulasi jumlah pelanggar untuk laporan
                        if cls_name == "no-helmet":
                            no_helmet_count += 1
                        elif cls_name == "no-vest":
                            no_vest_count += 1
                        elif cls_name == "no-safety shoes":
                            no_shoes_count += 1
                
                # Jika sepatu tidak terdeteksi (occluded), gunakan bottom person bbox (py2) sebagai fallback
                target_y_bottom = y_shoes_bottom if y_shoes_bottom is not None else py2
                worker_distance = estimate_distance(px1, py1, px2, py2, W, H)
                
                # Klasifikasi kepatuhan pekerja
                if person_has_violation:
                    melanggar_count += 1
                else:
                    lengkap_count += 1
                    
                # Anotasi visual untuk Bounding Box Manusia (Stage 1)
                p_color = (0, 0, 255) if person_has_violation else (0, 255, 0)
                cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), p_color, 3)
                
                status_text = f"Pekerja #{p_idx+1} ({worker_distance}m): "
                if person_has_violation:
                    status_text += f"MELANGGAR ({', '.join(person_violations)})"
                else:
                    status_text += "LENGKAP"
                    
                cv2.putText(
                    annotated_frame, status_text, (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, p_color, 2
                )

            # Simpan gambar hasil deteksi jika ada pekerja terdeteksi
            if total_people > 0:
                save_dir = Path("detections")
                save_dir.mkdir(parents=True, exist_ok=True)
                
                has_any_violation = len(frame_violations) > 0
                status_str = "melanggar" if has_any_violation else "lengkap"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                
                save_path = save_dir / f"deteksi_{timestamp}_{status_str}.jpg"
                cv2.imwrite(str(save_path), annotated_frame)

            # Catat akhir waktu inferensi murni
            t_pure_inf = time.time() - t_model_start
            
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
                
            # Update FPS Inferensi jika waktu murni terhitung
            alpha = 0.1
            if t_pure_inf > 0.0:
                fps_inf = 1.0 / t_pure_inf
                if current_fps_inf == 0.0:
                    current_fps_inf = fps_inf
                else:
                    current_fps_inf = alpha * fps_inf + (1.0 - alpha) * current_fps_inf
                
            # Ambil metrik resource sistem secara real-time
            cpu_pct, gpu_pct, ram_pct = get_system_metrics()
            
            # Log nilai FPS secara periodik (setiap 30 frame) agar log tidak penuh
            if frame_count % 30 == 0:
                log.info(
                    f"  [Performance] FPS Frame: {current_fps_frame:.2f} | FPS Inferensi: {current_fps_inf:.2f} | "
                    f"CPU: {cpu_pct:.1f}% | GPU: {gpu_pct:.1f}% | RAM: {ram_pct:.1f}%"
                )

            # ─────────────────────────────────────────────────────────────────────
            # PROSES ALARM COOLDOWN & ROUND-ROBIN QUEUE
            # ─────────────────────────────────────────────────────────────────────
            current_time = time.time()
            violations_to_alert = []
            
            for violation in frame_violations:
                # Uji apakah waktu cooldown 10 detik sudah terlewati
                if current_time - last_played[violation] >= CONFIG["cooldown_time"]:
                    violations_to_alert.append(violation)
                    # Tandai cooldown berjalan
                    last_played[violation] = current_time
                    
            # Jika minimal ada 1 pelanggaran baru yang lolos filter cooldown, picu alarm!
            if violations_to_alert:
                log.info(f"  [Alarm!] Terdeteksi Pelanggaran Baru: {violations_to_alert}")
                
                # Buat teks pesan template Telegram Bot berformat HTML
                waktu_str = datetime.now().strftime("%d %B %Y, %H:%M:%S") + " WIB"
                
                warning_text = (
                    f"🚨 <b>ALARM PELANGGARAN APD</b> 🚨\n\n"
                    f"📍 <b>Lokasi Kamera:</b> {CONFIG['lokasi_kamera']}\n"
                    f"🕒 <b>Waktu:</b> {waktu_str}\n\n"
                    f"📊 <b>Ringkasan Kepatuhan:</b>\n"
                    f"👥 Total Pekerja: {total_people} Orang\n"
                    f"✅ Pekerja lengkap: {lengkap_count} Orang\n"
                    f"❌ Pekerja melanggar: {melanggar_count} Orang\n\n"
                    f"⚠️ <b>Rincian Pelanggaran:</b>\n"
                )
                
                if no_helmet_count > 0:
                    warning_text += f"• {no_helmet_count} Orang : Tidak menggunakan Helm (no-helmet)\n"
                if no_vest_count > 0:
                    warning_text += f"• {no_vest_count} Orang : Tidak menggunakan Rompi (no-vest)\n"
                if no_shoes_count > 0:
                    warning_text += f"• {no_shoes_count} Orang : Tidak menggunakan Sepatu Safety (no-safety shoes)\n"
                    
                warning_text += f"\nMohon petugas HSE segera mengecek lokasi terkait."
                
                # Masukkan ke queue agar diproses thread latar belakang (Telegram & Audio bergantian)
                # Mengirimkan frame teranotasi sebagai visual bukti pelanggaran
                telegram_queue.put((warning_text, annotated_frame.copy(), time.time()))
                with audio_lock:
                    latest_audio_task = violations_to_alert



            # Tampilkan visual deteksi
            cv2.putText(
                annotated_frame, f"Lokasi: {CONFIG['lokasi_kamera']}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.putText(
                annotated_frame, f"Total Pekerja: {total_people}", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.putText(
                annotated_frame, f"FPS Frame: {current_fps_frame:.1f} | FPS Inf: {current_fps_inf:.1f}", (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            cv2.putText(
                annotated_frame, f"CPU: {cpu_pct:.1f}% | GPU: {gpu_pct:.1f}% | RAM: {ram_pct:.1f}%", (15, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            if not CONFIG.get("headless", False):
                try:
                    cv2.imshow("PROTMIND APD Detection - Edge Device", annotated_frame)
                    
                    # Keluar jika menekan tombol 'Q'
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        log.info("  Keluar atas permintaan pengguna.")
                        break
                except cv2.error as e:
                    log.warning(
                        "  [GUI Warning] Jendela tampilan tidak didukung oleh instalasi OpenCV Anda. "
                        "Mengaktifkan mode headless otomatis."
                    )
                    CONFIG["headless"] = True
                    time.sleep(0.001)
            else:
                # Mode headless: berikan jeda pendek non-blocking agar CPU tidak pinned 100%
                time.sleep(0.001)
    except KeyboardInterrupt:
        log.info("  Program dihentikan oleh pengguna (Ctrl+C).")
    finally:
        # Cleanup
        exit_event.set()
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        
        # Hentikan worker thread
        try:
            warning_queue.put(None)
        except Exception:
            pass
        worker_thread.join(timeout=1.0)
        log.info("  Sistem dimatikan dengan sukses.")

if __name__ == "__main__":
    main()
