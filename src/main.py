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
    "model_person"      : "yolov8n.pt",         # Model Stage 1 untuk deteksi person bawaan YOLOv8
    "model_ppe"         : "models/best_ppe.pt", # Model Stage 2 hasil training pada cropped person

    # Nilai Ambang Batas Kepercayaan (Confidence Threshold)
    "conf_person"       : 0.40,
    "conf_ppe"          : 0.40,

    # Manajemen Alarm & Audio
    "cooldown_time"     : int(os.getenv("COOLDOWN_TIME", "10")), # Jeda waktu minimum (detik) alarm sejenis tidak boleh berulang-ulang
    "audio_delay"       : int(os.getenv("AUDIO_DELAY", "3")),   # Jeda waktu (detik) pemutaran audio bergiliran (Round-Robin)

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
        # Untuk Jetson Nano Linux, 'aplay' adalah pilihan tercepat dengan latensi paling rendah
        try:
            subprocess.Popen(["aplay", str(wav_file.resolve())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            # Fallback jika aplay tidak ada, coba paplay (PulseAudio)
            try:
                subprocess.Popen(["paplay", str(wav_file.resolve())], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                log.error(f"  [Audio] Gagal memutar audio di Linux: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND QUEUE WORKER
# ─────────────────────────────────────────────────────────────────────────────
# Menggunakan antrean asinkron agar proses pemutaran suara jeda 3 detik 
# dan pengiriman Telegram tidak menyebabkan lag atau stuttering pada stream kamera.
warning_queue = queue.Queue()

def warning_worker():
    """Mengambil tugas pelanggaran dari queue untuk memutar suara dan kirim Telegram."""
    while True:
        task = warning_queue.get()
        if task is None:
            break
        
        violations_to_play, text_report, annotated_frame = task
        
        # 1. Kirim Alarm Laporan ke Telegram
        if CONFIG["telegram_token"] and CONFIG["telegram_chat_id"]:
            log.info("  [Telegram] Mengirim notifikasi alarm pelanggaran APD...")
            send_telegram_photo(text_report, annotated_frame)
        else:
            log.warning("  [Telegram] Kredensial tidak dikonfigurasi. Laporan Telegram dilewati.")
            print(f"\n{text_report}\n")

        # 2. Putar Audio Peringatan secara bergiliran (Round-Robin)
        for violation in violations_to_play:
            if violation == "no-helmet":
                log.info("  [Audio Warning] Memutar peringatan HELM...")
                play_audio(CONFIG["audio_helm"])
            elif violation == "no-vest":
                log.info("  [Audio Warning] Memutar peringatan ROMPI...")
                play_audio(CONFIG["audio_rompi"])
            elif violation == "no-safety shoes":
                log.info("  [Audio Warning] Memutar peringatan SEPATU...")
                play_audio(CONFIG["audio_sepatu"])
            
            # Berikan jeda 3 detik sebelum pemutaran suara berikutnya
            time.sleep(CONFIG["audio_delay"])
            
        warning_queue.task_done()

# Start background thread
worker_thread = threading.Thread(target=warning_worker, daemon=True)
worker_thread.start()

# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────
def send_telegram_photo(caption: str, frame: np.ndarray) -> None:
    """Mengirim pesan alarm teks lengkap dengan gambar hasil deteksi (color-coded) ke Telegram."""
    token = CONFIG["telegram_token"]
    chat_id = CONFIG["telegram_chat_id"]
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    
    # Encode frame gambar langsung di memori untuk efisiensi
    success, img_encoded = cv2.imencode(".jpg", frame)
    if not success:
        log.error("  [Telegram] Gagal mengompresi frame gambar.")
        return
        
    files = {"photo": ("alarm.jpg", img_encoded.tobytes(), "image/jpeg")}
    data = {"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}
    
    try:
        response = requests.post(url, data=data, files=files, timeout=12)
        if response.status_code == 200:
            log.info("  [Telegram] Berhasil mengirim laporan alarm!")
        else:
            log.error(f"  [Telegram] Gagal mengirim: {response.text}")
    except Exception as e:
        log.error(f"  [Telegram] Kesalahan koneksi ke bot Telegram: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# DETEKSI UTAMA (INFERENCE ENGINE)
# ─────────────────────────────────────────────────────────────────────────────
def main():
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
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f"  Gagal membuka kamera atau video source: {source}")
        sys.exit(1)
        
    log.info(f"  Kamera/Video berhasil terbuka. Sumber: {source}")
    log.info("  Tekan 'Q' pada jendela video untuk keluar.")
    log.info("-" * 60)

    # Frame skipping untuk performa Jetson Nano
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            log.info("  Selesai memproses video (EOF) atau kamera terputus.")
            break
            
        frame_count += 1
        
        # Salinan frame untuk anotasi visual
        annotated_frame = frame.copy()
        H, W, _ = frame.shape
        
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
                
                # Warna bounding box objek APD
                # Hijau untuk lengkap (helmet, vest, safety shoes)
                # Merah untuk pelanggaran (no-helmet, no-vest, no-safety shoes)
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
            
            # Klasifikasi kepatuhan pekerja
            if person_has_violation:
                melanggar_count += 1
            else:
                lengkap_count += 1
                
            # Anotasi visual untuk Bounding Box Manusia (Stage 1)
            # Merah: Pekerja melanggar aturan APD
            # Hijau: Pekerja patuh APD lengkap (tidak ada no-helmet / no-vest / no-shoes)
            p_color = (0, 0, 255) if person_has_violation else (0, 255, 0)
            cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), p_color, 3)
            
            status_text = f"Pekerja #{p_idx+1}: "
            if person_has_violation:
                status_text += f"MELANGGAR ({', '.join(person_violations)})"
            else:
                status_text += "LENGKAP"
                
            cv2.putText(
                annotated_frame, status_text, (px1, py1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, p_color, 2
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
            warning_queue.put((violations_to_alert, warning_text, annotated_frame.copy()))

        # Tampilkan visual deteksi
        cv2.putText(
            annotated_frame, f"Lokasi: {CONFIG['lokasi_kamera']}", (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            annotated_frame, f"Total Pekerja: {total_people}", (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        cv2.imshow("PROTMIND APD Detection - Edge Device", annotated_frame)
        
        # Keluar jika menekan tombol 'Q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            log.info("  Keluar atas permintaan pengguna.")
            break
            
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Hentikan worker thread
    warning_queue.put(None)
    worker_thread.join()
    log.info("  Sistem dimatikan dengan sukses.")

if __name__ == "__main__":
    main()
