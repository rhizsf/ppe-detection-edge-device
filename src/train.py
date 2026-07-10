"""
=============================================================================
PROTMIND — train.py
Sistem Deteksi Pelanggaran APD | YOLOv8n Training Script
=============================================================================
Fungsi   : Melatih model YOLOv8n pada dataset APD dan menghasilkan
           visualisasi grafik analisis pelatihan secara otomatis.
Hardware : GPU (direkomendasikan), CPU juga bisa tapi sangat lambat.
Output   : models/best.pt  ←  digunakan di Jetson Nano untuk inferensi
=============================================================================
"""

import os
import sys
import csv
import time
import shutil
import logging
import platform
from pathlib import Path
from datetime import datetime

import torch
import matplotlib
matplotlib.use("Agg")           # render tanpa display (cocok untuk server/headless)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# DETEKSI PLATFORM — Windows memerlukan setting khusus untuk DataLoader
# ─────────────────────────────────────────────────────────────────────────────
_IS_WINDOWS = platform.system() == "Windows"

# Windows: pin_memory + multi-worker menyebabkan "CUDA error: resource already mapped"
# karena Windows menggunakan spawn() bukan fork() untuk multiprocessing.
# Solusi: batasi workers dan nonaktifkan pin_memory di Windows.
_SAFE_WORKERS   = 0 if _IS_WINDOWS else 8   # 0 = single-process, aman di Windows
_PIN_MEMORY     = False if _IS_WINDOWS else True

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI UTAMA — ubah di sini sesuai kebutuhan
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Path & Pipeline Mode
    "two_stage"      : True,
    "train_person"   : True,
    "force_recreate" : False,
    "original_yaml"  : "assets/dataset/data.yaml",
    "cropped_yaml"   : "assets/dataset_cropped/data_cropped.yaml",
    "person_yaml"    : "assets/dataset_person/data_person.yaml",
    "data_yaml"      : "assets/dataset_cropped/data_cropped.yaml",
    "model_weights"  : "yolov8n.pt",
    "output_model"   : "models/best_ppe.pt",
    "output_person_model": "models/best_person.pt",
    "runs_dir"       : "runs",
    "graphs_dir"     : "models/training_graphs",

    # Hiperparameter Pelatihan
    "epochs"         : 150,
    "epochs_person"  : 50,
    "imgsz"          : 640,
    "batch"          : 16,
    "patience"       : 20,
    "lr0"            : 0.01,
    "lrf"            : 0.01,
    "momentum"       : 0.937,
    "weight_decay"   : 0.0005,
    "warmup_epochs"  : 3,
    "project_name"   : "protmind_70-_20-_10-auto labeling",  

    # Augmentasi — aktif untuk dataset konstruksi (kondisi cahaya bervariasi)
    "augment"        : True,
    "hsv_h"          : 0.015,
    "hsv_s"          : 0.7,
    "hsv_v"          : 0.4,
    "flipud"         : 0.0,
    "fliplr"         : 0.5,
    "mosaic"         : 0.0,
    "mixup"          : 0.0,
    "freeze"         : 10,
    "cls"            : 1.5,
    "cos_lr"         : True,
    "label_smoothing": 0.1,
    "degrees"        : 10.0,
    "translate"      : 0.1,
    "scale"          : 0.5,
    "shear"          : 2.0,
    
    # Device & DataLoader
    "device"         : "cuda" if torch.cuda.is_available() else "cpu",
    "workers"        : _SAFE_WORKERS,           # ← FIXED: 0 di Windows, 8 di Linux
    "amp"            : True,                    # Automatic Mixed Precision (hemat VRAM)
    "seed"           : 42,
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
        logging.FileHandler("training.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("PROTMIND-TRAIN")


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAS
# ─────────────────────────────────────────────────────────────────────────────
def verify_environment() -> None:
    """Validasi GPU, dependensi, dan path penting sebelum training dimulai."""
    log.info("=" * 60)
    log.info("  PROTMIND — YOLOv8n APD Training")
    log.info("=" * 60)

    # GPU check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram     = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"  GPU Terdeteksi : {gpu_name}")
        log.info(f"  VRAM Total     : {vram:.1f} GB")
    else:
        log.warning("  GPU tidak terdeteksi — training berjalan di CPU (LAMBAT)")
        log.warning("  Pastikan CUDA driver dan torch-CUDA sudah terinstall.")

    # File check
    target_yaml = CONFIG["original_yaml"] if CONFIG["two_stage"] else CONFIG["data_yaml"]
    if not Path(target_yaml).exists():
        log.error(f"  data.yaml tidak ditemukan: {target_yaml}")
        sys.exit(1)

    log.info(f"  Platform       : {platform.system()} {platform.release()}")
    log.info(f"  Data YAML Asli : {target_yaml}  ✓")
    if CONFIG["two_stage"]:
        log.info(f"  Pipeline Mode  : Two-Stage (Training pada cropped person)")
    log.info(f"  Device         : {CONFIG['device'].upper()}")
    log.info(f"  Epochs         : {CONFIG['epochs']}")
    log.info(f"  Batch Size     : {CONFIG['batch']}")
    log.info(f"  Image Size     : {CONFIG['imgsz']}px")
    log.info(f"  Workers        : {CONFIG['workers']} {'← Windows safe mode (0=single process)' if _IS_WINDOWS else ''}")
    log.info(f"  Pin Memory     : {_PIN_MEMORY}")
    log.info(f"  AMP            : {'Aktif' if CONFIG['amp'] else 'Nonaktif'}")
    log.info("=" * 60)


def cluster_unassigned_ppe(orig_labels, person_boxes, W, H):
    """
    Mengelompokkan anotasi APD yang tidak ter-cover oleh deteksi model person,
    lalu merekonstruksi kotak bounding box person secara heuristik untuk data training.
    """
    # 1. Konversi person_boxes ke koordinat normalisasi agar mudah dicocokkan
    norm_person_boxes = []
    for box in person_boxes:
        x1, y1, x2, y2 = box[:4]
        norm_person_boxes.append((x1/W, y1/H, x2/W, y2/H))
        
    # 2. Cari label APD asli yang tidak tercakup kotak person mana pun
    unassigned_ppe = []
    for label in orig_labels:
        cls_id, cx, cy, w, h = label
        assigned = False
        for px1, py1, px2, py2 in norm_person_boxes:
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                assigned = True
                break
        if not assigned:
            unassigned_ppe.append(label)
            
    if not unassigned_ppe:
        return []
        
    # 3. Kelompokkan APD tak ter-cover berdasarkan kedekatan sumbu X (threshold 0.15)
    unassigned_ppe.sort(key=lambda x: x[1])
    clusters = []
    current_cluster = [unassigned_ppe[0]]
    for label in unassigned_ppe[1:]:
        if label[1] - current_cluster[-1][1] <= 0.15:
            current_cluster.append(label)
        else:
            clusters.append(current_cluster)
            current_cluster = [label]
    clusters.append(current_cluster)
    
    # 4. Konstruksi kotak orang untuk setiap kelompok APD
    fallback_person_boxes = []
    for cluster in clusters:
        min_x = min(cx - w/2 for _, cx, _, w, _ in cluster)
        max_x = max(cx + w/2 for _, cx, _, w, _ in cluster)
        min_y = min(cy - h/2 for _, _, cy, _, h in cluster)
        max_y = max(cy + h/2 for _, _, cy, _, h in cluster)
        
        has_helmet = any(cls_id in [0, 1] for cls_id, _, _, _, _ in cluster)
        has_shoes = any(cls_id in [2, 4] for cls_id, _, _, _, _ in cluster)
        has_vest = any(cls_id in [3, 5] for cls_id, _, _, _, _ in cluster)
        
        # Perluas ke atas (untuk kepala/helm jika tidak terdeteksi)
        if not has_helmet:
            h_est = max_y - min_y
            if has_vest:
                min_y = max(0.0, min_y - 0.3 * h_est)
            else:
                min_y = max(0.0, min_y - 0.15 * h_est)
        else:
            min_y = max(0.0, min_y - 0.05)
            
        # Perluas ke bawah (untuk kaki/sepatu jika tidak terdeteksi)
        if not has_shoes:
            h_est = max_y - min_y
            if has_vest:
                max_y = min(1.0, max_y + 0.6 * h_est)
            else:
                max_y = min(1.0, max_y + 0.3 * h_est)
        else:
            max_y = min(1.0, max_y + 0.05)
            
        # Berikan padding horizontal
        w_est = max_x - min_x
        min_x = max(0.0, min_x - 0.1 * w_est - 0.05)
        max_x = min(1.0, max_x + 0.1 * w_est + 0.05)
        
        # Konversi kembali ke pixel absolute
        fallback_person_boxes.append([
            int(min_x * W),
            int(min_y * H),
            int(max_x * W),
            int(max_y * H)
        ])
        
    return fallback_person_boxes


def clean_dataset_caches(dataset_dir: Path) -> None:
    """
    Menghapus berkas labels.cache secara eksplisit di seluruh folder dataset
    untuk mencegah masalah caching YOLOv8 jika data diubah.
    """
    if not dataset_dir.exists():
        return
    for cache_file in dataset_dir.rglob("*.cache"):
        try:
            log.info(f"  [Cache Clean] Menghapus file cache: {cache_file.name} dari {cache_file.parent}")
            cache_file.unlink()
        except Exception as e:
            log.warning(f"  [Cache Clean] Gagal menghapus {cache_file.name}: {e}. Berkas mungkin sedang dikunci.")


def prepare_person_dataset() -> str:
    """
    Menghasilkan dataset baru yang berisi label deteksi person saja (class 0).
    Label person didapat dari model pretrained + fallback clustering pada anotasi APD.
    """
    import yaml
    import cv2
    
    orig_yaml_path = Path(CONFIG["original_yaml"])
    person_dir = Path("assets/dataset_person")
    person_yaml_path = Path(CONFIG["person_yaml"])
    
    # Load original data.yaml
    with open(orig_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
        
    if person_dir.exists() and CONFIG.get("force_recreate", False):
        log.info("  [Preprocess Person] 'force_recreate' aktif. Menghapus folder dataset person lama...")
        clean_dataset_caches(person_dir)
        shutil.rmtree(person_dir, ignore_errors=True)
        if person_yaml_path.exists():
            try:
                person_yaml_path.unlink()
            except Exception:
                pass
                
    if person_yaml_path.exists():
        log.info(f"  [Preprocess Person] Dataset person sudah ada di: {person_yaml_path.resolve()}")
        return str(person_yaml_path)
        
    log.info("=" * 60)
    log.info("  [Preprocess Person] MENYIAPKAN DATASET DETEKSI PERSON (STAGE 1)")
    log.info("=" * 60)
    
    orig_base_dir = orig_yaml_path.parent
    
    # Inisialisasi model person detector
    log.info("  Memuat model YOLOv8n untuk pseudo-labeling person...")
    person_model = YOLO(CONFIG["model_weights"])
    
    splits = {
        "train": data_cfg.get("train", "../train/images"),
        "val": data_cfg.get("val", "../valid/images"),
        "test": data_cfg.get("test", "../test/images")
    }
    
    # Buat direktori tujuan
    for split in splits.keys():
        (person_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (person_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
    for split_name, rel_path in splits.items():
        img_dir = (orig_base_dir / rel_path).resolve()
        if not img_dir.exists():
            clean_path = rel_path.replace("../", "")
            alt_dir = (orig_base_dir / clean_path).resolve()
            if alt_dir.exists():
                img_dir = alt_dir
            else:
                possible_paths = [
                    f"{split_name}/images",
                    f"valid/images" if split_name == "val" else "",
                    f"test/images" if split_name == "test" else ""
                ]
                found = False
                for p in possible_paths:
                    if p:
                        alt_dir = (orig_base_dir / p).resolve()
                        if alt_dir.exists():
                            img_dir = alt_dir
                            found = True
                            break
                if not found:
                    log.error(f"  Gagal menemukan direktori asli untuk split '{split_name}'.")
                    continue
                    
        log.info(f"  Memproses split '{split_name}' dari {img_dir}...")
        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
        
        person_count = 0
        
        for img_path in img_files:
            label_dir = img_path.parent.parent / "labels"
            label_path = label_dir / f"{img_path.stem}.txt"
            
            orig_labels = []
            if label_path.exists():
                with open(label_path, "r", encoding="utf-8") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:])
                            orig_labels.append((cls_id, cx, cy, w, h))
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W, _ = img.shape
            
            # Deteksi person
            results = person_model(img, classes=0, conf=0.25, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
            
            # Tambahkan fallback person boxes dari APD unassigned
            fallback_boxes = cluster_unassigned_ppe(orig_labels, boxes, W, H)
            if fallback_boxes:
                boxes.extend(fallback_boxes)
                
            # Simpan label person ke target (format YOLO: 0 cx cy w h)
            out_label_path = person_dir / split_name / "labels" / f"{img_path.stem}.txt"
            person_boxes_written = 0
            
            with open(out_label_path, "w", encoding="utf-8") as out_f:
                for box in boxes:
                    x1, y1, x2, y2 = box[:4]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    
                    cx = (x1 + x2) / 2.0 / W
                    cy = (y1 + y2) / 2.0 / H
                    w = (x2 - x1) / W
                    h = (y2 - y1) / H
                    
                    if w > 0.01 and h > 0.01:
                        out_f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                        person_boxes_written += 1
                        
            # Tulis gambar yang sudah di-resize ke target untuk meminimalkan RAM/VRAM
            out_img_path = person_dir / split_name / "images" / img_path.name
            
            # Resize jika ukuran gambar terlalu besar (misal > 1024px) untuk mencegah OOM pada Windows
            max_size = 1024
            if max(H, W) > max_size:
                scale = max_size / max(H, W)
                img_to_save = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                img_to_save = img
                
            cv2.imwrite(str(out_img_path), img_to_save)
            person_count += person_boxes_written
            
        log.info(f"  ✓ Split '{split_name}': berhasil menyimpan {len(img_files)} gambar dengan total {person_count} orang.")
        
    # Buat data_person.yaml
    person_yaml_content = {
        "path": str(person_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 1,
        "names": ["person"]
    }
    
    with open(person_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(person_yaml_content, f, default_flow_style=False)
        
    log.info(f"  ✓ File konfigurasi dataset person ditulis di: {person_yaml_path.resolve()}")
    log.info("=" * 60)
    
    return str(person_yaml_path)


def train_person_model() -> Path:
    """Melatih model YOLOv8n untuk mendeteksi class person saja."""
    log.info("=" * 60)
    log.info("  [Training] MELATIH CUSTOM PERSON DETECTOR (STAGE 1)")
    log.info("=" * 60)
    
    # Tentukan bobot model (cek jika ingin resume)
    weights_path = CONFIG["model_weights"]
    is_resuming = False
    
    if CONFIG.get("resume_person", False):
        last_person_path = Path(CONFIG["runs_dir"]) / (CONFIG["project_name"] + "_person") / "weights" / "last.pt"
        if last_person_path.exists():
            weights_path = str(last_person_path)
            is_resuming = True
            log.info(f"  [Resume] Melanjutkan pelatihan person dari checkpoint: {last_person_path}")
        else:
            # Fallback pencarian rekursif jika nama folder diubah otomatis oleh YOLO (misal protmind_balanced_v3_person-2)
            candidates = list(Path(CONFIG["runs_dir"]).rglob("**/weights/last.pt"))
            # Filter kandidat untuk yang memiliki _person di namanya
            candidates = [c for c in candidates if "_person" in str(c)]
            if candidates:
                candidates.sort(key=lambda x: x.stat().st_mtime)
                last_person_path = candidates[-1]
                weights_path = str(last_person_path)
                is_resuming = True
                log.info(f"  [Resume] Checkpoint person ditemukan lewat pencarian fallback: {last_person_path}")
            else:
                log.warning(f"  [Resume] Checkpoint person tidak ditemukan. Pelatihan dimulai dari awal.")
                
    log.info(f"  Memuat weights: {weights_path}")
    model = YOLO(weights_path)
    
    project_abs_path = str(Path(CONFIG["runs_dir"]).resolve())
    
    model.train(
        data          = CONFIG["person_yaml"],
        epochs        = CONFIG.get("epochs_person", 50),
        imgsz         = CONFIG["imgsz"],
        batch         = CONFIG["batch"],
        patience      = CONFIG["patience"],
        lr0           = CONFIG["lr0"],
        lrf           = CONFIG["lrf"],
        momentum      = CONFIG["momentum"],
        weight_decay  = CONFIG["weight_decay"],
        warmup_epochs = CONFIG["warmup_epochs"],
        augment       = CONFIG["augment"],
        device        = CONFIG["device"],
        workers       = CONFIG["workers"],
        amp           = CONFIG["amp"],
        seed          = CONFIG["seed"],
        project       = project_abs_path,
        name          = CONFIG["project_name"] + "_person",
        exist_ok      = True,
        verbose       = True,
        save          = True,
        val           = True,
        plots         = True,
        resume        = is_resuming
    )
    
    if hasattr(model, "trainer") and model.trainer is not None and hasattr(model.trainer, "save_dir"):
        run_dir = Path(model.trainer.save_dir)
    else:
        run_dir = find_latest_run(CONFIG["project_name"] + "_person")
        
    if not run_dir:
        raise RuntimeError("Gagal menemukan direktori hasil pelatihan person.")
        
    src = run_dir / "weights" / "best.pt"
    dst = Path(CONFIG["output_person_model"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    log.info(f"  ✓ Model person kustom terbaik disimpan ke: {dst.resolve()}")
    
    return dst


# def apply_programmatic_augmentations(img, labels):
#     """
#     Menerapkan augmentasi programmatif lokal secara acak:
#     1. Horizontal Flip (50% chance), dengan pembalikan koordinat BBox.
#     2. Brightness Adjustment (faktor acak dari 0.8 hingga 1.2 / -20% hingga +20%).
#     3. Gaussian Blur (maksimal 2px).
#     """
#     import random
#     import cv2
#     import numpy as np
#     
#     h_flipped = False
#     augmented_img = img.copy()
#     augmented_labels = [list(lbl) for lbl in labels]
#     
#     # 1. Horizontal Flip (50% probabilitas)
#     if random.random() < 0.5:
#         augmented_img = cv2.flip(augmented_img, 1)
#         h_flipped = True
#         for lbl in augmented_labels:
#             # lbl format: [cls_id, cx, cy, w, h]
#             # cx baru = 1.0 - cx lama
#             lbl[1] = 1.0 - lbl[1]
# 
#     # 2. Brightness adjustment (-20% hingga +20%)
#     hsv = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)
#     brightness_factor = random.uniform(0.8, 1.2)
#     v_new = np.clip(v.astype(np.int32) * brightness_factor, 0, 255).astype(np.uint8)
#     hsv_new = cv2.merge([h, s, v_new])
#     augmented_img = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
# 
#     # 3. Gaussian Blur (maksimal 2px)
#     blur_choice = random.choice([0, 1, 2])
#     if blur_choice == 1:
#         # Gaussian blur kernel 3x3, sigmaX=1.0
#         augmented_img = cv2.GaussianBlur(augmented_img, (3, 3), sigmaX=random.uniform(0.5, 1.5))
#     elif blur_choice == 2:
#         # Gaussian blur kernel 5x5, sigmaX=2.0 (maksimal 2px)
#         augmented_img = cv2.GaussianBlur(augmented_img, (5, 5), sigmaX=random.uniform(1.0, 2.0))
#         
#     return augmented_img, augmented_labels


def prepare_cropped_dataset() -> str:
    """
    Menghasilkan dataset baru yang berisi potongan (crops) person dari dataset asli.
    PPE labels direlokalisasi ke dalam koordinat crop person tersebut.
    """
    import cv2
    import yaml
    
    orig_yaml_path = Path(CONFIG["original_yaml"])
    cropped_dir = Path("assets/dataset_cropped")
    cropped_yaml_path = Path(CONFIG["cropped_yaml"])
    
    # Load original data.yaml
    with open(orig_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
        
    if cropped_dir.exists() and CONFIG.get("force_recreate", False):
        log.info("  [Preprocess] 'force_recreate' aktif. Menghapus folder dataset cropped lama...")
        clean_dataset_caches(cropped_dir)
        shutil.rmtree(cropped_dir, ignore_errors=True)
        if cropped_yaml_path.exists():
            try:
                cropped_yaml_path.unlink()
            except Exception:
                pass
        
    if cropped_yaml_path.exists():
        log.info(f"  [Preprocess] Dataset cropped sudah ada di: {cropped_yaml_path.resolve()}")
        # Perbarui file YAML ke format path absolut baru jika diperlukan
        try:
            with open(cropped_yaml_path, "r", encoding="utf-8") as f:
                existing = yaml.safe_load(f)
            if not existing or "path" not in existing:
                log.info("  [Preprocess] Memperbarui data_cropped.yaml ke format absolute path baru...")
                cropped_yaml_content = {
                    "path": str(cropped_dir.resolve()),
                    "train": "train/images",
                    "val": "val/images",
                    "test": "test/images",
                    "nc": len(data_cfg["names"]),
                    "names": data_cfg["names"]
                }
                with open(cropped_yaml_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cropped_yaml_content, f, default_flow_style=False)
                log.info("  [Preprocess] Pembaruan data_cropped.yaml selesai.")
        except Exception as e:
            log.warning(f"  Gagal memvalidasi/memperbarui data_cropped.yaml: {e}")
            
        log.info("  Melewati proses cropping. Jika ingin regenerasi, silakan aktifkan 'force_recreate' di CONFIG atau hapus folder 'assets/dataset_cropped'.")
        return str(cropped_yaml_path)
        
    log.info("=" * 60)
    log.info("  [Preprocess] MENYIAPKAN DATASET DENGAN CROPPING PERSON (STAGE 1)")
    log.info("=" * 60)
    
    orig_base_dir = orig_yaml_path.parent
    
    # Inisialisasi model person detector (Stage 1)
    custom_person_path = Path(CONFIG.get("output_person_model", "models/best_person.pt"))
    if custom_person_path.exists():
        log.info(f"  Memuat model person kustom hasil pelatihan: {custom_person_path.resolve()}")
        person_model = YOLO(str(custom_person_path))
    else:
        log.warning(f"  Model person kustom tidak ditemukan di {custom_person_path.resolve()}. Menggunakan model pretrained: {CONFIG['model_weights']}")
        person_model = YOLO(CONFIG["model_weights"])
    
    splits = {
        "train": data_cfg.get("train", "../train/images"),
        "val": data_cfg.get("val", "../valid/images"),
        "test": data_cfg.get("test", "../test/images")
    }
    
    # Buat direktori tujuan
    for split in splits.keys():
        (cropped_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (cropped_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
    for split_name, rel_path in splits.items():
        # Resolusi path gambar asli
        img_dir = (orig_base_dir / rel_path).resolve()
        if not img_dir.exists():
            # 1. Coba hilangkan awalan "../" yang sering ditambahkan Roboflow
            clean_path = rel_path.replace("../", "")
            alt_dir = (orig_base_dir / clean_path).resolve()
            if alt_dir.exists():
                img_dir = alt_dir
            else:
                # 2. Uji fallback folder standar (misal 'valid' vs 'val')
                possible_paths = [
                    f"{split_name}/images",
                    f"valid/images" if split_name == "val" else "",
                    f"test/images" if split_name == "test" else ""
                ]
                found = False
                for p in possible_paths:
                    if p:
                        alt_dir = (orig_base_dir / p).resolve()
                        if alt_dir.exists():
                            img_dir = alt_dir
                            found = True
                            break
                if not found:
                    log.error(f"  Gagal menemukan direktori asli untuk split '{split_name}'.")
                    log.error(f"  Pencarian gagal di: {(orig_base_dir / rel_path).resolve()}")
                    continue
            
        log.info(f"  Memproses split '{split_name}' dari {img_dir}...")
        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
        
        cropped_count = 0
        
        for img_path in img_files:
            # Path label asli
            label_dir = img_path.parent.parent / "labels"
            label_path = label_dir / f"{img_path.stem}.txt"
            
            orig_labels = []
            if label_path.exists():
                with open(label_path, "r", encoding="utf-8") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:])
                            orig_labels.append((cls_id, cx, cy, w, h))
            
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W, _ = img.shape
            
            # Deteksi person (class 0 pada model kustom atau pretrained)
            results = person_model(img, classes=0, conf=0.25, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
            
            # Tambahkan fallback person boxes dari APD unassigned
            fallback_boxes = cluster_unassigned_ppe(orig_labels, boxes, W, H)
            if fallback_boxes:
                boxes.extend(fallback_boxes)
            
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Pastikan koordinat dasar dalam batas gambar
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                
                box_w = x2 - x1
                box_h = y2 - y1
                
                # Tambahkan padding 10% di setiap sisi (total 20% lebar dan tinggi)
                # Padding mencegah bagian kepala (helmet) dan kaki (safety shoes) terpotong di tepi gambar crop
                pad_w = int(box_w * 0.10)
                pad_h = int(box_h * 0.10)
                
                x1_pad = max(0, x1 - pad_w)
                y1_pad = max(0, y1 - pad_h)
                x2_pad = min(W, x2 + pad_w)
                y2_pad = min(H, y2 + pad_h)
                
                crop_w = x2_pad - x1_pad
                crop_h = y2_pad - y1_pad
                if crop_w < 15 or crop_h < 15:
                    continue
                    
                # Crop gambar berdasarkan koordinat yang ber-padding
                crop_img = img[y1_pad:y2_pad, x1_pad:x2_pad]
                
                # Relokalisasi label PPE ke dalam crop person
                crop_labels = []
                for cls_id, cx, cy, w, h in orig_labels:
                    # Koordinat pixel absolut dari label PPE
                    px_center = cx * W
                    py_center = cy * H
                    pw = w * W
                    ph = h * H
                    
                    # Cek apakah center PPE berada di dalam bounding box person ASLI (sebelum padding)
                    # Ini mencegah mengambil PPE milik orang lain yang terdeteksi di area padding
                    if x1 <= px_center <= x2 and y1 <= py_center <= y2:
                        # Hitung koordinat relatif terhadap crop ber-padding
                        px1_new = max(px_center - pw/2, x1_pad) - x1_pad
                        py1_new = max(py_center - ph/2, y1_pad) - y1_pad
                        px2_new = min(px_center + pw/2, x2_pad) - x1_pad
                        py2_new = min(py_center + ph/2, y2_pad) - y1_pad
                        
                        # New center and dimensions in crop coordinates
                        cx_new = (px1_new + px2_new) / 2
                        cy_new = (py1_new + py2_new) / 2
                        w_new = px2_new - px1_new
                        h_new = py2_new - py1_new
                        
                        # Normalisasi terhadap ukuran crop ber-padding
                        cx_norm = cx_new / crop_w
                        cy_norm = cy_new / crop_h
                        w_norm = w_new / crop_w
                        h_norm = h_new / crop_h
                        
                        crop_labels.append((cls_id, cx_norm, cy_norm, w_norm, h_norm))
                        
                # Oversampling dinonaktifkan karena dataset asli sudah diaugmentasi di Roboflow
                oversample_factor = 1

                for r in range(oversample_factor):
                    suffix = f"_r{r}" if r > 0 else ""
                    crop_img_name = f"{img_path.stem}_person_{idx}{suffix}.jpg"
                    crop_label_name = f"{img_path.stem}_person_{idx}{suffix}.txt"
                    
                    crop_img_out = cropped_dir / split_name / "images" / crop_img_name
                    crop_lbl_out = cropped_dir / split_name / "labels" / crop_label_name
                    
                    # Resize crop agar maksimal berukuran CONFIG["imgsz"] (640px)
                    # Ini mencegah Out Of Memory pada OpenCV dan mempercepat training
                    crop_h_orig, crop_w_orig = crop_img.shape[:2]
                    max_crop_size = CONFIG.get("imgsz", 640)
                    if max(crop_h_orig, crop_w_orig) > max_crop_size:
                        scale = max_crop_size / max(crop_h_orig, crop_w_orig)
                        crop_img_to_save = cv2.resize(
                            crop_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                        )
                    else:
                        crop_img_to_save = crop_img
                        
                    cv2.imwrite(str(crop_img_out), crop_img_to_save)
                    with open(crop_lbl_out, "w", encoding="utf-8") as out_f:
                        for lbl in crop_labels:
                            out_f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")
                            
                    cropped_count += 1
                
        log.info(f"  ✓ Split '{split_name}': berhasil membuat {cropped_count} potong gambar person.")
        
    # Buat data_cropped.yaml dengan path absolut agar YOLOv8 tidak salah mencari direktori
    cropped_yaml_content = {
        "path": str(cropped_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(data_cfg["names"]),
        "names": data_cfg["names"]
    }
    
    with open(cropped_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cropped_yaml_content, f, default_flow_style=False)
        
    log.info(f"  ✓ File konfigurasi dataset baru berhasil ditulis di: {cropped_yaml_path.resolve()}")
    log.info("=" * 60)
    
    return str(cropped_yaml_path)


def find_latest_run(project_name: str) -> Path | None:
    """
    Temukan direktori run YOLO terbaru secara robust.
    Mendukung path absolute dan pencarian recursive (rglob) jika struktur folder terduplikasi.
    """
    base_dir = Path(CONFIG["runs_dir"]).resolve()
    
    # 1. Cari di {runs_dir}/detect/
    base_detect = base_dir / "detect"
    if base_detect.exists():
        candidates = sorted(
            base_detect.glob(f"{project_name}*"),
            key=lambda p: p.stat().st_mtime
        )
        if candidates:
            return candidates[-1]
            
    # 2. Cari langsung di {runs_dir}/
    candidates = sorted(
        base_dir.glob(f"{project_name}*"),
        key=lambda p: p.stat().st_mtime
    )
    if candidates:
        return candidates[-1]
        
    # 3. Fallback: Cari secara rekursif di seluruh {runs_dir} (mengatasi runs/detect/runs dll.)
    candidates = sorted(
        base_dir.rglob(f"**/{project_name}*"),
        key=lambda p: p.stat().st_mtime
    )
    return candidates[-1] if candidates else None


def load_results_csv(run_dir: Path) -> dict[str, list]:
    """
    Membaca results.csv yang dihasilkan YOLO secara otomatis.
    Mengembalikan dict: {metric_name: [list of float per epoch]}
    """
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        log.error(f"  results.csv tidak ditemukan di: {run_dir}")
        return {}

    data: dict[str, list] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                key = key.strip()
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(val))
                except ValueError:
                    data[key].append(None)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISASI GRAFIK
# ─────────────────────────────────────────────────────────────────────────────
PLOT_STYLE = {
    "train" : {"color": "#2196F3", "lw": 2.0},   # biru
    "val"   : {"color": "#F44336", "lw": 2.0},   # merah
    "metric": {"color": "#4CAF50", "lw": 2.0},   # hijau
}

def _smooth(values: list[float], weight: float = 0.6) -> list[float]:
    """Exponential moving average untuk kurva lebih mulus."""
    smoothed, last = [], values[0] if values else 0
    for v in values:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return smoothed


def _plot_metric(ax, epochs, values, label, style_key, smooth=True, ylabel=""):
    """Helper: plot satu kurva ke axes."""
    valid_idx = [i for i, v in enumerate(values) if v is not None]
    if not valid_idx:
        return
    ep  = [epochs[i] for i in valid_idx]
    val = [values[i] for i in valid_idx]

    st = PLOT_STYLE[style_key]
    if smooth and len(val) > 5:
        ax.plot(ep, _smooth(val), **st, label=f"{label} (smooth)")
        ax.plot(ep, val, color=st["color"], lw=0.6, alpha=0.35)
    else:
        ax.plot(ep, val, **st, label=label)

    ax.set_ylabel(ylabel or label, fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Epoch", fontsize=9)


def generate_loss_graphs(data: dict, graphs_dir: Path, epochs: list[int]) -> None:
    """
    Grafik 1 — Loss Curves (3 loss train + 3 loss val)
    Metric keys standar YOLOv8: train/box_loss, train/cls_loss, train/dfl_loss
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("PROTMIND APD — Loss Curves", fontsize=14, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    loss_pairs = [
        ("train/box_loss", "val/box_loss", "Box Loss",  0, 0),
        ("train/cls_loss", "val/cls_loss", "Class Loss", 0, 1),
        ("train/dfl_loss", "val/dfl_loss", "DFL Loss",  0, 2),
    ]

    for train_key, val_key, title, row, col in loss_pairs:
        ax = fig.add_subplot(gs[row, col])
        ax.set_title(title, fontsize=10, fontweight="bold")
        if train_key in data:
            _plot_metric(ax, epochs, data[train_key], "Train", "train", ylabel="Loss")
        if val_key in data:
            _plot_metric(ax, epochs, data[val_key],   "Val",   "val",   ylabel="Loss")

    # Row 2: combined overview
    ax_box = fig.add_subplot(gs[1, 0])
    ax_box.set_title("Total Box Loss (Train vs Val)", fontsize=10, fontweight="bold")
    for key, lbl, sk in [("train/box_loss","Train","train"),("val/box_loss","Val","val")]:
        if key in data:
            _plot_metric(ax_box, epochs, data[key], lbl, sk, ylabel="Loss")

    ax_cls = fig.add_subplot(gs[1, 1])
    ax_cls.set_title("Total Cls Loss (Train vs Val)", fontsize=10, fontweight="bold")
    for key, lbl, sk in [("train/cls_loss","Train","train"),("val/cls_loss","Val","val")]:
        if key in data:
            _plot_metric(ax_cls, epochs, data[key], lbl, sk, ylabel="Loss")

    ax_dfl = fig.add_subplot(gs[1, 2])
    ax_dfl.set_title("Total DFL Loss (Train vs Val)", fontsize=10, fontweight="bold")
    for key, lbl, sk in [("train/dfl_loss","Train","train"),("val/dfl_loss","Val","val")]:
        if key in data:
            _plot_metric(ax_dfl, epochs, data[key], lbl, sk, ylabel="Loss")

    out = graphs_dir / "01_loss_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  [Grafik] Loss Curves → {out}")


def generate_metric_graphs(data: dict, graphs_dir: Path, epochs: list[int]) -> None:
    """
    Grafik 2 — Performance Metrics
    Precision, Recall, mAP50, mAP50-95
    """
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("PROTMIND APD — Performance Metrics", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    metric_map = {
        "metrics/precision(B)" : ("Precision",    0),
        "metrics/recall(B)"    : ("Recall",        1),
        "metrics/mAP50(B)"     : ("mAP@50",        2),
        "metrics/mAP50-95(B)"  : ("mAP@50-95",     3),
    }

    for key, (title, col) in metric_map.items():
        ax = fig.add_subplot(gs[0, col])
        ax.set_title(title, fontsize=10, fontweight="bold")
        if key in data:
            _plot_metric(ax, epochs, data[key], title, "metric", ylabel="Score")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.9, color="orange", lw=1, linestyle="--", alpha=0.6, label="Target 0.9")
        ax.legend(fontsize=7)

    out = graphs_dir / "02_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  [Grafik] Performance Metrics → {out}")


def generate_lr_graph(data: dict, graphs_dir: Path, epochs: list[int]) -> None:
    """Grafik 3 — Learning Rate Schedule."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("PROTMIND APD — Learning Rate Schedule", fontsize=13, fontweight="bold")

    lr_keys = ["lr/pg0", "lr/pg1", "lr/pg2"]
    lr_labels = ["LR Group 0 (backbone)", "LR Group 1 (neck)", "LR Group 2 (head)"]
    colors    = ["#9C27B0", "#FF9800", "#009688"]

    for ax, key, label, color in zip(axes, lr_keys, lr_labels, colors):
        ax.set_title(label, fontsize=9)
        if key in data:
            valid = [(epochs[i], data[key][i]) for i in range(len(epochs)) if data[key][i] is not None]
            if valid:
                ep_v, lr_v = zip(*valid)
                ax.plot(ep_v, lr_v, color=color, lw=2)
        ax.set_ylabel("Learning Rate", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = graphs_dir / "03_learning_rate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  [Grafik] Learning Rate → {out}")


def generate_summary_dashboard(data: dict, graphs_dir: Path, epochs: list[int]) -> None:
    """
    Grafik 4 — Dashboard Ringkasan (1 halaman, semua metrik kunci).
    Cocok untuk ditempelkan di laporan Tugas Akhir.
    """
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(
        "PROTMIND — Model Training Dashboard\nYOLOv8n | Dataset APD 6 Kelas",
        fontsize=15, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)

    panels = [
        # (row, col, colspan, train_key, val_key, title, ylabel, ylim)
        (0, 0, 1, "train/box_loss", "val/box_loss",  "Box Loss",   "Loss",  None),
        (0, 1, 1, "train/cls_loss", "val/cls_loss",  "Class Loss", "Loss",  None),
        (0, 2, 1, "train/dfl_loss", "val/dfl_loss",  "DFL Loss",   "Loss",  None),
        (0, 3, 1, None, "metrics/precision(B)",       "Precision",  "Score", (0,1.05)),
        (1, 0, 1, None, "metrics/recall(B)",           "Recall",     "Score", (0,1.05)),
        (1, 1, 1, None, "metrics/mAP50(B)",            "mAP@50",     "Score", (0,1.05)),
        (1, 2, 1, None, "metrics/mAP50-95(B)",         "mAP@50-95",  "Score", (0,1.05)),
        (1, 3, 1, "lr/pg0", None,                       "LR (pg0)",   "LR",    None),
    ]

    for (row, col, span, tkey, vkey, title, ylabel, ylim) in panels:
        ax = fig.add_subplot(gs[row, col:col+span])
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.25)
        if ylim:
            ax.set_ylim(*ylim)

        if tkey and tkey in data:
            _plot_metric(ax, epochs, data[tkey], "Train", "train", ylabel=ylabel)
        if vkey and vkey in data:
            style = "val" if (tkey and tkey in data) else "metric"
            _plot_metric(ax, epochs, data[vkey], vkey.split("/")[-1].split("(")[0], style, ylabel=ylabel)

    # Row 3: span semua kolom untuk tabel ringkasan metrik akhir
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")

    final_metrics = {}
    for key in ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]:
        if key in data:
            vals = [v for v in data[key] if v is not None]
            if vals:
                final_metrics[key.split("/")[-1].replace("(B)","")] = f"{vals[-1]:.4f}"

    if final_metrics:
        col_labels = list(final_metrics.keys())
        cell_text  = [list(final_metrics.values())]
        table = ax_table.table(
            cellText   = cell_text,
            colLabels  = col_labels,
            loc        = "center",
            cellLoc    = "center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(2.0, 2.5)
        ax_table.set_title("Metrik Akhir Pelatihan (Epoch Terakhir)", fontsize=11, fontweight="bold", pad=10)

    out = graphs_dir / "dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  [Grafik] Dashboard → {out}")


def generate_all_graphs(run_dir: Path, timestamp: str = None) -> None:
    """Orkestrasi semua pembuatan grafik setelah training selesai."""
    log.info("")
    log.info("=" * 60)
    log.info("  ANALISIS GRAFIK PELATIHAN")
    log.info("=" * 60)

    # 1. Tentukan direktori grafik default (models/training_graphs)
    graphs_dir = Path(CONFIG["graphs_dir"])
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Tentukan direktori grafik ber-timestamp jika ada
    graphs_ts_dir = None
    if timestamp:
        graphs_ts_dir = graphs_dir.parent / f"training_graphs_{timestamp}"
        graphs_ts_dir.mkdir(parents=True, exist_ok=True)

    data = load_results_csv(run_dir)
    if not data:
        log.error("  Tidak ada data untuk divisualisasikan.")
        return

    # Buat daftar epoch
    epoch_key = "epoch" if "epoch" in data else list(data.keys())[0]
    n_epochs   = len(data[epoch_key])
    epochs     = list(range(1, n_epochs + 1))

    # Sesuai permintaan pengguna: hanya grafik kustom dashboard saja
    generate_summary_dashboard(data, graphs_dir, epochs)
    if graphs_ts_dir:
        generate_summary_dashboard(data, graphs_ts_dir, epochs)

    # Salin grafik bawaan YOLOv8 jika tersedia
    yolo_graphs = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "F1_curve.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "results.png"
    ]
    for g in yolo_graphs:
        src_g = run_dir / g
        if src_g.exists():
            try:
                # Salin ke folder default
                shutil.copy2(src_g, graphs_dir / g)
                # Salin ke folder ber-timestamp
                if graphs_ts_dir:
                    shutil.copy2(src_g, graphs_ts_dir / g)
                log.info(f"  [Grafik] Berhasil menyalin grafik bawaan YOLOv8: {g}")
            except Exception as e:
                log.warning(f"  [Grafik] Gagal menyalin grafik {g}: {e}")

    log.info("")
    log.info(f"  ✓ Grafik default tersimpan di: {graphs_dir.resolve()}")
    if graphs_ts_dir:
        log.info(f"  ✓ Grafik versi baru tersimpan di: {graphs_ts_dir.resolve()}")
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING UTAMA
# ─────────────────────────────────────────────────────────────────────────────
def run_training() -> Path:
    """Eksekusi pelatihan YOLOv8n dan kembalikan path direktori run hasil training."""
    # Tentukan bobot model (cek jika ingin resume)
    weights_path = CONFIG["model_weights"]
    is_resuming = False
    
    if CONFIG.get("resume", False):
        last_path = Path(CONFIG["runs_dir"]) / CONFIG["project_name"] / "weights" / "last.pt"
        if last_path.exists():
            weights_path = str(last_path)
            is_resuming = True
            log.info(f"  [Resume] Melanjutkan pelatihan APD dari checkpoint: {last_path}")
        else:
            # Fallback pencarian rekursif jika nama folder diubah otomatis oleh YOLO (misal protmind_balanced_v3-2)
            candidates = list(Path(CONFIG["runs_dir"]).rglob("**/weights/last.pt"))
            # Filter kandidat untuk yang tidak memiliki _person di namanya
            candidates = [c for c in candidates if "_person" not in str(c)]
            if candidates:
                # Ambil kandidat paling baru berdasarkan waktu modifikasi
                candidates.sort(key=lambda x: x.stat().st_mtime)
                last_path = candidates[-1]
                weights_path = str(last_path)
                is_resuming = True
                log.info(f"  [Resume] Checkpoint ditemukan lewat pencarian fallback: {last_path}")
            else:
                log.warning(f"  [Resume] Checkpoint APD tidak ditemukan. Pelatihan dimulai dari awal.")

    log.info(f"  Memuat weights: {weights_path}")
    model = YOLO(weights_path)

    log.info("  Memulai pelatihan...")
    start_time = time.time()

    # Gunakan absolute path untuk project agar YOLOv8 tidak menduplikasi folder secara kacau
    project_abs_path = str(Path(CONFIG["runs_dir"]).resolve())

    model.train(
        data          = CONFIG["data_yaml"],
        epochs        = CONFIG["epochs"],
        imgsz         = CONFIG["imgsz"],
        batch         = CONFIG["batch"],
        patience      = CONFIG["patience"],
        lr0           = CONFIG["lr0"],
        lrf           = CONFIG["lrf"],
        momentum      = CONFIG["momentum"],
        weight_decay  = CONFIG["weight_decay"],
        warmup_epochs = CONFIG["warmup_epochs"],
        augment       = CONFIG["augment"],
        hsv_h         = CONFIG["hsv_h"],
        hsv_s         = CONFIG["hsv_s"],
        hsv_v         = CONFIG["hsv_v"],
        flipud        = CONFIG["flipud"],
        fliplr        = CONFIG["fliplr"],
        mosaic        = CONFIG["mosaic"],
        mixup         = CONFIG["mixup"],
        degrees       = CONFIG.get("degrees", 0.0),
        translate     = CONFIG.get("translate", 0.1),
        scale         = CONFIG.get("scale", 0.5),
        shear         = CONFIG.get("shear", 0.0),
        device        = CONFIG["device"],
        workers       = CONFIG["workers"],          # 0 di Windows → pin_memory otomatis nonaktif
        amp           = CONFIG["amp"],
        seed          = CONFIG["seed"],
        freeze        = CONFIG["freeze"],           # Bekukan layer backbone awal untuk transfer learning lebih stabil
        cls           = CONFIG["cls"],              # Tingkatkan bobot klasifikasi untuk menyeimbangkan kelas minoritas
        cos_lr        = CONFIG.get("cos_lr", False), # Gunakan Cosine Learning Rate scheduler
        label_smoothing = CONFIG.get("label_smoothing", 0.0), # Gunakan Label Smoothing untuk mencegah overfitting
        project       = project_abs_path,
        name          = CONFIG["project_name"],
        exist_ok      = True,
        verbose       = True,
        save          = True,
        save_period   = 10,         # simpan checkpoint setiap 10 epoch
        val           = True,
        plots         = True,       # YOLO akan buat plot bawaannya juga
        resume        = is_resuming
    )

    elapsed = time.time() - start_time
    hours, rem = divmod(int(elapsed), 3600)
    mins, secs  = divmod(rem, 60)
    log.info(f"  Training selesai dalam: {hours}j {mins}m {secs}d")

    # Ambil save_dir langsung dari trainer (keandalan 100%)
    if hasattr(model, "trainer") and model.trainer is not None and hasattr(model.trainer, "save_dir"):
        run_dir = Path(model.trainer.save_dir)
        log.info(f"  [Trainer] Direktori run dideteksi dari trainer: {run_dir.resolve()}")
        return run_dir

    # Fallback jika trainer tidak memiliki save_dir
    fallback_dir = find_latest_run(CONFIG["project_name"])
    if fallback_dir:
        log.info(f"  [Trainer] Direktori run dideteksi lewat pencarian fallback: {fallback_dir.resolve()}")
        return fallback_dir

    raise RuntimeError("Gagal menentukan direktori hasil pelatihan.")


def copy_best_model(run_dir: Path, timestamp: str = None) -> None:
    """Salin best.pt ke direktori models/ dan folder grafik bertanggal agar mudah diakses."""
    src = run_dir / "weights" / "best.pt"
    
    if not src.exists():
        log.error(f"  best.pt tidak ditemukan di: {src}")
        return
        
    # 1. Salin ke default path (models/best_ppe.pt)
    dst_default = Path(CONFIG["output_model"])
    dst_default.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_default)
    size_mb = dst_default.stat().st_size / 1e6
    log.info(f"  ✓ Model terbaik disalin ke default: {dst_default.resolve()} ({size_mb:.1f} MB)")
    
    # 2. Salin ke path ber-timestamp jika ada
    if timestamp:
        dst_ts = dst_default.parent / f"best_ppe_{timestamp}.pt"
        shutil.copy2(src, dst_ts)
        log.info(f"  ✓ Model terbaik versi baru disimpan: {dst_ts.resolve()}")
        
        # 3. Salin ke folder training_graphs_{timestamp}/best.pt
        dst_graph_dir = Path(CONFIG["graphs_dir"]).parent / f"training_graphs_{timestamp}"
        dst_graph_dir.mkdir(parents=True, exist_ok=True)
        dst_graph_best = dst_graph_dir / "best.pt"
        shutil.copy2(src, dst_graph_best)
        log.info(f"  ✓ Model terbaik disalin ke folder grafik timestamp: {dst_graph_best.resolve()}")


def evaluate_best_model(model_path: str, data_yaml: str):
    """
    Menjalankan validasi pada model terbaik untuk mengekstrak metrik per kelas secara mendalam,
    terutama memantau kelas minoritas 'no-safety shoes'.
    """
    log.info("")
    log.info("=" * 60)
    log.info("  EVALUASI MENDALAM MODEL TERBAIK PER KELAS")
    log.info("=" * 60)
    try:
        model = YOLO(model_path)
        metrics = model.val(data=data_yaml, split="val", verbose=False)
        
        # Ambil daftar nama kelas
        class_names = metrics.names
        
        # Ambil metrik per kelas
        log.info(f"{'Kelas':<20} | {'Precision':<10} | {'Recall':<10} | {'mAP50':<10} | {'mAP50-95':<10}")
        log.info("-" * 68)
        
        for i, name in class_names.items():
            p = metrics.box.p[i] if i < len(metrics.box.p) else 0.0
            r = metrics.box.r[i] if i < len(metrics.box.r) else 0.0
            ap50 = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0.0
            ap50_95 = metrics.box.ap[i] if i < len(metrics.box.ap) else 0.0
            
            log.info(f"{name:<20} | {p:<10.4f} | {r:<10.4f} | {ap50:<10.4f} | {ap50_95:<10.4f}")
            
            # Peringatan khusus untuk kelas dengan ap50 < 0.8
            if ap50 < 0.8:
                log.warning(f"  [PERINGATAN] Kelas '{name}' memiliki mAP50 ({ap50:.4f}) di bawah target 85%!")
                
        # Tampilkan ringkasan overall
        log.info("-" * 68)
        log.info(f"{'OVERALL':<20} | {metrics.box.mp:<10.4f} | {metrics.box.mr:<10.4f} | {metrics.box.map50:<10.4f} | {metrics.box.map:<10.4f}")
        log.info("=" * 60)
    except Exception as e:
        log.error(f"Gagal menjalankan evaluasi per kelas: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Protmind YOLOv8 Training Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Hanya melakukan preprocessing dataset, cropping, dan pengecekan tanpa pelatihan.")
    parser.add_argument("--epochs", type=int, default=None, help="Jumlah epoch pelatihan model APD/Stage 2 (default dari CONFIG: 100).")
    parser.add_argument("--epochs-person", type=int, default=None, help="Jumlah epoch pelatihan model Person/Stage 1 (default dari CONFIG: 50).")
    parser.add_argument("--resume", action="store_true", help="Melanjutkan pelatihan model APD dari checkpoint last.pt.")
    parser.add_argument("--resume-person", action="store_true", help="Melanjutkan pelatihan model Person dari checkpoint last.pt.")
    parser.add_argument("--train-person", action="store_true", help="Paksa latih ulang model Person Detector (Stage 1) bahkan jika models/best_person.pt sudah ada.")
    parser.add_argument("--recreate", action="store_true", help="Paksa regenerasi dataset cropped.")
    args = parser.parse_args()

    # Perbarui konfigurasi dinamis berdasarkan argumen CLI
    if args.epochs is not None:
        CONFIG["epochs"] = args.epochs
        log.info(f"  [CLI Override] Epoch APD (Stage 2) diatur ke: {args.epochs}")
    if args.epochs_person is not None:
        CONFIG["epochs_person"] = args.epochs_person
        log.info(f"  [CLI Override] Epoch Person (Stage 1) diatur ke: {args.epochs_person}")
    if args.resume:
        CONFIG["resume"] = True
        log.info("  [CLI Override] Mode Resume diaktifkan untuk model APD (Stage 2).")
    if args.resume_person:
        CONFIG["resume_person"] = True
        log.info("  [CLI Override] Mode Resume diaktifkan untuk model Person (Stage 1).")
    if args.recreate:
        CONFIG["force_recreate"] = True
        log.info("  [CLI Override] Paksa regenerasi dataset cropped diaktifkan.")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info(f"  Sesi dimulai: {timestamp}")

    # 1. Validasi lingkungan
    verify_environment()

    # Cek mode dry-run (hanya penyiapan dataset/cropping)
    if args.dry_run:
        log.info("  [Dry-Run] Menjalankan uji coba penyiapan dataset...")
        if CONFIG.get("train_person", False):
            prepare_person_dataset()
        if CONFIG["two_stage"]:
            prepare_cropped_dataset()
        log.info("")
        log.info("=" * 60)
        log.info("  ✓ DRY RUN PENYIAPAN DATASET SELESAI")
        log.info("  Dataset person dan cropped person siap digunakan untuk pelatihan asli.")
        log.info("=" * 60)
        sys.exit(0)

    # 2. Latih model Stage 1 (Person Detector) jika dikonfigurasi
    if CONFIG.get("train_person", False):
        custom_person_path = Path(CONFIG["output_person_model"])
        if custom_person_path.exists() and not args.train_person:
            log.info(f"  [Stage 1] Model person kustom dideteksi di: {custom_person_path.resolve()}")
            log.info("  Melewati pelatihan Stage 1. Menggunakan model person kustom yang sudah ada untuk cropping.")
        else:
            person_yaml = prepare_person_dataset()
            CONFIG["person_yaml"] = person_yaml
            train_person_model()

    # 3. Jalankan pemotongan dataset jika mode Two-Stage aktif
    if CONFIG["two_stage"]:
        cropped_yaml = prepare_cropped_dataset()
        CONFIG["data_yaml"] = cropped_yaml

    # 4. Jalankan training dan dapatkan direktori hasil secara langsung
    run_dir = run_training()
    log.info(f"  Direktori run aktif: {run_dir.resolve()}")
    
    # Hasilkan timestamp untuk versi unik model dan grafik
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 5. Salin best.pt ke models/ (sebagai file default dan file ber-timestamp)
    copy_best_model(run_dir, timestamp=run_timestamp)

    # Evaluasi kelas minoritas secara detail
    evaluate_best_model(CONFIG["output_model"], CONFIG["data_yaml"])

    # 6. Hasilkan semua grafik analisis (di folder default dan folder ber-timestamp)
    generate_all_graphs(run_dir, timestamp=run_timestamp)

    log.info("")
    log.info("=" * 60)
    log.info("  ✓ PIPELINE TRAINING PROTMIND SELESAI")
    log.info(f"  Model PPE siap deploy: {CONFIG['output_model']}")
    if CONFIG.get("train_person", False):
        log.info(f"  Model Person siap deploy: {CONFIG['output_person_model']}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()