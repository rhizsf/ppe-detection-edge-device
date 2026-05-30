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
    "two_stage"      : True,                    # True = Latih model pada cropped person (Stage 2)
    "original_yaml"  : "assets/dataset/data.yaml", # Path dataset asli
    "cropped_yaml"   : "assets/dataset_cropped/data_cropped.yaml", # Path dataset cropped hasil pemrosesan
    "data_yaml"      : "assets/dataset_cropped/data_cropped.yaml", # Target training (akan diperbarui otomatis)
    "model_weights"  : "yolov8n.pt",            # pretrained backbone dari Ultralytics
    "output_model"   : "models/best_ppe.pt",    # tujuan salin model terbaik hasil training
    "runs_dir"       : "runs",                  # ← FIXED: jangan pakai "runs/detect"
    "graphs_dir"     : "models/training_graphs",# direktori simpan grafik analisis

    # Hiperparameter Pelatihan
    "epochs"         : 100,
    "imgsz"          : 640,
    "batch"          : 16,                      # turunkan ke 8 jika VRAM tidak cukup
    "patience"       : 20,                      # early stopping
    "lr0"            : 0.01,                    # learning rate awal
    "lrf"            : 0.001,                   # learning rate akhir (cosine decay)
    "momentum"       : 0.937,
    "weight_decay"   : 0.0005,
    "warmup_epochs"  : 3,
    "project_name"   : "protmind_apdv2",        # nama subfolder

    # Augmentasi — aktif untuk dataset konstruksi (kondisi cahaya bervariasi)
    "augment"        : True,
    "hsv_h"          : 0.015,
    "hsv_s"          : 0.7,
    "hsv_v"          : 0.4,
    "flipud"         : 0.1,
    "fliplr"         : 0.5,
    "mosaic"         : 1.0,
    "mixup"          : 0.1,

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
    
    if cropped_yaml_path.exists():
        log.info(f"  [Preprocess] Dataset cropped sudah ada di: {cropped_yaml_path.resolve()}")
        log.info("  Melewati proses cropping. Jika ingin regenerasi, silakan hapus folder 'assets/dataset_cropped'.")
        return str(cropped_yaml_path)
        
    log.info("=" * 60)
    log.info("  [Preprocess] MENYIAPKAN DATASET DENGAN CROPPING PERSON (STAGE 1)")
    log.info("=" * 60)
    
    # Load original data.yaml
    with open(orig_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
        
    orig_base_dir = orig_yaml_path.parent
    
    # Inisialisasi model person detector (Stage 1)
    log.info("  Memuat model YOLOv8n untuk deteksi person...")
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
            
            # Deteksi person (class 0 pada COCO)
            results = person_model(img, classes=0, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
            
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Pastikan koordinat dalam batas gambar
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                
                crop_w = x2 - x1
                crop_h = y2 - y1
                if crop_w < 15 or crop_h < 15:
                    continue
                    
                # Crop gambar
                crop_img = img[y1:y2, x1:x2]
                
                # Relokalisasi label PPE ke dalam crop person
                crop_labels = []
                for cls_id, cx, cy, w, h in orig_labels:
                    # Koordinat pixel absolut dari label PPE
                    px_center = cx * W
                    py_center = cy * H
                    pw = w * W
                    ph = h * H
                    
                    # Cek apakah center PPE berada di dalam bounding box person
                    if x1 <= px_center <= x2 and y1 <= py_center <= y2:
                        # Top-left and bottom-right relative to crop
                        px1_new = max(px_center - pw/2, x1) - x1
                        py1_new = max(py_center - ph/2, y1) - y1
                        px2_new = min(px_center + pw/2, x2) - x1
                        py2_new = min(py_center + ph/2, y2) - y1
                        
                        # New center and dimensions in crop coordinates
                        cx_new = (px1_new + px2_new) / 2
                        cy_new = (py1_new + py2_new) / 2
                        w_new = px2_new - px1_new
                        h_new = py2_new - py1_new
                        
                        # Normalisasi terhadap ukuran crop
                        cx_norm = cx_new / crop_w
                        cy_norm = cy_new / crop_h
                        w_norm = w_new / crop_w
                        h_norm = h_new / crop_h
                        
                        crop_labels.append((cls_id, cx_norm, cy_norm, w_norm, h_norm))
                        
                # Simpan crop gambar person
                crop_img_name = f"{img_path.stem}_person_{idx}.jpg"
                crop_label_name = f"{img_path.stem}_person_{idx}.txt"
                
                crop_img_out = cropped_dir / split_name / "images" / crop_img_name
                crop_lbl_out = cropped_dir / split_name / "labels" / crop_label_name
                
                cv2.imwrite(str(crop_img_out), crop_img)
                
                with open(crop_lbl_out, "w", encoding="utf-8") as out_f:
                    for lbl in crop_labels:
                        out_f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")
                        
                cropped_count += 1
                
        log.info(f"  ✓ Split '{split_name}': berhasil membuat {cropped_count} potong gambar person.")
        
    # Buat data_cropped.yaml
    cropped_yaml_content = {
        "train": "../train/images",
        "val": "../val/images",
        "test": "../test/images",
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
    Temukan direktori run YOLO terbaru.
    YOLO menyimpan hasil di: {runs_dir}/detect/{project_name}/
    """
    base_detect = Path(CONFIG["runs_dir"]) / "detect"
    if base_detect.exists():
        candidates = sorted(
            base_detect.glob(f"{project_name}*"),
            key=lambda p: p.stat().st_mtime
        )
        if candidates:
            return candidates[-1]
    base = Path(CONFIG["runs_dir"])
    candidates = sorted(base.glob(f"{project_name}*"), key=lambda p: p.stat().st_mtime)
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
        "PROTMIND — Model Training Dashboard\nYOLOv8n | Dataset APD 7 Kelas",
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

    out = graphs_dir / "04_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info(f"  [Grafik] Dashboard → {out}")


def generate_all_graphs(run_dir: Path) -> None:
    """Orkestrasi semua pembuatan grafik setelah training selesai."""
    log.info("")
    log.info("=" * 60)
    log.info("  ANALISIS GRAFIK PELATIHAN")
    log.info("=" * 60)

    graphs_dir = Path(CONFIG["graphs_dir"])
    graphs_dir.mkdir(parents=True, exist_ok=True)

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
                shutil.copy2(src_g, graphs_dir / g)
                log.info(f"  [Grafik] Berhasil menyalin grafik bawaan YOLOv8: {g}")
            except Exception as e:
                log.warning(f"  [Grafik] Gagal menyalin grafik {g}: {e}")

    log.info("")
    log.info(f"  ✓ Semua grafik tersimpan di: {graphs_dir.resolve()}")
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING UTAMA
# ─────────────────────────────────────────────────────────────────────────────
def run_training() -> Path:
    """Eksekusi pelatihan YOLOv8n dan kembalikan path direktori run."""
    log.info("  Memuat pretrained weights YOLOv8n...")
    model = YOLO(CONFIG["model_weights"])

    log.info("  Memulai pelatihan...")
    start_time = time.time()

    results = model.train(
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
        device        = CONFIG["device"],
        workers       = CONFIG["workers"],          # 0 di Windows → pin_memory otomatis nonaktif
        amp           = CONFIG["amp"],
        seed          = CONFIG["seed"],
        project       = CONFIG["runs_dir"],
        name          = CONFIG["project_name"],
        exist_ok      = False,
        verbose       = True,
        save          = True,
        save_period   = 10,         # simpan checkpoint setiap 10 epoch
        val           = True,
        plots         = True,       # YOLO akan buat plot bawaannya juga
    )

    elapsed = time.time() - start_time
    hours, rem = divmod(int(elapsed), 3600)
    mins, secs  = divmod(rem, 60)
    log.info(f"  Training selesai dalam: {hours}j {mins}m {secs}d")
    return results


def copy_best_model(run_dir: Path) -> None:
    """Salin best.pt ke direktori models/ agar mudah diakses."""
    src = run_dir / "weights" / "best.pt"
    dst = Path(CONFIG["output_model"])
    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.exists():
        shutil.copy2(src, dst)
        size_mb = dst.stat().st_size / 1e6
        log.info(f"  ✓ Model terbaik disalin: {dst.resolve()} ({size_mb:.1f} MB)")
    else:
        log.error(f"  best.pt tidak ditemukan di: {src}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info(f"  Sesi dimulai: {timestamp}")

    # 1. Validasi lingkungan
    verify_environment()

    # Cek mode dry-run (hanya penyiapan dataset/cropping)
    if "--dry-run" in sys.argv:
        log.info("  [Dry-Run] Menjalankan uji coba penyiapan dataset...")
        if CONFIG["two_stage"]:
            prepare_cropped_dataset()
        log.info("")
        log.info("=" * 60)
        log.info("  ✓ DRY RUN PENYIAPAN DATASET SELESAI")
        log.info("  Dataset cropped person siap digunakan untuk pelatihan asli.")
        log.info("=" * 60)
        sys.exit(0)

    # 2. Jalankan pemotongan dataset jika mode Two-Stage aktif
    if CONFIG["two_stage"]:
        cropped_yaml = prepare_cropped_dataset()
        CONFIG["data_yaml"] = cropped_yaml

    # 3. Jalankan training
    run_training()

    # 4. Temukan direktori run hasil training
    run_dir = find_latest_run(CONFIG["project_name"])
    if run_dir is None:
        log.error("  Direktori run tidak ditemukan setelah training.")
        sys.exit(1)
    log.info(f"  Direktori run: {run_dir.resolve()}")

    # 5. Salin best.pt ke models/
    copy_best_model(run_dir)

    # 6. Hasilkan semua grafik analisis
    generate_all_graphs(run_dir)

    log.info("")
    log.info("=" * 60)
    log.info("  ✓ PIPELINE TRAINING PROTMIND SELESAI")
    log.info(f"  Model siap deploy: {CONFIG['output_model']}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()