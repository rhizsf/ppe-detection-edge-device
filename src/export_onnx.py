"""
=============================================================================
PROTMIND — export_onnx.py
Utility script to export YOLOv8 PyTorch (.pt) weights to ONNX (.onnx) format.
=============================================================================
"""

import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 PyTorch model to ONNX format")
    parser.add_argument(
        "--weights", 
        type=str, 
        default="models/best_ppe.pt", 
        help="Path to the PyTorch weights file (.pt) (default: models/best_ppe.pt)"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=640, 
        help="Image size for the model input (default: 640)"
    )
    parser.add_argument(
        "--half", 
        action="store_true", 
        help="Export with half-precision (FP16)"
    )
    parser.add_argument(
        "--int8", 
        action="store_true", 
        help="Export with INT8 quantization"
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Weights file not found at {weights_path.resolve()}")
        sys.exit(1)

    print("=" * 60)
    print(f"Exporting model: {weights_path.name}")
    print(f"Source Path: {weights_path.resolve()}")
    print(f"Input Image Size: {args.imgsz}x{args.imgsz}")
    print(f"FP16 Precision: {args.half}")
    print(f"INT8 Quantization: {args.int8}")
    print("=" * 60)

    try:
        # Load the PyTorch YOLO model
        model = YOLO(str(weights_path))
        
        # Export the model
        # format='onnx' creates a .onnx file in the same directory as the weights
        output_path = model.export(
            format="onnx",
            imgsz=args.imgsz,
            half=args.half,
            int8=args.int8,
            dynamic=False,
            simplify=True  # Simplifies the ONNX graph for faster inference on edge devices
        )
        
        print("-" * 60)
        print("✓ Model successfully exported!")
        print(f"ONNX Model saved at: {output_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during export: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
