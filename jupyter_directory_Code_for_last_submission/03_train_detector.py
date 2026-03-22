"""
03_train_detector.py — Train YOLOv8 detector on COCO + augmented product images.

Trains YOLOv8 to:
  1. Detect product regions (from aug_product_images weak supervision)
  2. Classify specific products (from COCO detailed annotations)

Dataset: data/yolo_detection/ (created by 02_prepare_detection_data.py)
  - Train: 5199 images (199 COCO + 5000 aug)
  - Val:   1967 images (49 COCO + 1918 aug)
  - Classes: 356 product categories

Output: runs/detect/train*/
  - weights/best.pt (best validation mAP)
  - weights/last.pt (final epoch)
  - predictions on validation set

Usage:
    python 03_train_detector.py [options]

Options:
    --model MODEL       YOLOv8 model size: nano, small, medium, large, xlarge (default: large)
    --epochs EPOCHS     Training epochs (default: 150)
    --batch BATCH       Batch size (default: 32)
    --imgsz IMGSZ       Image size (default: 640)
    --device DEVICE     GPU device: 0, 1, 2 or cpu (default: 0)
    --patience PATIENCE Early stopping patience (default: 50)
    --resume            Resume from last.pt if exists
"""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO
import torch

# Fix for PyTorch 2.6+ where weights_only=True is the default,
# which breaks older ultralytics versions loading YOLOv8 .pt files.
import torch.serialization
torch.serialization.add_safe_globals([])  # no-op to ensure module is loaded
# Monkey-patch torch.load to default weights_only=False for ultralytics compat
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_YAML = Path("data/yolo_detection/dataset.yaml")
RUNS_DIR     = Path("runs/detect")
SEED         = 42

# Model sizes and recommended settings
MODEL_CONFIGS = {
    "nano":    {"weights": "yolov8n.pt",  "size": 416},
    "small":   {"weights": "yolov8s.pt",  "size": 416},
    "medium":  {"weights": "yolov8m.pt",  "size": 512},
    "large":   {"weights": "yolov8l.pt",  "size": 640},
    "xlarge":  {"weights": "yolov8x.pt",  "size": 640},
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 detector on COCO + augmented product images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 03_train_detector.py                    # Train with defaults (large, 150 epochs)
  python 03_train_detector.py --model xlarge     # Train YOLOv8x
  python 03_train_detector.py --epochs 300       # Train for 300 epochs
  python 03_train_detector.py --batch 16         # Smaller batch size
  python 03_train_detector.py --resume           # Resume from last.pt
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="large",
        choices=list(MODEL_CONFIGS.keys()),
        help="YOLOv8 model size (default: large)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Number of training epochs (default: 150)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Image size (default: model-specific). Use for fine-tuning."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device ID: 0, 1, 2 or 'cpu' (default: 0)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience in epochs (default: 50)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last.pt if it exists"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed training logs"
    )
    
    return parser.parse_args()


def check_prerequisites():
    """Verify dataset and weights exist."""
    print("=" * 60)
    print("  CHECKING PREREQUISITES")
    print("=" * 60)
    
    # Check dataset
    if not DATASET_YAML.exists():
        print(f"  ERROR: {DATASET_YAML} not found")
        print(f"     Run: python 02_prepare_detection_data.py")
        raise FileNotFoundError(f"Dataset not found: {DATASET_YAML}")
    
    print(f"  Dataset: {DATASET_YAML}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu_name} (device count: {gpu_count})")
        print(f"  VRAM: {vram_gb:.1f} GB")
    else:
        print(f"  GPU not available, will use CPU (SLOW)")
    
    print()


def main():
    """Main training function."""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("  YOLOv8 DETECTOR TRAINING")
    print("=" * 60)
    
    # Check prerequisites
    check_prerequisites()
    
    # Get model config
    model_size = args.model
    config = MODEL_CONFIGS[model_size]
    weights = config["weights"]
    default_imgsz = config["size"]
    imgsz = args.imgsz if args.imgsz else default_imgsz
    
    print("  CONFIGURATION")
    print("  " + "-" * 56)
    print(f"    Model       : YOLOv8{model_size.upper()} (weights: {weights})")
    print(f"    Epochs      : {args.epochs}")
    print(f"    Batch size  : {args.batch}")
    print(f"    Image size  : {imgsz}x{imgsz}")
    print(f"    Device      : {'GPU ' + args.device if args.device != 'cpu' else 'CPU'}")
    print(f"    Patience    : {args.patience} epochs (early stopping)")
    print(f"    Dataset     : {DATASET_YAML}")
    print(f"    Output dir  : {RUNS_DIR}")
    
    if args.resume:
        print(f"    Resume      : YES (from last.pt)")
    
    print("=" * 60)
    print()
    
    # Load model
    print("  Loading model...")
    if args.resume:
        last_pt = RUNS_DIR / "train" / "weights" / "last.pt"
        if last_pt.exists():
            print(f"    Resuming from: {last_pt}")
            model = YOLO(str(last_pt))
        else:
            print(f"    last.pt not found, starting fresh with {weights}")
            model = YOLO(weights)
    else:
        model = YOLO(weights)
    
    print(f"    Model loaded")
    print()
    
    # Training hyperparameters
    # Tuned for product detection with 356 classes
    train_args = {
        "data": str(DATASET_YAML),
        "epochs": args.epochs,
        "imgsz": imgsz,
        "batch": args.batch,
        "device": args.device,
        "patience": args.patience,
        "save": True,
        "save_period": -1,  # Save only best
        "exist_ok": True,
        "verbose": args.verbose,
        "seed": SEED,
        
        # Augmentation strategy for product detection
        "hsv_h": 0.015,     # HSV hue augmentation (small for products)
        "hsv_s": 0.7,       # HSV saturation
        "hsv_v": 0.4,       # HSV value
        "degrees": 10,      # Rotation (small, products are axis-aligned)
        "translate": 0.1,   # Translation
        "scale": 0.5,       # Scale variation
        "flipud": 0.5,      # Vertical flip
        "fliplr": 0.5,      # Horizontal flip
        "mosaic": 1.0,      # Mosaic augmentation
        "mixup": 0.1,       # Mixup augmentation
        "copy_paste": 0.0,  # Copy-paste disabled (products shouldn't overlap)
        
        # Optimization
        "optimizer": "SGD",  # SGD works well for detection
        "lr0": 0.01,        # Initial learning rate
        "lrf": 0.01,        # Final learning rate
        "momentum": 0.937,  # Momentum
        "weight_decay": 0.0005,
        
        # Validation & logging
        "val": True,
        "plots": True,
        "half": True,       # Use FP16 (faster, requires GPU)
        "rect": False,      # Rectangle training (slower but handles different aspect ratios)
        "cos_lr": False,    # Cosine learning rate
        "label_smoothing": 0.0,  # No label smoothing (help with many classes)
        
        # Callbacks & output
        "project": str(RUNS_DIR),
        "name": "train",
        "workers": 8,
    }
    
    print("  TRAINING HYPERPARAMETERS")
    print("  " + "-" * 56)
    print(f"    Learning rate (initial) : {train_args['lr0']}")
    print(f"    Optimizer               : {train_args['optimizer']}")
    print(f"    Augmentation            : HSV, Rotation, Scale, Flip, Mosaic")
    print(f"    Mixed precision (FP16)  : {train_args['half']}")
    print(f"    Workers                 : {train_args['workers']}")
    print("=" * 60)
    print()
    
    # Train
    print("  STARTING TRAINING...")
    print()
    try:
        results = model.train(**train_args)
        
        print()
        print("=" * 60)
        print("  TRAINING COMPLETE")
        print("=" * 60)
        
        # Print results summary
        if results:
            print(f"  Best model saved to: {RUNS_DIR}/train/weights/best.pt")
            print(f"  Last model saved to: {RUNS_DIR}/train/weights/last.pt")
            print(f"  Results saved to: {RUNS_DIR}/train/")
            print()
            print("  Key metrics:")
            if hasattr(results, 'results_dict'):
                metrics = results.results_dict
                if 'metrics/mAP50' in metrics:
                    print(f"    mAP@50   : {metrics['metrics/mAP50']:.4f}")
                if 'metrics/mAP50-95' in metrics:
                    print(f"    mAP@50-95: {metrics['metrics/mAP50-95']:.4f}")
        
        print()
        print("  NEXT STEPS:")
        print("    1. Evaluate on test set")
        print("       python 04_prepare_classification_data.py")
        print("    2. Build submission")
        print("       python 06_build_submission.py")
        print()
        
    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("  TRAINING INTERRUPTED BY USER")
        print("=" * 60)
        print(f"  Checkpoint saved: {RUNS_DIR}/train/weights/last.pt")
        print("  Resume training with: python 03_train_detector.py --resume")
        print()
    except Exception as e:
        print()
        print("=" * 60)
        print("  TRAINING FAILED")
        print("=" * 60)
        print(f"  Error: {e}")
        print()
        raise


if __name__ == "__main__":
    main()