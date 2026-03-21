
# ╔══════════════════════════════════════════════════════════╗
# ║       RETAIL SHELF DETECTION — CONTROL PANEL            ║
# ║                   edit me to run                        ║
# ╚══════════════════════════════════════════════════════════╝
#
# HOW TO USE:
#   1. Set enabled = True / False for each model below
#   2. Set RUN modes  (train / evaluate / test)
#   3. Adjust TRAINING settings if needed
#   4. Run:  python run.py


# ┌─────────────────────────────────────────────────────────┐
# │  MODEL SELECTION                                        │
# └─────────────────────────────────────────────────────────┘

MODELS = {

    # ── GroundingDINO ─────────────────────────────────────────────────────────
    # DETR-style detector with a ResNet-50 backbone + FPN + Transformer encoder/decoder
    "groundingdino": {
        "enabled": False,                        # ← True = include in this run
        "config":  "configs/default.yaml",      # ← change only if you moved the config file
        "weights": None,                        # ← pretrained init weights  (None = train from scratch)
        "resume":  None,                        # ← resume interrupted training (None = start fresh)
    },

    # ── DEIMv2 ────────────────────────────────────────────────────────────────
    # DINO-style detector with deformable attention + contrastive denoising (CDN)
    "deimv2": {
        "enabled": True,                       # ← True = include in this run
        "config":  "configs/deimv2.yaml",       # ← change only if you moved the config file
        "weights": None,                        # ← pretrained init weights  (None = train from scratch)
        "resume":  None,                        # ← resume interrupted training (None = start fresh)
    },

    # ── SAM ───────────────────────────────────────────────────────────────────
    # SAM ViT-B image encoder repurposed as a prompt-free detector
    "sam": {
        "enabled": True,                       # ← True = include in this run
        "config":  "configs/sam.yaml",          # ← change only if you moved the config file
        "weights": None,                        # ← pretrained init weights  (None = train from scratch)
        "resume":  None,                        # ← resume interrupted training (None = start fresh)
    },

    # ── YOLOv8x ───────────────────────────────────────────────────────────────
    # Ultralytics YOLOv8 extra-large variant
    "yolov8x": {
        "enabled": False,                       # ← True = include in this run
        "config":  "configs/yolov8x.yaml",      # ← change only if you moved the config file
        "weights": None,                        # ← pretrained init weights  (None = train from scratch)
        "resume":  None,                        # ← resume interrupted training (None = start fresh)
    },

    # ── RT-DETR ───────────────────────────────────────────────────────────────
    # Real-Time Detection Transformer
    "rtdetr": {
        "enabled": False,                       # ← True = include in this run
        "config":  "configs/rtdetr.yaml",       # ← change only if you moved the config file
        "weights": None,                        # ← pretrained init weights  (None = train from scratch)
        "resume":  None,                        # ← resume interrupted training (None = start fresh)
    },

}


# ┌─────────────────────────────────────────────────────────┐
# │  RUN MODES                                              │
# └─────────────────────────────────────────────────────────┘

RUN = {
    "train":    True,   # fit the model on the training split
                        # requires: data configured in the model's config yaml

    "evaluate": True,   # run COCOeval on the validation split after training
                        # requires: a trained checkpoint to exist

    "test":     False,  # run inference on the full dataset (no labels) and write
                        # predictions.json in COCO results format
                        # requires: a trained checkpoint to exist
}


# ┌─────────────────────────────────────────────────────────┐
# │  TRAINING SETTINGS                                      │
# └─────────────────────────────────────────────────────────┘
# These values override the matching fields in each model's config yaml.

TRAINING = {
    "epochs":         20,        # total training epochs          [range: 1 – ∞]
    "batch_size":     2,         # images per GPU step            [range: 1 – GPU VRAM limit]
    "device":         "cuda",    # compute device                 [options: "cuda" | "cpu"]
    "num_workers":    4,         # DataLoader worker processes    [range: 0 – cpu count]
    "experiment_tag": "exp_DEIMv2_1.0", # tag appended to output dirs    [any string, no spaces]
}


# ┌─────────────────────────────────────────────────────────┐
# │  DATA                                                   │
# └─────────────────────────────────────────────────────────┘
# Defines where training data lives and how it is split.
# Multiple source directories may be listed; all will be merged before splitting.

DATA = {
    # List of directories containing COCO-format datasets to load.
    # Each entry is a dict with "ann_file" and "img_dir" keys, or a plain path
    # string pointing to a directory that contains annotations.json + images/.
    "sources": [
        "data/raw/coco_dataset/train",          # primary annotated dataset
    ],

    # Fraction of the merged dataset assigned to each split.
    # Values must sum to 1.0.
    "split_ratios": {
        "train": 0.70,
        "val":   0.20,
        "test":  0.10,
    },

    # When False the test fraction is merged into the validation split
    # (useful when the dataset is too small to hold out a separate test set).
    "use_test_split": False,

    # Directory where augmented images and annotations will be saved
    # when AUGMENTATION["save_augmented"] is True.
    "augmented_data_dir": "data/augmented",
}


# ┌─────────────────────────────────────────────────────────┐
# │  AUGMENTATION                                           │
# └─────────────────────────────────────────────────────────┘
# Toggle individual Albumentations transforms that will be wired in Prompt 3.
# Set "save_augmented": True to persist augmented samples to DATA["augmented_data_dir"].

AUGMENTATION = {
    "save_augmented": False,    # write augmented images + annotations to disk

    # ── Geometry ──────────────────────────────────────────────────────────────
    "horizontal_flip": {
        "enabled": True,
        "p": 0.5,               # probability of applying the flip
    },

    "shift_scale_rotate": {
        "enabled": True,
        "shift_limit":  0.05,   # max absolute fraction of image size for shift
        "scale_limit":  0.10,   # max relative scale change  (±10 %)
        "rotate_limit": 10,     # max rotation in degrees    (±10°)
    },

    # ── Colour / brightness ───────────────────────────────────────────────────
    "random_brightness_contrast": {
        "enabled": True,
    },

    "hue_saturation_value": {
        "enabled": True,
    },

    "random_gamma": {
        "enabled": False,
    },

    "clahe": {
        "enabled": False,       # Contrast Limited Adaptive Histogram Equalisation
    },

    # ── Blur / noise ──────────────────────────────────────────────────────────
    "blur_one_of": {
        "enabled": True,
        "compression_quality_lower": 60,    # lower bound for JPEG compression quality
                                            # used by ImageCompression in the blur group
    },

    "gauss_noise": {
        "enabled": True,
    },

    "iso_noise": {
        "enabled": False,
    },

    # ── Dropout ───────────────────────────────────────────────────────────────
    "coarse_dropout": {
        "enabled": True,
        "max_holes":  8,        # maximum number of rectangular cutout regions
        "max_height": 32,       # maximum height of each cutout region (pixels)
        "max_width":  32,       # maximum width  of each cutout region (pixels)
    },
}


# ┌─────────────────────────────────────────────────────────┐
# │  COMPONENTS                                             │
# └─────────────────────────────────────────────────────────┘
# Per-component enable flags discovered during the codebase audit.
# These will be wired into the trainer once the trainer supports per-component
# toggling.  For now they serve as documentation and future hook points.

COMPONENTS = {
    # ── Backbone ──────────────────────────────────────────────────────────────
    "backbone": True,   # TimmBackbone (ResNet-50 / ResNet-101 / ConvNeXt-B)
                        # wired when trainer supports it

    # ── Neck ──────────────────────────────────────────────────────────────────
    "neck": True,       # FPN (Feature Pyramid Network)
                        # wired when trainer supports it
}


# ┌─────────────────────────────────────────────────────────┐
# │  TUNING                                                 │
# └─────────────────────────────────────────────────────────┘
# Hyperparameters that override their equivalents in each model's YAML config.

TUNING = {
    "optimizer":          "adamw",  # options: "sgd" | "adam" | "adamw"
    "learning_rate":      1e-4,
    "weight_decay":       1e-4,
    "scheduler":          "cosine", # options: "cosine" | "step" | "none"
    "warmup_epochs":      3,
    "grad_clip_max_norm": 1.0,
    "freeze_backbone":    False,    # freeze all backbone weights during training
    "frozen_layers":      0,        # number of backbone stages to freeze (0 = none)
}
