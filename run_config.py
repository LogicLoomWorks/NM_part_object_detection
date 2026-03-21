
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
