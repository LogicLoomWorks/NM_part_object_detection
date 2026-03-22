"""
05_train_classifier.py — Train EfficientNet-B3 classifier on cropped product images.

FIX: Patches torch.load for PyTorch 2.6+ compatibility.

UPGRADED for H100 NVL:
  - Model: EfficientNet-B3 at 300×300 — ONNX ~48 MB, safe for 420 MB limit
  - Batch size: 512 train / 1024 val
  - 40 epochs with cosine annealing + 5-epoch warmup
  - Stronger augmentation: RandAugment, CutMix, MixUp
  - Gradient accumulation for effective batch size of 1024
  - Class-weighted sampling for imbalanced dataset
  - Exponential Moving Average (EMA) for better generalization

Uses timm 0.9.12, exports to ONNX opset 17.

Outputs:
    submission/classifier.onnx
    submission/classifier_idx_to_category_id.json
"""

# ══════════════════════════════════════════════════════════════════════════════
# FIX: Patch torch.load BEFORE any model loading
# PyTorch 2.6 defaults weights_only=True, which breaks older checkpoint loading
# ══════════════════════════════════════════════════════════════════════════════
import torch

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
# ══════════════════════════════════════════════════════════════════════════════

import json
import math
import time
import copy
from pathlib import Path
from collections import Counter

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

import timm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data/classifier")
SUBMISSION  = Path("submission")
OUT_ONNX    = SUBMISSION / "classifier.onnx"
OUT_MAP     = SUBMISSION / "classifier_idx_to_category_id.json"

# ── Model config ──
MODEL_NAME   = "efficientnet_b3"
IMG_SIZE     = 300

# ── Batch & training — tuned for H100 NVL ──
BATCH_TRAIN  = 128
BATCH_VAL    = 256
ACCUM_STEPS  = 8
EPOCHS       = 40
LR           = 1e-3
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 5
NUM_WORKERS  = 8
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SEED         = 42

# ── EMA config ──
EMA_DECAY    = 0.9998

# ── MixUp / CutMix ──
MIXUP_ALPHA  = 0.2
CUTMIX_ALPHA = 1.0
MIXUP_PROB   = 0.5

SUBMISSION.mkdir(exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Verify dataset ────────────────────────────────────────────────────────────
if not (DATA_DIR / "train").exists():
    print(f"ERROR: {DATA_DIR / 'train'} not found.")
    print("Run: python 04_prepare_classification_data.py first.")
    raise SystemExit(1)

print("=" * 60)
print("  CLASSIFIER TRAINING — EfficientNet-B3 (H100 NVL)")
print("=" * 60)
print(f"  Device : {DEVICE}")
if DEVICE == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU    : {gpu_name}")
    print(f"  VRAM   : {vram_gb:.1f} GB")


# ── EMA helper ────────────────────────────────────────────────────────────────
class ModelEMA:
    """Exponential Moving Average of model weights for better generalization."""
    def __init__(self, model, decay=0.9998):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.ema.state_dict()


# ── MixUp / CutMix ───────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    H, W = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(W, cx + cut_w // 2)
    y2 = min(H, cy + cut_h // 2)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Data transforms ───────────────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.300, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.33)),
])

val_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

train_ds = datasets.ImageFolder(str(DATA_DIR / "train"), transform=train_tf)
val_ds   = datasets.ImageFolder(str(DATA_DIR / "val"),   transform=val_tf)
n_classes = len(train_ds.classes)

print(f"  Model  : {MODEL_NAME} @ {IMG_SIZE}x{IMG_SIZE}")
print(f"  Train  : {len(train_ds)} images, {n_classes} classes")
print(f"  Val    : {len(val_ds)} images")
print(f"  Batch  : {BATCH_TRAIN} (effective: {BATCH_TRAIN * ACCUM_STEPS} with accum)")
print(f"  Epochs : {EPOCHS}, Warmup: {WARMUP_EPOCHS}")

# ── Class-weighted sampling ───────────────────────────────────────────────────
print("\n  Building class-weighted sampler...")
class_counts = Counter(train_ds.targets)
total_samples = len(train_ds.targets)
n_cls = len(class_counts)

weights_per_class = {}
for cls_idx, count in class_counts.items():
    weights_per_class[cls_idx] = 1.0 / math.sqrt(count)

sample_weights = [weights_per_class[t] for t in train_ds.targets]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

min_count = min(class_counts.values())
max_count = max(class_counts.values())
print(f"  Class distribution: min={min_count}, max={max_count}, ratio={max_count/max(1,min_count):.0f}x")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_TRAIN, sampler=sampler,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_VAL, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=True,
)

# ── Build model ───────────────────────────────────────────────────────────────
print(f"\nBuilding {MODEL_NAME} with {n_classes} output classes ...")
model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=n_classes,
    drop_rate=0.3,
    drop_path_rate=0.2,
)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parameters: {total_params/1e6:.1f}M total, {train_params/1e6:.1f}M trainable")

ema = ModelEMA(model, decay=EMA_DECAY)
print(f"  EMA decay: {EMA_DECAY}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))

def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == "cuda"))

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\nStarting training ({EPOCHS} epochs) ...")
print(f"  Optimizer: AdamW, lr={LR}, wd={WEIGHT_DECAY}")
print(f"  Schedule : Linear warmup ({WARMUP_EPOCHS}ep) + Cosine decay")
print(f"  MixUp    : alpha={MIXUP_ALPHA}, CutMix alpha={CUTMIX_ALPHA}")
print(f"  Accum    : {ACCUM_STEPS} steps (effective batch {BATCH_TRAIN * ACCUM_STEPS})")
print()

best_val_acc  = 0.0
best_ema_acc  = 0.0
best_ckpt     = Path("runs/classifier_best.pt")
best_ema_ckpt = Path("runs/classifier_best_ema.pt")
best_ckpt.parent.mkdir(exist_ok=True)

t_start = time.time()

for epoch in range(1, EPOCHS + 1):
    # ── Train ──
    model.train()
    running_loss    = 0.0
    running_correct = 0
    running_total   = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        use_mix = np.random.random() < 0.8
        if use_mix:
            if np.random.random() < MIXUP_PROB:
                imgs, targets_a, targets_b, lam = mixup_data(imgs, labels, MIXUP_ALPHA)
            else:
                imgs, targets_a, targets_b, lam = cutmix_data(imgs, labels, CUTMIX_ALPHA)

        with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
            out = model(imgs)
            if use_mix:
                loss = mixup_criterion(criterion, out, targets_a, targets_b, lam)
            else:
                loss = criterion(out, labels)
            loss = loss / ACCUM_STEPS

        scaler.scale(loss).backward()

        if (batch_idx + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)

        running_loss += loss.item() * ACCUM_STEPS * imgs.size(0)
        preds = out.argmax(dim=1)
        if not use_mix:
            running_correct += (preds == labels).sum().item()
            running_total   += imgs.size(0)
        else:
            running_correct += (preds == targets_a).sum().item()
            running_total   += imgs.size(0)

    train_loss = running_loss / running_total if running_total else 0
    train_acc  = running_correct / running_total if running_total else 0

    scheduler.step()

    # ── Val (both model and EMA) ──
    for eval_name, eval_model in [("model", model), ("ema", ema.ema)]:
        eval_model.eval()
        val_correct  = 0
        val_total    = 0
        top5_correct = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
                    out = eval_model(imgs)
                preds = out.argmax(dim=1)
                val_correct  += (preds == labels).sum().item()
                val_total    += imgs.size(0)
                top5 = out.topk(min(5, n_classes), dim=1).indices
                top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

        val_acc  = val_correct / val_total if val_total else 0
        val_top5 = top5_correct / val_total if val_total else 0

        if eval_name == "model":
            model_val_acc = val_acc
            model_val_top5 = val_top5
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), str(best_ckpt))
        else:
            ema_val_acc = val_acc
            ema_val_top5 = val_top5
            if val_acc > best_ema_acc:
                best_ema_acc = val_acc
                torch.save(ema.state_dict(), str(best_ema_ckpt))

    elapsed = time.time() - t_start
    current_lr = optimizer.param_groups[0]['lr']
    print(
        f"  Epoch {epoch:3d}/{EPOCHS}  "
        f"loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
        f"val_top1={model_val_acc:.3f}/{ema_val_acc:.3f}(ema)  "
        f"val_top5={model_val_top5:.3f}/{ema_val_top5:.3f}(ema)  "
        f"lr={current_lr:.2e}  [{elapsed:.0f}s]"
    )

print(f"\n  Best val accuracy: model={best_val_acc:.3f}, ema={best_ema_acc:.3f}")

# ── Load best weights ─────────────────────────────────────────────────────────
if best_ema_acc >= best_val_acc:
    print(f"\n  Using EMA weights (val_acc={best_ema_acc:.3f})")
    model.load_state_dict(torch.load(str(best_ema_ckpt), map_location=DEVICE))
else:
    print(f"\n  Using model weights (val_acc={best_val_acc:.3f})")
    model.load_state_dict(torch.load(str(best_ckpt), map_location=DEVICE))
model.eval()

# ── Export to ONNX ────────────────────────────────────────────────────────────
print(f"\nExporting to ONNX (opset 17) ...")

dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

torch.onnx.export(
    model,
    dummy,
    str(OUT_ONNX),
    opset_version=17,
    input_names=["images"],
    output_names=["logits"],
    dynamic_axes=None,
    do_constant_folding=True,
)
onnx_size_mb = OUT_ONNX.stat().st_size / 1024**2
print(f"  Exported: {OUT_ONNX}  ({onnx_size_mb:.1f} MB)")

if onnx_size_mb > 400:
    print(f"  WARNING: ONNX file is {onnx_size_mb:.1f} MB, submission limit is 420 MB total!")

# ── Write classifier_idx_to_category_id.json ─────────────────────────────────
src_map = DATA_DIR / "classifier_idx_to_category_id.json"
if src_map.exists():
    raw_map = json.loads(src_map.read_text(encoding="utf-8"))
    folder_to_coco = {}
    for folder_name, if_idx in train_ds.class_to_idx.items():
        coco_cid = raw_map.get(folder_name)
        if coco_cid is not None:
            folder_to_coco[str(if_idx)] = int(coco_cid)
        else:
            folder_to_coco[str(if_idx)] = 0
    OUT_MAP.write_text(json.dumps(folder_to_coco, indent=2), encoding="utf-8")
    print(f"  Wrote {OUT_MAP}")
else:
    fallback = {str(i): i for i in range(n_classes)}
    OUT_MAP.write_text(json.dumps(fallback, indent=2), encoding="utf-8")
    print(f"  WARNING: {src_map} not found — wrote identity mapping")

total_elapsed = time.time() - t_start
print(f"\n{'='*60}")
print(f"  DONE — {total_elapsed/60:.1f} minutes total")
print(f"  Best val accuracy: model={best_val_acc:.3f}, ema={best_ema_acc:.3f}")
print(f"  ONNX size: {onnx_size_mb:.1f} MB (limit: 420 MB total for submission)")
print(f"\nNext: python 06_build_submission.py")
print(f"{'='*60}")