#!/usr/bin/env python3
"""DEIMv2 training + evaluation experiment.

Trains for 20 epochs, evaluates with COCOeval + sklearn,
prints a rich summary table, saves results to:
  prompt_data/deimv2_results.json
  prompt_data/deimv2_history.json

If OOM / crash / NaN loss occurs: prints full traceback and exits.
Restart from epoch 1 by deleting the checkpoint dir and re-running.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME  = "deimv2"
CONFIG_PATH = "configs/deimv2.yaml"
TAG         = "exp_DEIMv2_1.0"
DEVICE      = "cuda"
EPOCHS      = 20
BATCH_SIZE  = 2
NUM_WORKERS = 4
OUTPUT_DIR  = Path("prompt_data")

# ── Verify config exists immediately ──────────────────────────────────────────
if not Path(CONFIG_PATH).exists():
    print(f"ERROR: config not found: {CONFIG_PATH}")
    sys.exit(1)


# ── History callback ──────────────────────────────────────────────────────────

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class HistoryCallback(Callback):
    """Records per-epoch train/val losses."""

    def __init__(self):
        self.history: list[dict] = []
        self._train_loss = float("nan")

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        self._train_loss = float(m.get("train/loss_total", float("nan")))

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        row = {
            "epoch":          trainer.current_epoch,
            "train_loss":     self._train_loss,
            "val_loss":       float(m.get("val/loss_total",  float("nan"))),
            "val_loss_ce":    float(m.get("val/loss_ce",     float("nan"))),
            "val_loss_bbox":  float(m.get("val/loss_bbox",   float("nan"))),
            "val_loss_giou":  float(m.get("val/loss_giou",   float("nan"))),
        }
        self.history.append(row)
        self._train_loss = float("nan")

        # Detect NaN and raise so the trainer crashes cleanly
        if np.isnan(row["val_loss"]):
            raise RuntimeError(
                f"NaN val_loss detected at epoch {trainer.current_epoch}. "
                "Fix the root cause (lr too large, bad data, etc.) and restart from epoch 1."
            )


# ── Config builder ────────────────────────────────────────────────────────────

def build_cfg():
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(CONFIG_PATH)
    overrides = OmegaConf.create({
        "training": {
            "max_epochs":     EPOCHS,
            "checkpoint_dir": f"models/checkpoints/{MODEL_NAME}/{TAG}",
            "log_dir":        f"experiments/{MODEL_NAME}/{TAG}",
        },
        "data": {
            "batch_size":  BATCH_SIZE,
            "num_workers": NUM_WORKERS,
        },
        "inference": {
            "device": DEVICE,
        },
    })
    return OmegaConf.merge(cfg, overrides)


# ── Training ──────────────────────────────────────────────────────────────────

def do_train(cfg) -> tuple[str, list]:
    """Run training; return (best_checkpoint_path, history)."""
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger
    from training.trainer import DetectionLightningModule
    import run_config

    ckpt_dir = cfg.training.checkpoint_dir
    log_dir  = cfg.training.log_dir
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    module = DetectionLightningModule(cfg)

    history_cb = HistoryCallback()
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        monitor="val/loss_total",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = CSVLogger(
        save_dir=str(Path(log_dir).parent),
        name=Path(log_dir).name,
    )

    accelerator = "gpu" if DEVICE == "cuda" else "cpu"

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        gradient_clip_val=run_config.TUNING.get("grad_clip_max_norm", 1.0),
        log_every_n_steps=cfg.training.log_every_n_steps,
        default_root_dir=ckpt_dir,
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor, history_cb],
        accelerator=accelerator,
        devices=1,
    )

    trainer.fit(module)

    best = checkpoint_cb.best_model_path or str(Path(ckpt_dir) / "last.ckpt")
    return best, history_cb.history


# ── IoU helper ────────────────────────────────────────────────────────────────

def _box_iou_xywh(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between [N,4] and [M,4] boxes in absolute (x,y,w,h) format.
    Returns [N, M] IoU matrix.
    """
    x1a, y1a = boxes1[:, 0], boxes1[:, 1]
    x2a = x1a + boxes1[:, 2]
    y2a = y1a + boxes1[:, 3]

    x1b, y1b = boxes2[:, 0], boxes2[:, 1]
    x2b = x1b + boxes2[:, 2]
    y2b = y1b + boxes2[:, 3]

    ix1 = np.maximum(x1a[:, None], x1b[None, :])
    iy1 = np.maximum(y1a[:, None], y1b[None, :])
    ix2 = np.minimum(x2a[:, None], x2b[None, :])
    iy2 = np.minimum(y2a[:, None], y2b[None, :])

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    area1 = (x2a - x1a) * (y2a - y1a)
    area2 = (x2b - x1b) * (y2b - y1b)
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + 1e-6)


# ── Evaluation ────────────────────────────────────────────────────────────────

def do_evaluate(cfg, checkpoint_path: str, history: list) -> dict:
    """Run full evaluation; return metrics dict."""
    import torch
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from sklearn.metrics import accuracy_score
    from torch.utils.data import DataLoader, Subset
    from training.dataset import COCODetectionDataset, collate_fn
    from training.transforms import build_val_transforms
    from inference.postprocess import apply_nms, threshold_filter
    from training.trainer import DetectionLightningModule
    from tqdm import tqdm
    import run_config

    ann_file = cfg.data.ann_file
    img_dir  = cfg.data.img_dir
    max_size = int(cfg.data.max_size)
    seed     = int(cfg.data.seed)

    # ── Build val subset (same split as training) ─────────────────────────────
    full_dataset = COCODetectionDataset(
        ann_file=ann_file,
        img_dir=img_dir,
        transforms=build_val_transforms(max_size),
    )
    n = len(full_dataset)

    ratios  = run_config.DATA["split_ratios"]
    n_train = int(n * ratios["train"])
    n_val   = int(n * ratios["val"])

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()

    val_indices = perm[n_train : n_train + n_val]
    if not run_config.DATA.get("use_test_split", False):
        val_indices = val_indices + perm[n_train + n_val :]

    val_dataset  = Subset(full_dataset, val_indices)
    val_img_ids  = [full_dataset.img_ids[i] for i in val_indices]

    loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # ── Load model from checkpoint ────────────────────────────────────────────
    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt_state = torch.load(checkpoint_path, map_location=DEVICE)
    module = DetectionLightningModule(cfg)
    # Extract model weights from Lightning checkpoint
    model_state = {
        k.removeprefix("model."): v
        for k, v in ckpt_state["state_dict"].items()
        if k.startswith("model.")
    }
    module.model.load_state_dict(model_state, strict=False)
    model = module.model.to(DEVICE).eval()

    coco_results: list[dict] = []
    pred_labels_matched: list[int] = []
    gt_labels_matched:   list[int] = []

    # ── Inference loop ────────────────────────────────────────────────────────
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating val split", unit="batch"):
            images     = batch["images"].to(DEVICE)
            image_mask = batch.get("masks")
            if image_mask is not None:
                image_mask = image_mask.to(DEVICE)
            _, _, H, W = images.shape

            outputs    = model(images, image_mask)
            preds_list = threshold_filter(
                outputs["pred_logits"],
                outputs["pred_boxes"],
                threshold=cfg.inference.threshold,
                top_k=cfg.inference.top_k,
            )
            preds_list = apply_nms(preds_list, iou_threshold=cfg.inference.nms_threshold)

            for pred, img_id, target in zip(
                preds_list, batch["image_ids"], batch["targets"]
            ):
                if len(pred["boxes"]) == 0:
                    continue

                cx, cy, bw, bh = pred["boxes"].unbind(-1)
                x1    = ((cx - bw / 2) * W).cpu().numpy()
                y1    = ((cy - bh / 2) * H).cpu().numpy()
                abs_w = (bw * W).cpu().numpy()
                abs_h = (bh * H).cpu().numpy()
                scores = pred["scores"].cpu().numpy()
                labels = pred["labels"].cpu().numpy()

                pred_boxes_xywh = np.stack([x1, y1, abs_w, abs_h], axis=1)

                # COCO results
                for score, label, x, y, w, h in zip(
                    scores, labels, x1, y1, abs_w, abs_h
                ):
                    coco_results.append({
                        "image_id":   int(img_id),
                        "category_id": full_dataset.idx_to_cat_id[int(label)],
                        "bbox":  [float(x), float(y), float(w), float(h)],
                        "score": float(score),
                    })

                # Classification accuracy: match preds -> GT via IoU >= 0.5
                gt_boxes_norm = target["boxes"].cpu().numpy()   # [G, 4] norm cx,cy,w,h
                gt_labels_arr = target["labels"].cpu().numpy()

                if len(gt_labels_arr) > 0:
                    gt_x = (gt_boxes_norm[:, 0] - gt_boxes_norm[:, 2] / 2) * W
                    gt_y = (gt_boxes_norm[:, 1] - gt_boxes_norm[:, 3] / 2) * H
                    gt_w = gt_boxes_norm[:, 2] * W
                    gt_h = gt_boxes_norm[:, 3] * H
                    gt_boxes_xywh = np.stack([gt_x, gt_y, gt_w, gt_h], axis=1)

                    iou_mat = _box_iou_xywh(pred_boxes_xywh, gt_boxes_xywh)  # [P, G]
                    best_gt  = iou_mat.argmax(axis=1)
                    best_iou = iou_mat.max(axis=1)

                    for pi, (gi, iou_val) in enumerate(zip(best_gt, best_iou)):
                        if iou_val >= 0.5:
                            pred_labels_matched.append(int(labels[pi]))
                            gt_labels_matched.append(int(gt_labels_arr[gi]))

    # ── COCO Detection Metrics ────────────────────────────────────────────────
    if not coco_results:
        print("  WARNING: no predictions — all detection metrics will be 0.")
        map50 = map50_95 = precision = recall = 0.0
    else:
        coco_gt_eval = COCO(ann_file)
        coco_dt      = coco_gt_eval.loadRes(coco_results)
        evaluator    = COCOeval(coco_gt_eval, coco_dt, "bbox")
        evaluator.params.imgIds = val_img_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        map50_95 = float(evaluator.stats[0])   # mAP @ 0.50:0.95
        map50    = float(evaluator.stats[1])   # mAP @ 0.50

        # Precision @ IoU=0.50, area=all, maxDets=100  (index [0, :, :, 0, 2])
        prec     = evaluator.eval["precision"]  # [T, R, K, A, M]
        prec50   = prec[0, :, :, 0, 2]
        valid_p  = prec50[prec50 > -1]
        precision = float(valid_p.mean()) if len(valid_p) > 0 else 0.0

        # Recall  @ IoU=0.50, area=all, maxDets=100  (index [0, :, 0, 2])
        rec      = evaluator.eval["recall"]     # [T, K, A, M]
        rec50    = rec[0, :, 0, 2]
        valid_r  = rec50[rec50 > -1]
        recall   = float(valid_r.mean()) if len(valid_r) > 0 else 0.0

    # ── Classification Accuracy ───────────────────────────────────────────────
    if pred_labels_matched:
        cls_acc = float(accuracy_score(gt_labels_matched, pred_labels_matched))
    else:
        print("  WARNING: no matched detections for classification accuracy — defaulting to 0.")
        cls_acc = 0.0

    # ── Final val loss from history ───────────────────────────────────────────
    loss_final = history[-1]["val_loss"] if history else float("nan")

    return {
        "detection_mAP50":         map50,
        "detection_mAP50_95":      map50_95,
        "classification_accuracy": cls_acc,
        "combined_score":          0.7 * map50_95 + 0.3 * cls_acc,
        "precision":               precision,
        "recall":                  recall,
        "loss_final":              loss_final,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import io
    import os
    from rich.console import Console
    from rich.table import Table

    # Force UTF-8 output on Windows to avoid cp1252 issues with box-drawing chars
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    console = Console(file=utf8_stdout, highlight=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold green]Config:[/] {CONFIG_PATH}  OK")

    cfg = build_cfg()

    # ── Train ─────────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]=== Training DEIMv2 ===[/]")
    console.print(f"  epochs={EPOCHS}  batch={BATCH_SIZE}  device={DEVICE}  "
                  f"workers={NUM_WORKERS}  tag={TAG}")

    start_time = time.time()
    try:
        best_ckpt, history = do_train(cfg)
    except RuntimeError:
        console.print("\n[bold red]TRAINING FAILED (RuntimeError):[/]")
        traceback.print_exc()
        console.print(
            "\n[yellow]To restart from epoch 1, delete the checkpoint directory "
            f"models/checkpoints/{MODEL_NAME}/{TAG}/ and re-run.[/]"
        )
        sys.exit(1)
    except Exception:
        console.print("\n[bold red]TRAINING FAILED:[/]")
        traceback.print_exc()
        console.print(
            "\n[yellow]To restart from epoch 1, delete the checkpoint directory "
            f"models/checkpoints/{MODEL_NAME}/{TAG}/ and re-run.[/]"
        )
        sys.exit(1)

    training_time = (time.time() - start_time) / 60.0
    console.print(f"\n[bold green]Training complete:[/] {training_time:.1f} min")
    console.print(f"[bold green]Best checkpoint :[/] {best_ckpt}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    console.print("\n[bold cyan]=== Evaluating DEIMv2 ===[/]")
    try:
        metrics = do_evaluate(cfg, best_ckpt, history)
    except Exception:
        console.print("\n[bold red]EVALUATION FAILED:[/]")
        traceback.print_exc()
        sys.exit(1)

    # ── Assemble full results ─────────────────────────────────────────────────
    results = {
        **metrics,
        "training_time_minutes": round(training_time, 2),
        "checkpoint_path":       str(best_ckpt),
        "experiment_tag":        TAG,
        "epochs":                EPOCHS,
        "config":                CONFIG_PATH,
    }

    # ── Save JSON files ────────────────────────────────────────────────────────
    results_path = OUTPUT_DIR / "deimv2_results.json"
    history_path = OUTPUT_DIR / "deimv2_history.json"

    results_path.write_text(json.dumps(results, indent=2))
    history_path.write_text(json.dumps(history, indent=2))

    console.print(f"\n[bold green]Results saved :[/] {results_path}")
    console.print(f"[bold green]History saved :[/] {history_path}")

    # ── Rich summary table ────────────────────────────────────────────────────
    table = Table(
        title=f"DEIMv2 Results - {TAG}",
        show_lines=True,
        header_style="bold magenta",
    )
    table.add_column("Metric",  style="cyan",      no_wrap=True)
    table.add_column("Value",   style="bold white", justify="right")

    rows = [
        ("detection_mAP50",         f"{results['detection_mAP50']:.4f}"),
        ("detection_mAP50_95",      f"{results['detection_mAP50_95']:.4f}"),
        ("classification_accuracy", f"{results['classification_accuracy']:.4f}"),
        ("combined_score",          f"{results['combined_score']:.4f}"),
        ("precision",               f"{results['precision']:.4f}"),
        ("recall",                  f"{results['recall']:.4f}"),
        ("loss_final",              f"{results['loss_final']:.4f}"),
        ("training_time_minutes",   f"{results['training_time_minutes']:.1f}"),
        ("checkpoint_path",         str(best_ckpt)),
    ]
    for metric, value in rows:
        table.add_row(metric, value)

    console.print(table)
    console.file.flush()


if __name__ == "__main__":
    main()
