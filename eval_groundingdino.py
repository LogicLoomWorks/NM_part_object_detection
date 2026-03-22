#!/usr/bin/env python3
"""Evaluation-only script for the best GroundingDINO sweep checkpoint.

Best sweep run: sweep__groundingdino__aug_none__backbone=0__neck=1
  val/loss_total = 2814.90  (lowest of the two sweep runs, 1 epoch)
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

MODEL_NAME  = "groundingdino"
TAG         = "sweep__groundingdino__aug_none__backbone=0__neck=1"
DEVICE      = "cuda"
BATCH_SIZE  = 2
NUM_WORKERS = 4
CONFIG_PATH = "configs/default.yaml"
OUTPUT_DIR  = Path("prompt_data")
# After 1 epoch the model confidence is very low; use a low threshold
EVAL_THRESHOLD = 0.05


def load_history() -> list[dict]:
    csv_files = sorted(glob.glob(
        f"experiments/{MODEL_NAME}/{TAG}/version_*/metrics.csv"
    ))
    if not csv_files:
        return []
    epoch_data: dict[int, dict] = {}
    with open(csv_files[-1]) as f:
        for row in csv.DictReader(f):
            ep = row.get("epoch", "")
            if not ep or not ep.strip():
                continue
            try:
                ep = int(float(ep))
            except ValueError:
                continue
            if ep not in epoch_data:
                epoch_data[ep] = {"epoch": ep}
            for k, v in row.items():
                if v and v.strip():
                    try:
                        epoch_data[ep][k] = float(v)
                    except ValueError:
                        pass
    history = []
    for ep in sorted(epoch_data):
        r = epoch_data[ep]
        history.append({
            "epoch":         int(r["epoch"]),
            "train_loss":    r.get("train/loss_total", float("nan")),
            "val_loss":      r.get("val/loss_total",   float("nan")),
            "val_loss_ce":   r.get("val/loss_ce",      float("nan")),
            "val_loss_bbox": r.get("val/loss_bbox",    float("nan")),
            "val_loss_giou": r.get("val/loss_giou",    float("nan")),
        })
    return history


def box_iou_xywh(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    b1x1, b1y1 = boxes1[:, 0], boxes1[:, 1]
    b1x2 = b1x1 + boxes1[:, 2]
    b1y2 = b1y1 + boxes1[:, 3]
    b2x1, b2y1 = boxes2[:, 0], boxes2[:, 1]
    b2x2 = b2x1 + boxes2[:, 2]
    b2y2 = b2y1 + boxes2[:, 3]
    ix1 = np.maximum(b1x1[:, None], b2x1[None, :])
    iy1 = np.maximum(b1y1[:, None], b2y1[None, :])
    ix2 = np.minimum(b1x2[:, None], b2x2[None, :])
    iy2 = np.minimum(b1y2[:, None], b2y2[None, :])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    a1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    a2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    union = a1[:, None] + a2[None, :] - inter
    return inter / (union + 1e-6)


def main() -> None:
    from omegaconf import OmegaConf
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from sklearn.metrics import accuracy_score
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    from inference.postprocess import apply_nms, threshold_filter
    from models.groundingdino.model import load_checkpoint
    from training.dataset import COCODetectionDataset, collate_fn
    from training.transforms import build_val_transforms
    import run_config

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg = OmegaConf.merge(cfg, OmegaConf.create({
        "training": {
            "checkpoint_dir": f"models/checkpoints/{MODEL_NAME}/{TAG}",
            "log_dir":        f"experiments/{MODEL_NAME}/{TAG}",
        },
        "data":      {"batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS},
        "inference": {"device": DEVICE, "threshold": EVAL_THRESHOLD},
    }))

    # ── History ───────────────────────────────────────────────────────────────
    history = load_history()
    print(f"History: {len(history)} epochs")

    # ── Val split ──────────────────────────────────────────────────────────────
    full_dataset = COCODetectionDataset(
        ann_file=cfg.data.ann_file,
        img_dir=cfg.data.img_dir,
        transforms=build_val_transforms(int(cfg.data.max_size)),
    )
    n = len(full_dataset)
    ratios  = run_config.DATA["split_ratios"]
    n_train = int(n * ratios["train"])
    n_val   = int(n * ratios["val"])
    generator = torch.Generator().manual_seed(int(cfg.data.seed))
    perm = torch.randperm(n, generator=generator).tolist()
    val_indices = perm[n_train : n_train + n_val]
    if not run_config.DATA.get("use_test_split", False):
        val_indices = val_indices + perm[n_train + n_val :]
    val_dataset = Subset(full_dataset, val_indices)
    val_img_ids = [full_dataset.img_ids[i] for i in val_indices]
    print(f"Val set: {len(val_indices)} images")

    loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt_path = f"models/checkpoints/{MODEL_NAME}/{TAG}/best.ckpt"
    print(f"Loading checkpoint: {ckpt_path}")
    model, _ = load_checkpoint(ckpt_path, cfg, device=DEVICE)
    model.eval()

    # ── Inference ─────────────────────────────────────────────────────────────
    coco_results: list[dict] = []
    pred_labels_matched: list[int] = []
    gt_labels_matched:   list[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating val split"):
            images     = batch["images"].to(DEVICE)
            image_mask = batch.get("masks")
            if image_mask is not None:
                image_mask = image_mask.to(DEVICE)
            _, _, H, W = images.shape

            outputs    = model(images, image_mask)
            preds_list = threshold_filter(
                outputs["pred_logits"], outputs["pred_boxes"],
                threshold=cfg.inference.threshold,
                top_k=cfg.inference.top_k,
            )
            preds_list = apply_nms(preds_list, iou_threshold=cfg.inference.nms_threshold)

            for pred, img_id, target in zip(preds_list, batch["image_ids"], batch["targets"]):
                if len(pred["boxes"]) == 0:
                    continue
                cx, cy, bw, bh = pred["boxes"].unbind(-1)
                x1    = ((cx - bw / 2) * W).cpu().numpy()
                y1    = ((cy - bh / 2) * H).cpu().numpy()
                abs_w = (bw * W).cpu().numpy()
                abs_h = (bh * H).cpu().numpy()
                scores = pred["scores"].cpu().numpy()
                labels = pred["labels"].cpu().numpy()
                pred_xywh = np.stack([x1, y1, abs_w, abs_h], axis=1)

                for score, label, x, y, w, h in zip(scores, labels, x1, y1, abs_w, abs_h):
                    coco_results.append({
                        "image_id":    int(img_id),
                        "category_id": full_dataset.idx_to_cat_id[int(label)],
                        "bbox":  [float(x), float(y), float(w), float(h)],
                        "score": float(score),
                    })

                gt_norm  = target["boxes"].cpu().numpy()
                gt_lbls  = target["labels"].cpu().numpy()
                if len(gt_lbls) > 0:
                    gt_x = (gt_norm[:, 0] - gt_norm[:, 2] / 2) * W
                    gt_y = (gt_norm[:, 1] - gt_norm[:, 3] / 2) * H
                    gt_w = gt_norm[:, 2] * W
                    gt_h = gt_norm[:, 3] * H
                    gt_xywh  = np.stack([gt_x, gt_y, gt_w, gt_h], axis=1)
                    iou_mat  = box_iou_xywh(pred_xywh, gt_xywh)
                    best_gt  = iou_mat.argmax(axis=1)
                    best_iou = iou_mat.max(axis=1)
                    for pi, (gi, iov) in enumerate(zip(best_gt, best_iou)):
                        if iov >= 0.5:
                            pred_labels_matched.append(int(labels[pi]))
                            gt_labels_matched.append(int(gt_lbls[gi]))

    print(f"Total predictions: {len(coco_results)}")
    print(f"Matched for cls_acc: {len(pred_labels_matched)}")

    # ── COCO metrics ───────────────────────────────────────────────────────────
    if coco_results:
        coco_gt_eval = COCO(cfg.data.ann_file)
        coco_dt      = coco_gt_eval.loadRes(coco_results)
        evaluator    = COCOeval(coco_gt_eval, coco_dt, "bbox")
        evaluator.params.imgIds = val_img_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        map50_95 = float(evaluator.stats[0])
        map50    = float(evaluator.stats[1])
        prec  = evaluator.eval["precision"]
        prec50 = prec[0, :, :, 0, 2]
        valid_p = prec50[prec50 > -1]
        precision = float(valid_p.mean()) if len(valid_p) > 0 else 0.0
        rec   = evaluator.eval["recall"]
        rec50 = rec[0, :, 0, 2]
        valid_r = rec50[rec50 > -1]
        recall = float(valid_r.mean()) if len(valid_r) > 0 else 0.0
    else:
        map50 = map50_95 = precision = recall = 0.0
        print("WARNING: no predictions above threshold=0.05")

    cls_acc    = float(accuracy_score(gt_labels_matched, pred_labels_matched)) if pred_labels_matched else 0.0
    loss_final = history[-1]["val_loss"] if history else float("nan")

    results = {
        "detection_mAP50":         map50,
        "detection_mAP50_95":      map50_95,
        "classification_accuracy": cls_acc,
        "combined_score":          0.7 * map50_95 + 0.3 * cls_acc,
        "precision":               precision,
        "recall":                  recall,
        "loss_final":              loss_final,
        "checkpoint_path":         str(Path(ckpt_path).resolve()),
        "experiment_tag":          TAG,
        "epochs":                  1,
        "config":                  CONFIG_PATH,
        "eval_threshold_used":     EVAL_THRESHOLD,
    }

    # ── Print rich table ────────────────────────────────────────────────────────
    import io
    from rich.console import Console
    from rich.table import Table
    utf8_out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    console = Console(file=utf8_out, highlight=False)

    table = Table(title=f"GroundingDINO Results - {TAG}", show_lines=True, header_style="bold magenta")
    table.add_column("Metric",  style="cyan", no_wrap=True)
    table.add_column("Value",   style="bold white", justify="right")
    for k, v in results.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
        else:
            table.add_row(k, str(v))
    console.print(table)
    console.file.flush()

    # ── Save ───────────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "groundingdino_results.json").write_text(json.dumps(results, indent=2))
    print(f"Saved: {OUTPUT_DIR / 'groundingdino_results.json'}")


if __name__ == "__main__":
    main()
