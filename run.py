#!/usr/bin/env python3
"""Multi-model run entrypoint.

Edit run_config.py to control which models are active, run modes, and
global training overrides.  Then simply run:

    python run.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent))

from omegaconf import OmegaConf, DictConfig

import run_config
from models import MODEL_REGISTRY


# ── Pipeline functions ────────────────────────────────────────────────────────

def train_model(model_name: str, cfg: DictConfig, spec: dict) -> Optional[str]:
    """Train a model with DetectionLightningModule + Lightning Trainer.

    Returns path to best checkpoint on success, None on failure.
    """
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger
    from training.trainer import DetectionLightningModule

    ckpt_dir = cfg.training.checkpoint_dir
    log_dir  = cfg.training.log_dir
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    module = DetectionLightningModule(cfg)

    # Load pretrained weights into model (not a full checkpoint resume)
    if spec.get("weights"):
        import torch
        weights_path = spec["weights"]
        print(f"  Loading pretrained weights: {weights_path}")
        state = torch.load(weights_path, map_location="cpu")
        if "state_dict" in state:
            state = {k.removeprefix("model."): v for k, v in state["state_dict"].items()}
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        module.model.load_state_dict(state, strict=False)

    resume_ckpt = spec.get("resume") or None

    logger = CSVLogger(
        save_dir=str(Path(log_dir).parent),
        name=Path(log_dir).name,
    )

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

    device = run_config.TRAINING["device"]
    accelerator = "gpu" if device == "cuda" else "cpu"

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        gradient_clip_val=cfg.training.grad_clip,
        log_every_n_steps=cfg.training.log_every_n_steps,
        default_root_dir=ckpt_dir,
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor],
        accelerator=accelerator,
        devices=1,
    )
    trainer.fit(module, ckpt_path=resume_ckpt)

    best = checkpoint_cb.best_model_path or str(Path(ckpt_dir) / "last.ckpt")
    print(f"  Best checkpoint: {best}")
    return best


def evaluate_model(model_name: str, cfg: DictConfig, checkpoint_path: str) -> Dict:
    """Run COCO evaluation on the validation split; return metrics dict."""
    from inference.predictor import Predictor

    print(f"  Checkpoint: {checkpoint_path}")
    predictor = Predictor(cfg, checkpoint_path, device=run_config.TRAINING["device"])
    metrics = predictor.evaluate(cfg.data.ann_file, cfg.data.img_dir)

    print(f"  mAP    : {metrics['mAP']:.4f}")
    print(f"  mAP_50 : {metrics['mAP_50']:.4f}")
    print(f"  mAP_75 : {metrics['mAP_75']:.4f}")

    # Persist results
    out_path = Path(cfg.training.log_dir) / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"  Results saved → {out_path}")
    return metrics


def test_model(model_name: str, cfg: DictConfig, checkpoint_path: str) -> None:
    """Run inference on the full dataset; write COCO-format predictions.json."""
    import torch
    from torch.utils.data import DataLoader
    from inference.predictor import Predictor
    from inference.postprocess import apply_nms, threshold_filter
    from training.dataset import COCODetectionDataset, collate_fn
    from training.transforms import build_val_transforms

    tag = run_config.TRAINING["experiment_tag"]
    pred_path = Path(f"experiments/{model_name}/{tag}/predictions.json")
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Checkpoint: {checkpoint_path}")
    device = run_config.TRAINING["device"]
    predictor = Predictor(cfg, checkpoint_path, device=device)

    dataset = COCODetectionDataset(
        ann_file=cfg.data.ann_file,
        img_dir=cfg.data.img_dir,
        transforms=build_val_transforms(cfg.data.max_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
    )

    coco_results = []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            image_mask = batch.get("masks")
            if image_mask is not None:
                image_mask = image_mask.to(device)
            _, _, H, W = images.shape

            outputs = predictor.model(images, image_mask)
            preds = threshold_filter(
                outputs["pred_logits"],
                outputs["pred_boxes"],
                threshold=cfg.inference.threshold,
                top_k=cfg.inference.top_k,
            )
            preds = apply_nms(preds, iou_threshold=cfg.inference.nms_threshold)

            for pred, img_id in zip(preds, batch["image_ids"]):
                cx, cy, bw, bh = pred["boxes"].unbind(-1)
                x1     = (cx - bw / 2) * W
                y1     = (cy - bh / 2) * H
                abs_w  = bw * W
                abs_h  = bh * H
                for score, label, x, y, w, h in zip(
                    pred["scores"].tolist(),
                    pred["labels"].tolist(),
                    x1.tolist(), y1.tolist(),
                    abs_w.tolist(), abs_h.tolist(),
                ):
                    coco_results.append({
                        "image_id":   int(img_id),
                        "category_id": dataset.idx_to_cat_id[int(label)],
                        "bbox":  [x, y, w, h],
                        "score": float(score),
                    })

    pred_path.write_text(json.dumps(coco_results, indent=2))
    print(f"  {len(coco_results)} detections → {pred_path}")


# ── Config merging ────────────────────────────────────────────────────────────

def _apply_training_overrides(cfg: DictConfig, model_name: str, tag: str) -> DictConfig:
    t = run_config.TRAINING
    overrides = OmegaConf.create({
        "training": {
            "max_epochs":     t["epochs"],
            "checkpoint_dir": f"models/checkpoints/{model_name}/{tag}",
            "log_dir":        f"experiments/{model_name}/{tag}",
        },
        "data": {
            "batch_size":  t["batch_size"],
            "num_workers": t["num_workers"],
        },
        "inference": {
            "device": t["device"],
        },
    })
    return OmegaConf.merge(cfg, overrides)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    tag       = run_config.TRAINING["experiment_tag"]
    run_train = run_config.RUN["train"]
    run_eval  = run_config.RUN["evaluate"]
    run_test  = run_config.RUN["test"]

    active_modes = [m for m, on in run_config.RUN.items() if on]

    enabled_models = {
        name: spec
        for name, spec in run_config.MODELS.items()
        if spec["enabled"]
    }

    if not enabled_models:
        print("No models enabled in run_config.MODELS — nothing to do.")
        sys.exit(0)

    # Track per-model outcomes for the summary table
    summary: list[dict] = []

    for model_name, spec in enabled_models.items():

        # ── header ───────────────────────────────────────────────────────────
        sep = "─" * 64
        print(f"\n{sep}")
        print(f"  Model  : {model_name}")
        print(f"  Tag    : {tag}")
        print(f"  Modes  : {', '.join(active_modes) or 'none'}")
        print(f"  Config : {spec['config']}")
        print(sep)

        row = {"model": model_name, "train": "—", "evaluate": "—",
               "test": "—", "mAP": None}

        # ── load + merge config ───────────────────────────────────────────────
        if not Path(spec["config"]).exists():
            print(f"  ERROR: config not found: {spec['config']} — skipping.")
            row["train"] = row["evaluate"] = row["test"] = "SKIP (no config)"
            summary.append(row)
            continue

        cfg = OmegaConf.load(spec["config"])
        cfg = _apply_training_overrides(cfg, model_name, tag)

        # ── validate registry ─────────────────────────────────────────────────
        if model_name not in MODEL_REGISTRY:
            print(f"  ERROR: '{model_name}' not in MODEL_REGISTRY — skipping.")
            row["train"] = row["evaluate"] = row["test"] = "SKIP (no registry entry)"
            summary.append(row)
            continue

        # ── train ─────────────────────────────────────────────────────────────
        best_ckpt: Optional[str] = spec.get("resume") or spec.get("weights")

        if run_train:
            print(f"\n  ── train {'─' * 48}")
            try:
                best_ckpt = train_model(model_name, cfg, spec)
                row["train"] = "OK"
            except Exception:
                print(f"  TRAIN ERROR:")
                traceback.print_exc()
                row["train"] = "FAILED"

        # ── evaluate ──────────────────────────────────────────────────────────
        if run_eval:
            print(f"\n  ── evaluate {'─' * 44}")
            ckpt = best_ckpt or str(Path(cfg.training.checkpoint_dir) / "best.ckpt")
            if not Path(ckpt).exists():
                print(f"  SKIP: checkpoint not found: {ckpt}")
                row["evaluate"] = "SKIP (no checkpoint)"
            else:
                try:
                    metrics = evaluate_model(model_name, cfg, ckpt)
                    row["evaluate"] = "OK"
                    row["mAP"] = metrics.get("mAP")
                except Exception:
                    print(f"  EVALUATE ERROR:")
                    traceback.print_exc()
                    row["evaluate"] = "FAILED"

        # ── test ──────────────────────────────────────────────────────────────
        if run_test:
            print(f"\n  ── test {'─' * 48}")
            ckpt = best_ckpt or str(Path(cfg.training.checkpoint_dir) / "best.ckpt")
            if not Path(ckpt).exists():
                print(f"  SKIP: checkpoint not found: {ckpt}")
                row["test"] = "SKIP (no checkpoint)"
            else:
                try:
                    test_model(model_name, cfg, ckpt)
                    row["test"] = "OK"
                except Exception:
                    print(f"  TEST ERROR:")
                    traceback.print_exc()
                    row["test"] = "FAILED"

        summary.append(row)

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 64}")
    print(f"  SUMMARY  (tag: {tag})")
    print(f"{'═' * 64}")
    col_w = 18
    header = (
        f"  {'Model':<16}"
        f"{'Train':>{col_w}}"
        f"{'Evaluate':>{col_w}}"
        f"{'Test':>{col_w}}"
        f"{'mAP':>10}"
    )
    print(header)
    print(f"  {'─' * 62}")
    for row in summary:
        map_str = f"{row['mAP']:.4f}" if row["mAP"] is not None else "—"
        print(
            f"  {row['model']:<16}"
            f"{row['train']:>{col_w}}"
            f"{row['evaluate']:>{col_w}}"
            f"{row['test']:>{col_w}}"
            f"{map_str:>10}"
        )
    print(f"{'═' * 64}\n")


if __name__ == "__main__":
    main()
