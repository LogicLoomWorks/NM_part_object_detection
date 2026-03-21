"""PyTorch Lightning training module for multi-model detection.

Handles:
  - Model dispatch via MODEL_REGISTRY (GroundingDINO, DEIMv2, SAM, YOLOv8x, RT-DETR)
  - Multi-source COCO dataset loading with ConcatDataset
  - Configurable train / val / test split driven by run_config.DATA
  - Optimizer dispatch (sgd / adam / adamw) from run_config.TUNING
  - Cosine-with-warmup / step / none scheduler from run_config.TUNING
  - Gradient clipping via configure_gradient_clipping hook
  - Backbone layer freezing from run_config.TUNING
  - Component enable/disable state logged from run_config.COMPONENTS
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Subset
import pytorch_lightning as pl

from models import MODEL_REGISTRY
from models.groundingdino.matcher import HungarianMatcher
import run_config as _rc
from training.dataset import COCODetectionDataset, collate_fn
from training.transforms import build_train_transforms, build_val_transforms
from training.losses import SetCriterion

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_source(source) -> tuple[str, str]:
    """Return (ann_file, img_dir) from a DATA source entry.

    Accepts either:
      - a str path to a directory containing ``annotations.json`` + ``images/``
      - a dict with ``"ann_file"`` and ``"img_dir"`` keys
    """
    if isinstance(source, dict):
        return source["ann_file"], source["img_dir"]

    base = Path(source)
    if not base.exists():
        raise ValueError(f"DATA source directory not found: {base}")
    ann_file = base / "annotations.json"
    img_dir  = base / "images"
    if not ann_file.exists():
        raise ValueError(f"annotations.json not found in source {base}")
    if not img_dir.exists():
        raise ValueError(f"images/ directory not found in source {base}")
    return str(ann_file), str(img_dir)


def _freeze_backbone_layers(model: torch.nn.Module, frozen_layers: int) -> None:
    """Freeze the first *frozen_layers* children of the backbone.

    For TimmBackbone the underlying timm model lives at ``model.backbone.model``.
    For _BackboneProxy (YOLOv8x / RT-DETR) the proxy itself is the backbone.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        log.warning("Model has no .backbone attribute — skipping backbone freeze.")
        return

    # TimmBackbone wraps the timm model in .model; fall back to the proxy itself
    inner    = getattr(backbone, "model", backbone)
    children = list(inner.children())
    n_freeze = min(frozen_layers, len(children))

    frozen_params = 0
    for child in children[:n_freeze]:
        for param in child.parameters():
            param.requires_grad_(False)
            frozen_params += param.numel()

    log.info("Froze %d backbone child(ren) (%d parameters).", n_freeze, frozen_params)


# ──────────────────────────────────────────────────────────────────────────────
# Lightning module
# ──────────────────────────────────────────────────────────────────────────────

class DetectionLightningModule(pl.LightningModule):
    """Lightning wrapper for any detector registered in MODEL_REGISTRY."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg

        # ── Model dispatch ────────────────────────────────────────────────
        model_name = cfg.model.name
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )
        self.model = MODEL_REGISTRY[model_name](cfg)

        # ── Loss / matcher ────────────────────────────────────────────────
        matcher = HungarianMatcher(
            cost_class=cfg.training.matcher.cost_class,
            cost_bbox=cfg.training.matcher.cost_bbox,
            cost_giou=cfg.training.matcher.cost_giou,
        )

        lw = cfg.training.loss_weights
        weight_dict: Dict[str, float] = {
            "loss_ce":   float(lw.loss_ce),
            "loss_bbox": float(lw.loss_bbox),
            "loss_giou": float(lw.loss_giou),
        }
        # Auxiliary decoder-layer losses — only when the model uses aux_loss
        if cfg.model.get("aux_loss", False):
            num_dec = cfg.model.transformer.num_decoder_layers
            for i in range(num_dec - 1):
                weight_dict[f"loss_ce_{i}"]   = float(lw.loss_ce)
                weight_dict[f"loss_bbox_{i}"] = float(lw.loss_bbox)
                weight_dict[f"loss_giou_{i}"] = float(lw.loss_giou)

        self.criterion = SetCriterion(
            num_classes=cfg.model.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            focal_alpha=float(cfg.training.get("focal_alpha", 0.25)),
            focal_gamma=float(cfg.training.get("focal_gamma", 2.0)),
        )

        self._train_dataset: Optional[Subset] = None
        self._val_dataset:   Optional[Subset] = None

    # ------------------------------------------------------------------
    # Forward / step
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
    ) -> Dict:
        return self.model(images, image_mask)

    def _shared_step(self, batch: Dict, stage: str) -> torch.Tensor:
        images = batch["images"].to(self.device)
        image_mask = batch.get("masks")
        if image_mask is not None:
            image_mask = image_mask.to(self.device)

        outputs = self.model(images, image_mask)
        targets = [
            {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()}
            for t in batch["targets"]
        ]
        losses = self.criterion(outputs, targets)
        losses["loss_total"] = sum(losses.values())

        for k, v in losses.items():
            self.log(
                f"{stage}/{k}",
                v,
                prog_bar=(k == "loss_total"),
                batch_size=len(targets),
                sync_dist=True,
            )
        return losses["loss_total"]

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    # ------------------------------------------------------------------
    # Gradient clipping
    # ------------------------------------------------------------------

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ) -> None:
        max_norm = float(_rc.TUNING.get("grad_clip_max_norm", 1.0))
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        base_lr      = float(_rc.TUNING["learning_rate"])
        backbone_lr  = float(self.cfg.training.backbone_lr)
        weight_decay = float(_rc.TUNING["weight_decay"])

        backbone_ids = {id(p) for p in self.model.backbone.parameters()}
        backbone_params = [
            p for p in self.model.backbone.parameters() if p.requires_grad
        ]
        other_params = [
            p for p in self.model.parameters()
            if id(p) not in backbone_ids and p.requires_grad
        ]

        param_groups = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": other_params,    "lr": base_lr},
        ]

        opt_name = _rc.TUNING.get("optimizer", "adamw").lower()
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(
                param_groups, momentum=0.9, weight_decay=weight_decay
            )
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
        else:  # "adamw" (default)
            optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        sched_name    = _rc.TUNING.get("scheduler", "cosine").lower()
        max_epochs    = int(self.cfg.training.max_epochs)
        warmup_epochs = int(_rc.TUNING.get("warmup_epochs", 3))

        if sched_name == "cosine":
            from torch.optim.lr_scheduler import (  # noqa: PLC0415
                CosineAnnealingLR, LinearLR, SequentialLR,
            )
            warmup = LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=max(1, max_epochs - warmup_epochs),
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        if sched_name == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=list(self.cfg.training.lr_milestones),
                gamma=float(self.cfg.training.lr_gamma),
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        # "none" — no scheduler
        return {"optimizer": optimizer}

    # ------------------------------------------------------------------
    # Data setup: multi-source ConcatDataset with configurable splits
    # ------------------------------------------------------------------

    def setup(self, stage: str) -> None:
        if self._train_dataset is not None:
            return  # already initialised (Lightning may call setup twice)

        # ── COMPONENTS log ────────────────────────────────────────────────
        for component, enabled in _rc.COMPONENTS.items():
            log.info(
                "Component %-12s: %s",
                component,
                "enabled" if enabled else "disabled",
            )

        # ── Parse sources ─────────────────────────────────────────────────
        sources: List[tuple[str, str]] = [
            _parse_source(src) for src in _rc.DATA["sources"]
        ]

        max_size = int(self.cfg.data.max_size)
        seed     = int(self.cfg.data.seed)

        # ── Probe total dataset size ──────────────────────────────────────
        probe_datasets = [
            COCODetectionDataset(ann_file=ann, img_dir=img, transforms=None)
            for ann, img in sources
        ]
        n = len(ConcatDataset(probe_datasets))
        del probe_datasets

        # ── Compute index split ───────────────────────────────────────────
        ratios  = _rc.DATA["split_ratios"]
        n_train = int(n * ratios["train"])
        n_val   = int(n * ratios["val"])
        # n_test absorbs rounding remainder
        n_test  = n - n_train - n_val

        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=generator).tolist()

        train_indices = perm[:n_train]
        val_indices   = perm[n_train : n_train + n_val]
        test_indices  = perm[n_train + n_val :]

        if not _rc.DATA.get("use_test_split", False):
            # Merge test fraction into validation when no separate test set needed
            val_indices = val_indices + test_indices

        # ── Build datasets with transforms ────────────────────────────────
        train_concat = ConcatDataset([
            COCODetectionDataset(
                ann_file=ann,
                img_dir=img,
                transforms=build_train_transforms(max_size),
            )
            for ann, img in sources
        ])
        val_concat = ConcatDataset([
            COCODetectionDataset(
                ann_file=ann,
                img_dir=img,
                transforms=build_val_transforms(max_size),
            )
            for ann, img in sources
        ])

        self._train_dataset = Subset(train_concat, train_indices)
        self._val_dataset   = Subset(val_concat,   val_indices)

        log.info(
            "Dataset split — train: %d  val: %d  (use_test_split=%s, sources=%d)",
            len(train_indices), len(val_indices),
            _rc.DATA.get("use_test_split", False),
            len(sources),
        )

        # ── Backbone freeze ───────────────────────────────────────────────
        if _rc.TUNING.get("freeze_backbone", False):
            frozen_layers = int(_rc.TUNING.get("frozen_layers", 0))
            if frozen_layers > 0:
                _freeze_backbone_layers(self.model, frozen_layers)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    module = DetectionLightningModule(cfg)

    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename="epoch={epoch:02d}-val_loss={val/loss_total:.4f}",
        monitor="val/loss_total",
        mode="min",
        save_top_k=3,
        auto_insert_metric_name=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        # gradient_clip_val drives configure_gradient_clipping; value is
        # overridden inside the hook using _rc.TUNING["grad_clip_max_norm"]
        gradient_clip_val=_rc.TUNING.get("grad_clip_max_norm", 1.0),
        log_every_n_steps=cfg.training.log_every_n_steps,
        callbacks=[checkpoint_cb, lr_monitor],
        default_root_dir=cfg.training.checkpoint_dir,
    )

    trainer.fit(module)
