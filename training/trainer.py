"""PyTorch Lightning training module for GroundingDINO.

Handles:
  - Model construction from config
  - AdamW optimizer with separate backbone / rest learning rates
  - MultiStepLR scheduling
  - Reproducible train/val split from a single COCO annotations file
  - Per-step loss logging to Lightning's logger (wandb, tensorboard, etc.)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

from models.groundingdino.matcher import HungarianMatcher
from models.groundingdino.model import build_model

if TYPE_CHECKING:
    from models.groundingdino.model import GroundingDINO
from training.dataset import COCODetectionDataset, collate_fn
from training.transforms import build_train_transforms, build_val_transforms
from training.losses import SetCriterion


class DetectionLightningModule(pl.LightningModule):
    """Lightning wrapper for the GroundingDINO detector."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg

        self.model: GroundingDINO = build_model(cfg)

        matcher = HungarianMatcher(
            cost_class=cfg.training.matcher.cost_class,
            cost_bbox=cfg.training.matcher.cost_bbox,
            cost_giou=cfg.training.matcher.cost_giou,
        )

        # Build weight_dict covering the last layer and all aux layers
        lw = cfg.training.loss_weights
        weight_dict: Dict[str, float] = {
            "loss_ce": float(lw.loss_ce),
            "loss_bbox": float(lw.loss_bbox),
            "loss_giou": float(lw.loss_giou),
        }
        for i in range(cfg.model.transformer.num_decoder_layers - 1):
            weight_dict[f"loss_ce_{i}"] = float(lw.loss_ce)
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
        self._val_dataset: Optional[Subset] = None

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
            {k: v.to(self.device) for k, v in t.items()}
            for t in batch["targets"]
        ]
        losses = self.criterion(outputs, targets)

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
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        backbone_ids = {id(p) for p in self.model.backbone.parameters()}
        backbone_params = [
            p for p in self.model.backbone.parameters() if p.requires_grad
        ]
        other_params = [
            p
            for p in self.model.parameters()
            if id(p) not in backbone_ids and p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.cfg.training.backbone_lr},
                {"params": other_params, "lr": self.cfg.training.base_lr},
            ],
            weight_decay=self.cfg.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(self.cfg.training.lr_milestones),
            gamma=self.cfg.training.lr_gamma,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # ------------------------------------------------------------------
    # Data setup: reproducible train/val split from single annotations file
    # ------------------------------------------------------------------

    def setup(self, stage: str) -> None:
        if self._train_dataset is not None:
            return  # already set up (e.g. called twice by Lightning)

        # Use a lightweight probe to discover total dataset size and indices
        probe = COCODetectionDataset(
            ann_file=self.cfg.data.ann_file,
            img_dir=self.cfg.data.img_dir,
            transforms=None,
        )
        n = len(probe)
        del probe

        n_val = max(1, int(n * self.cfg.data.val_split))
        n_train = n - n_val

        generator = torch.Generator().manual_seed(self.cfg.data.seed)
        perm = torch.randperm(n, generator=generator).tolist()
        train_indices = perm[:n_train]
        val_indices = perm[n_train:]

        # Create two separate dataset instances with different transforms
        # so train and val augmentation don't interfere with each other
        train_ds = COCODetectionDataset(
            ann_file=self.cfg.data.ann_file,
            img_dir=self.cfg.data.img_dir,
            transforms=build_train_transforms(self.cfg.data.max_size),
        )
        val_ds = COCODetectionDataset(
            ann_file=self.cfg.data.ann_file,
            img_dir=self.cfg.data.img_dir,
            transforms=build_val_transforms(self.cfg.data.max_size),
        )
        self._train_dataset = Subset(train_ds, train_indices)
        self._val_dataset = Subset(val_ds, val_indices)

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