"""COCO detection dataset for GroundingDINO-style training.

This dataset:
- loads COCO annotations
- remaps category IDs to contiguous 0-indexed class indices
- applies Albumentations transforms
- returns boxes as normalized [cx, cy, w, h]
- provides a collate_fn that pads images in a batch

Expected target format:
    {
        "boxes": FloatTensor [N, 4]   # normalized cx, cy, w, h
        "labels": LongTensor [N]
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from training.transforms import get_train_transforms, get_val_transforms


class COCODetectionDataset(Dataset):
    """COCO-format object detection dataset."""

    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        transforms=None,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.transforms = transforms
        self.coco = COCO(ann_file)

        # Build stable contiguous label mapping:
        # original COCO category_id -> [0, num_classes-1]
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_idx: Dict[int, int] = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
        self.idx_to_cat_id: Dict[int, int] = {idx: cat_id for cat_id, idx in self.cat_id_to_idx.items()}

        # Keep only images that have at least one annotation
        self.img_ids: List[int] = sorted(
            img_id
            for img_id in self.coco.getImgIds()
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        )

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Dict:
        img_id = self.img_ids[idx]

        # ------------------------------------------------------------------
        # Load image
        # ------------------------------------------------------------------
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / img_info["file_name"]

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ------------------------------------------------------------------
        # Load annotations
        # ------------------------------------------------------------------
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        bboxes: List[List[float]] = []
        labels: List[int] = []

        for ann in anns:
            # Skip crowd boxes if present
            if ann.get("iscrowd", 0) == 1:
                continue

            x, y, w, h = ann["bbox"]

            # Filter degenerate boxes
            if w <= 1 or h <= 1:
                continue

            bboxes.append([x, y, w, h])  # Albumentations expects COCO format here
            labels.append(self.cat_id_to_idx[ann["category_id"]])

        # ------------------------------------------------------------------
        # Apply transforms
        # ------------------------------------------------------------------
        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=bboxes,
                category_ids=labels,
            )
            image = transformed["image"]              # Tensor [3, H, W]
            bboxes = list(transformed["bboxes"])      # still in COCO xywh absolute format
            labels = list(transformed["category_ids"])
            _, height, width = image.shape
        else:
            height, width = image.shape[:2]
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # ------------------------------------------------------------------
        # Convert boxes:
        # COCO [x, y, w, h] absolute
        # -> normalized [cx, cy, w, h]
        # ------------------------------------------------------------------
        if len(bboxes) > 0:
            boxes_np = np.asarray(bboxes, dtype=np.float32)

            x = boxes_np[:, 0]
            y = boxes_np[:, 1]
            w = boxes_np[:, 2]
            h = boxes_np[:, 3]

            cx = (x + 0.5 * w) / width
            cy = (y + 0.5 * h) / height
            w = w / width
            h = h / height

            boxes = np.stack([cx, cy, w, h], axis=1)
            boxes = np.clip(boxes, 0.0, 1.0)

            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.long)

        return {
            "image": image,
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": img_id,
            "orig_size": torch.tensor([img_info["height"], img_info["width"]], dtype=torch.long),
            "size": torch.tensor([height, width], dtype=torch.long),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for detection batches.

    Pads images to the largest H and W in the batch.

    Returns:
        {
            "images": FloatTensor [B, 3, Hmax, Wmax]
            "image_mask": BoolTensor [B, Hmax, Wmax]
            "targets": List[Dict]
            "image_ids": List[int]
            "orig_sizes": LongTensor [B, 2]
            "sizes": LongTensor [B, 2]
        }

    Mask convention used here:
        False = valid image region
        True  = padded region
    This is the convention commonly expected by transformer models.
    """
    batch_size = len(batch)
    max_h = max(sample["image"].shape[1] for sample in batch)
    max_w = max(sample["image"].shape[2] for sample in batch)

    images = torch.zeros((batch_size, 3, max_h, max_w), dtype=batch[0]["image"].dtype)
    image_mask = torch.ones((batch_size, max_h, max_w), dtype=torch.bool)

    targets = []
    image_ids = []
    orig_sizes = []
    sizes = []

    for i, sample in enumerate(batch):
        image = sample["image"]
        _, h, w = image.shape

        images[i, :, :h, :w] = image
        image_mask[i, :h, :w] = False  # valid region

        targets.append(
            {
                "boxes": sample["boxes"],
                "labels": sample["labels"],
                "image_id": sample["image_id"],
            }
        )
        image_ids.append(sample["image_id"])
        orig_sizes.append(sample["orig_size"])
        sizes.append(sample["size"])

    return {
        "images": images,
        "masks": image_mask,
        "targets": targets,
        "image_ids": image_ids,
        "orig_sizes": torch.stack(orig_sizes),
        "sizes": torch.stack(sizes),
    }


def build_train_dataset(
    ann_file: str,
    img_dir: str,
    max_size: int = 800,
) -> COCODetectionDataset:
    """Convenience builder for training dataset."""
    return COCODetectionDataset(
        ann_file=ann_file,
        img_dir=img_dir,
        transforms=get_train_transforms(max_size=max_size),
    )


def build_val_dataset(
    ann_file: str,
    img_dir: str,
    max_size: int = 800,
) -> COCODetectionDataset:
    """Convenience builder for validation dataset."""
    return COCODetectionDataset(
        ann_file=ann_file,
        img_dir=img_dir,
        transforms=get_val_transforms(max_size=max_size),
    )