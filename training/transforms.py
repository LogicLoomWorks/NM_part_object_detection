"""
Albumentations-based transform pipelines.

Box format passed to albumentations: 'albumentations'
  (normalised [x_min, y_min, x_max, y_max])
  Albumentations expects this format internally, so we convert from
  normalised cx/cy/w/h → albumentations format before the pipeline
  and back afterwards via a wrapper.

For simplicity we wrap the pipeline so the dataset can always pass
  bboxes=[cx,cy,w,h] (normalised) and labels=[int].
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _cxcywh_to_albu(boxes: list) -> list:
    """Normalised cx/cy/w/h → normalised x_min/y_min/x_max/y_max."""
    out = []
    for cx, cy, w, h in boxes:
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        out.append([
            max(0.0, x1), max(0.0, y1),
            min(1.0, x2), min(1.0, y2),
        ])
    return out


def _albu_to_cxcywh(boxes: list) -> list:
    """Normalised x_min/y_min/x_max/y_max → normalised cx/cy/w/h."""
    out = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1
        out.append([cx, cy, w, h])
    return out


class BoxConvertWrapper:
    """
    Wraps an albumentations Compose pipeline.
    Accepts bboxes in normalised cx/cy/w/h.
    Returns  bboxes in normalised cx/cy/w/h.

    Accepts both positional and keyword arguments so it is compatible
    with dataset.__getitem__ which calls:
        self.transforms(image=img, bboxes=boxes, labels=labels)
    """

    def __init__(self, pipeline: A.Compose):
        self.pipeline = pipeline

    def __call__(self, image=None, bboxes=None, labels=None):
        # Defensive defaults
        if bboxes is None:
            bboxes = []
        if labels is None:
            labels = []

        albu_boxes = _cxcywh_to_albu(bboxes)
        result = self.pipeline(
            image=image,
            bboxes=albu_boxes,
            category_ids=labels,
        )
        out_boxes  = _albu_to_cxcywh(result["bboxes"])
        out_labels = result["category_ids"]
        return {
            "image":  result["image"],
            "bboxes": out_boxes,
            "labels": out_labels,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Public factories
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transforms(max_size: int = 800) -> BoxConvertWrapper:
    pipeline = A.Compose([
        A.LongestMaxSize(max_size=max_size),
        A.PadIfNeeded(
            min_height=None, min_width=None,
            pad_height_divisor=32, pad_width_divisor=32,
            border_mode=0,  # cv2.BORDER_CONSTANT
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.1, p=0.5,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ],
        bbox_params=A.BboxParams(
            format="albumentations",
            label_fields=["category_ids"],
            min_visibility=0.1,
        ),
    )
    return BoxConvertWrapper(pipeline)


def get_val_transforms(max_size: int = 800) -> BoxConvertWrapper:
    pipeline = A.Compose([
        A.LongestMaxSize(max_size=max_size),
        A.PadIfNeeded(
            min_height=None, min_width=None,
            pad_height_divisor=32, pad_width_divisor=32,
            border_mode=0,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ],
        bbox_params=A.BboxParams(
            format="albumentations",
            label_fields=["category_ids"],
            min_visibility=0.1,
        ),
    )
    return BoxConvertWrapper(pipeline)

"""
training/transforms.py
----------------------
Albumentations-based transform pipelines for training and validation.

Exported names
--------------
build_train_transforms(max_size)   ← used internally by dataset.py
build_val_transforms(max_size)     ← used internally by dataset.py
get_train_transforms(max_size)     ← alias kept for notebook / external callers
get_val_transforms(max_size)       ← alias kept for notebook / external callers
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


def build_train_transforms(max_size: int = 800) -> A.Compose:
    """
    Full augmentation pipeline used during training.

    Box format
    ----------
    albumentations expects bboxes in 'coco' format  [x_min, y_min, w, h]
    (absolute pixel values).  COCODetectionDataset converts to normalised
    cx/cy/w/h *after* the transform is applied, so we keep 'coco' here.
    """
    return A.Compose(
        [
            # 1. Resize so the longest edge ≤ max_size
            A.LongestMaxSize(max_size=max_size, interpolation=1),
            # 2. Pad to the next multiple of 32 (bottom-right padding)
            A.PadIfNeeded(
                min_height=None,
                min_width=None,
                pad_height_divisor=32,
                pad_width_divisor=32,
                position="top_left",        # keeps (0,0) valid → mask convention
                border_mode=0,              # cv2.BORDER_CONSTANT
                value=0,
            ),
            # 3. Geometric augmentations
            A.HorizontalFlip(p=0.5),
            # 4. Colour augmentations
            A.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.1,
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.1),
            # 5. Normalise → tensor
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_visibility=0.1,     # drop boxes that become nearly invisible
            min_area=1.0,
        ),
    )


def build_val_transforms(max_size: int = 800) -> A.Compose:
    """
    Deterministic pipeline used for validation / inference.
    No geometric or colour augmentations — only resize, pad, normalise.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max_size, interpolation=1),
            A.PadIfNeeded(
                min_height=None,
                min_width=None,
                pad_height_divisor=32,
                pad_width_divisor=32,
                position="top_left",
                border_mode=0,
                value=0,
            ),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_visibility=0.1,
            min_area=1.0,
        ),
    )


# ── Aliases so notebook cells and any external callers keep working ──────────
get_train_transforms = build_train_transforms
get_val_transforms   = build_val_transforms