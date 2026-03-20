"""
training/transforms.py

Albumentations-based transform pipelines.

Public names expected by callers
---------------------------------
  get_train_transforms(max_size)   ← notebook imports this
  get_val_transforms(max_size)

  build_train_transforms(max_size) ← dataset.py imports this
  build_val_transforms(max_size)   ← dataset.py imports this
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


def get_train_transforms(max_size: int = 800) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max_size),
            A.PadIfNeeded(
                min_height=None,
                min_width=None,
                pad_height_divisor=32,
                pad_width_divisor=32,
                border_mode=0,
                value=0,
            ),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.4, contrast=0.4,
                saturation=0.4, hue=0.1, p=0.8,
            ),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco",          # [x,y,w,h] — dataset converts to cx/cy/w/h after
            label_fields=["labels"],
            min_visibility=0.1,
        ),
    )


def get_val_transforms(max_size: int = 800) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max_size),
            A.PadIfNeeded(
                min_height=None,
                min_width=None,
                pad_height_divisor=32,
                pad_width_divisor=32,
                border_mode=0,
                value=0,
            ),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["labels"],
            min_visibility=0.1,
        ),
    )


# ── aliases so dataset.py's  `from training.transforms import
#    build_train_transforms, build_val_transforms`  keeps working ──────────────
build_train_transforms = get_train_transforms
build_val_transforms   = get_val_transforms