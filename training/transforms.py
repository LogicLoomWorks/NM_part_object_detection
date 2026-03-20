"""
training/transforms.py
----------------------
Albumentations-based transform pipelines for training and validation.

Box format used by albumentations: 'coco'  [x_min, y_min, w, h]  (absolute pixels).
COCODetectionDataset converts to normalised cx/cy/w/h after the transform.

Exported names
--------------
build_train_transforms(max_size)   <- used by dataset.py
build_val_transforms(max_size)     <- used by dataset.py
get_train_transforms(max_size)     <- alias for notebooks / external callers
get_val_transforms(max_size)       <- alias for notebooks / external callers
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


def build_train_transforms(max_size: int = 800) -> A.Compose:
    """Full augmentation pipeline used during training."""
    return A.Compose(
        [
            # 1. Resize so the longest edge <= max_size
            A.LongestMaxSize(max_size=max_size, interpolation=1),
            # 2. Pad to the next multiple of 32 (top-left padding keeps (0,0) valid)
            A.PadIfNeeded(
                min_height=None,
                min_width=None,
                pad_height_divisor=32,
                pad_width_divisor=32,
                position="top_left",
                border_mode=0,          # cv2.BORDER_CONSTANT
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
            # 5. Normalise -> tensor
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


def build_val_transforms(max_size: int = 800) -> A.Compose:
    """Deterministic pipeline used for validation / inference."""
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


# Aliases so notebook cells and external callers keep working
get_train_transforms = build_train_transforms
get_val_transforms   = build_val_transforms