"""Albumentations transforms for GroundingDINO-style object detection.

Design goals:
- keep boxes in COCO format [x, y, w, h] while inside Albumentations
- resize longest side to max_size
- pad spatial size to a multiple of 32
- apply light but useful augmentation for shelf-product detection
- normalize using ImageNet statistics
- convert image to PyTorch tensor
"""

from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _bbox_params() -> A.BboxParams:
    """Shared bbox configuration for COCO-format detection boxes."""
    return A.BboxParams(
        format="coco",
        label_fields=["labels"],
        min_area=4.0,
        min_visibility=0.3,
        clip=True,
    )


def build_train_transforms(max_size: int = 800) -> A.Compose:
    """Training transforms.

    Pipeline:
    1. resize longest side to max_size
    2. pad to nearest multiple of 32
    3. random horizontal flip
    4. mild color / intensity augmentation
    5. mild blur / noise augmentation
    6. normalize
    7. convert to tensor
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max_size),

            A.PadIfNeeded(
                min_height=None,
                min_width=None,
                pad_height_divisor=32,
                pad_width_divisor=32,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
            ),

            A.HorizontalFlip(p=0.5),

            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=1.0,
                    ),
                    A.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.05,
                        p=1.0,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=8,
                        sat_shift_limit=15,
                        val_shift_limit=10,
                        p=1.0,
                    ),
                ],
                p=0.5,
            ),

            A.OneOf(
                [
                    A.GaussNoise(p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ],
                p=0.2,
            ),

            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=_bbox_params(),
    )


def build_val_transforms(max_size: int = 800) -> A.Compose:
    """Validation / inference transforms.

    No random augmentation.
    Only resize, pad, normalize, and convert to tensor.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=max_size),

            A.PadIfNeeded(
                min_height=None,
                min_width=None,
                pad_height_divisor=32,
                pad_width_divisor=32,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
            ),

            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        bbox_params=_bbox_params(),
    )