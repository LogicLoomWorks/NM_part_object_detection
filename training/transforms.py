"""
training/transforms.py

Albumentations-based transform pipelines driven by run_config.

Public names expected by callers
---------------------------------
  get_train_transforms(max_size)   <- used by notebooks
  get_val_transforms(max_size)
  build_train_transforms(max_size) <- aliased from get_train_transforms
  build_val_transforms(max_size)   <- aliased from get_val_transforms
  save_augmented_sample(image, bboxes, category_ids, filename_stem)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

import run_config

_AUG  = run_config.AUGMENTATION
_DATA = run_config.DATA

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)

_BBOX_PARAMS = A.BboxParams(
    format="coco",                   # [x, y, w, h] absolute — dataset converts after
    label_fields=["category_ids"],   # must match the key dataset.py passes
    min_visibility=0.3,              # intentionally raised from 0.1
)


# ── Train pipeline ────────────────────────────────────────────────────────────

def get_train_transforms(max_size: int = 800) -> A.Compose:
    transforms: list = [
        A.LongestMaxSize(max_size=max_size),
        A.PadIfNeeded(
            min_height=None,
            min_width=None,
            pad_height_divisor=32,
            pad_width_divisor=32,
            border_mode=0,
            fill=0,
        ),
    ]

    # ── Geometry ──────────────────────────────────────────────────────────────
    if _AUG["horizontal_flip"]["enabled"]:
        transforms.append(A.HorizontalFlip(p=_AUG["horizontal_flip"]["p"]))

    if _AUG["random_brightness_contrast"]["enabled"]:
        transforms.append(A.RandomBrightnessContrast(p=0.5))

    if _AUG["hue_saturation_value"]["enabled"]:
        transforms.append(A.HueSaturationValue(p=0.5))

    if _AUG["random_gamma"]["enabled"]:
        transforms.append(A.RandomGamma(p=0.5))

    if _AUG["clahe"]["enabled"]:
        transforms.append(A.CLAHE(p=0.5))

    if _AUG["shift_scale_rotate"]["enabled"]:
        cfg_ssr = _AUG["shift_scale_rotate"]
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=cfg_ssr["shift_limit"],
                scale_limit=cfg_ssr["scale_limit"],
                rotate_limit=min(cfg_ssr["rotate_limit"], 15),  # hard cap at 15°
                p=0.5,
            )
        )

    # ── Blur / noise ──────────────────────────────────────────────────────────
    if _AUG["blur_one_of"]["enabled"]:
        quality_lower = _AUG["blur_one_of"]["compression_quality_lower"]
        transforms.append(
            A.OneOf(
                [
                    A.GaussianBlur(p=1.0),
                    A.MotionBlur(p=1.0),
                    # albumentations >=2.0 uses quality_range instead of quality_lower
                    A.ImageCompression(
                        quality_range=(quality_lower, 100),
                        p=1.0,
                    ),
                ],
                p=0.4,
            )
        )

    if _AUG["gauss_noise"]["enabled"]:
        transforms.append(A.GaussNoise(p=0.3))

    if _AUG["iso_noise"]["enabled"]:
        transforms.append(A.ISONoise(p=0.3))

    # ── Dropout ───────────────────────────────────────────────────────────────
    if _AUG["coarse_dropout"]["enabled"]:
        cfg_cd = _AUG["coarse_dropout"]
        max_holes  = cfg_cd["max_holes"]
        max_height = cfg_cd["max_height"]
        max_width  = cfg_cd["max_width"]
        # albumentations >=2.0 renamed the parameters
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(1, max_holes),
                hole_height_range=(1, max_height),
                hole_width_range=(1, max_width),
                p=0.4,
            )
        )

    # ── Always-on: normalise + convert to tensor ──────────────────────────────
    transforms.extend([
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])

    return A.Compose(transforms, bbox_params=_BBOX_PARAMS)


# ── Val pipeline ──────────────────────────────────────────────────────────────

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
                fill=0,
            ),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ],
        bbox_params=_BBOX_PARAMS,
    )


# ── Augmented-sample saver ────────────────────────────────────────────────────

def save_augmented_sample(
    image: np.ndarray,
    bboxes: List[List[float]],
    category_ids: List[int],
    filename_stem: str,
) -> None:
    """Write an augmented image + annotations to DATA["augmented_data_dir"].

    When DATA["save_augmented"] is False this function is a no-op.

    Args:
        image:         RGB uint8 array (H, W, 3).
        bboxes:        List of COCO-format [x, y, w, h] absolute boxes.
        category_ids:  Matching list of integer category IDs.
        filename_stem: Base filename without extension (e.g. "img_00001_aug0").
    """
    if not _DATA["save_augmented"]:
        return

    out_dir = Path(_DATA["augmented_data_dir"])
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Write image (convert RGB → BGR for cv2)
    img_path = img_dir / f"{filename_stem}.jpg"
    cv2.imwrite(str(img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Append annotation records to augmented_annotations.json
    ann_path = out_dir / "augmented_annotations.json"
    if ann_path.exists():
        with ann_path.open() as f:
            data = json.load(f)
    else:
        data = {"images": [], "annotations": [], "categories": []}

    img_id = len(data["images"]) + 1
    data["images"].append(
        {"id": img_id, "file_name": f"images/{filename_stem}.jpg"}
    )
    for bbox, cat_id in zip(bboxes, category_ids):
        x, y, w, h = bbox
        data["annotations"].append(
            {
                "id": len(data["annotations"]) + 1,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
            }
        )

    with ann_path.open("w") as f:
        json.dump(data, f, indent=2)


# ── Aliases so existing callers keep working ──────────────────────────────────
build_train_transforms = get_train_transforms
build_val_transforms   = get_val_transforms
