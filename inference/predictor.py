"""High-level inference API.

Predictor wraps the trained model for:
  - Single-image inference (predict_image)
  - Full-dataset COCO evaluation (evaluate)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from models.groundingdino.model import load_checkpoint
from training.dataset import COCODetectionDataset, collate_fn
from training.transforms import build_val_transforms
from inference.postprocess import apply_nms, threshold_filter


class Predictor:
    """Run inference with a trained GroundingDINO checkpoint."""

    def __init__(
        self,
        cfg: DictConfig,
        checkpoint_path: str,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = load_checkpoint(checkpoint_path, cfg)
        self.model.to(self.device).eval()
        self.transforms = build_val_transforms(cfg.data.max_size)

    @torch.no_grad()
    def predict_image(self, image_path: str) -> Dict:
        """Run inference on a single image.

        Returns:
            scores: np.ndarray (N,)  confidence scores
            labels: np.ndarray (N,)  0-indexed class indices
            boxes:  np.ndarray (N, 4) absolute x1/y1/x2/y2 pixel coordinates
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        result = self.transforms(image=image, bboxes=[], labels=[])
        tensor = result["image"].unsqueeze(0).to(self.device)  # (1, 3, H, W)
        _, _, H, W = tensor.shape

        outputs = self.model(tensor)
        preds = threshold_filter(
            outputs["pred_logits"],
            outputs["pred_boxes"],
            threshold=self.cfg.inference.threshold,
            top_k=self.cfg.inference.top_k,
        )
        preds = apply_nms(preds, iou_threshold=self.cfg.inference.nms_threshold)
        pred = preds[0]

        # Convert normalised cx/cy/w/h → absolute x1/y1/x2/y2 in original image space
        scale_x = orig_w / W
        scale_y = orig_h / H
        cx, cy, bw, bh = pred["boxes"].unbind(-1)
        x1 = (cx - bw / 2) * W * scale_x
        y1 = (cy - bh / 2) * H * scale_y
        x2 = (cx + bw / 2) * W * scale_x
        y2 = (cy + bh / 2) * H * scale_y
        boxes_abs = torch.stack([x1, y1, x2, y2], dim=-1)

        return {
            "scores": pred["scores"].cpu().numpy(),
            "labels": pred["labels"].cpu().numpy(),
            "boxes": boxes_abs.cpu().numpy(),
        }

    @torch.no_grad()
    def evaluate(self, ann_file: str, img_dir: str) -> Dict:
        """Run COCO evaluation on the full dataset.

        Returns:
            {'mAP': float, 'mAP_50': float, 'mAP_75': float}
        """
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        dataset = COCODetectionDataset(
            ann_file=ann_file,
            img_dir=img_dir,
            transforms=build_val_transforms(self.cfg.data.max_size),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            collate_fn=collate_fn,
        )

        coco_results: List[Dict] = []
        for batch in loader:
            images = batch["images"].to(self.device)
            image_mask = batch.get("masks")          # fix: was "image_mask"
            if image_mask is not None:
                image_mask = image_mask.to(self.device)
            _, _, H, W = images.shape

            outputs = self.model(images, image_mask)
            preds = threshold_filter(
                outputs["pred_logits"],
                outputs["pred_boxes"],
                threshold=self.cfg.inference.threshold,
                top_k=self.cfg.inference.top_k,
            )
            preds = apply_nms(preds, iou_threshold=self.cfg.inference.nms_threshold)

            for pred, img_id in zip(preds, batch["image_ids"]):
                cx, cy, bw, bh = pred["boxes"].unbind(-1)
                x1 = (cx - bw / 2) * W
                y1 = (cy - bh / 2) * H
                abs_w = bw * W
                abs_h = bh * H
                for score, label, x, y, w, h in zip(
                    pred["scores"].tolist(),
                    pred["labels"].tolist(),
                    x1.tolist(),
                    y1.tolist(),
                    abs_w.tolist(),
                    abs_h.tolist(),
                ):
                    coco_results.append(
                        {
                            "image_id": int(img_id),
                            "category_id": dataset.idx_to_cat_id[int(label)],
                            "bbox": [x, y, w, h],
                            "score": float(score),
                        }
                    )

        if not coco_results:
            return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}

        coco_gt = COCO(ann_file)
        coco_dt = coco_gt.loadRes(coco_results)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        return {
            "mAP": float(evaluator.stats[0]),
            "mAP_50": float(evaluator.stats[1]),
            "mAP_75": float(evaluator.stats[2]),
        }


def run_inference(
    cfg: DictConfig,
    input_path: str,
    output_dir: str,
    checkpoint_path: str,
) -> None:
    """CLI helper: run inference on a single image and save JSON results."""
    predictor = Predictor(cfg, checkpoint_path)
    result = predictor.predict_image(input_path)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(input_path).stem + "_predictions.json")
    out_path.write_text(
        json.dumps(
            {
                "scores": result["scores"].tolist(),
                "labels": result["labels"].tolist(),
                "boxes": result["boxes"].tolist(),
            },
            indent=2,
        )
    )
    print(f"Predictions saved to {out_path}")


def run_evaluation(cfg: DictConfig, checkpoint_path: str) -> None:
    """CLI helper: evaluate a checkpoint on the full training dataset."""
    predictor = Predictor(cfg, checkpoint_path)
    metrics = predictor.evaluate(cfg.data.ann_file, cfg.data.img_dir)
    print("Evaluation results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")