#!/usr/bin/env python3
"""Submission entrypoint for the GroundingDINO product detector.

Usage
-----
Train with default config (ResNet-50 backbone):
    python run.py train

Train with a different backbone:
    python run.py train --backbone convnext_base

Run inference on a single image:
    python run.py infer --checkpoint models/checkpoints/best.ckpt \
                        --input path/to/image.jpg --output results/

Evaluate a checkpoint on the full dataset:
    python run.py eval --checkpoint models/checkpoints/best.ckpt

All modes accept --config to override the default config file, and
--backbone to override only the backbone section.
"""
import argparse
import sys
from pathlib import Path

# Ensure the project root is importable as a Python package root
sys.path.insert(0, str(Path(__file__).parent))

from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GroundingDINO product detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("mode", choices=["train", "infer", "eval"])
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to OmegaConf YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--backbone",
        default=None,
        choices=["resnet50", "resnet101", "convnext_base"],
        help="Override backbone from configs/backbone/<name>.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to model checkpoint (.ckpt or .pt)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input image path (infer mode)",
    )
    parser.add_argument(
        "--output",
        default="results/",
        help="Output directory for predictions (infer mode)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = OmegaConf.load(args.config)

    # Optionally override backbone section from a dedicated backbone config
    if args.backbone:
        backbone_cfg = OmegaConf.load(f"configs/backbone/{args.backbone}.yaml")
        cfg = OmegaConf.merge(cfg, {"model": {"backbone": backbone_cfg}})

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    if args.mode == "train":
        import pytorch_lightning as pl
        from pytorch_lightning.loggers import CSVLogger
        from training.trainer import DetectionLightningModule

        logger = CSVLogger(
            save_dir="logs",
            name="groundingdino_product_detector",
            version=args.backbone or "default",
        )

        module = DetectionLightningModule(cfg)

        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            gradient_clip_val=cfg.training.grad_clip,
            log_every_n_steps=cfg.training.log_every_n_steps,
            default_root_dir=cfg.training.checkpoint_dir,
            logger=logger,
        )
        trainer.fit(module)

    # ------------------------------------------------------------------
    # Infer
    # ------------------------------------------------------------------
    elif args.mode == "infer":
        if args.checkpoint is None:
            print("Error: --checkpoint is required for infer mode", file=sys.stderr)
            sys.exit(1)
        if args.input is None:
            print("Error: --input is required for infer mode", file=sys.stderr)
            sys.exit(1)

        from inference.predictor import run_inference
        run_inference(cfg, args.input, args.output, args.checkpoint)

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------
    elif args.mode == "eval":
        if args.checkpoint is None:
            print("Error: --checkpoint is required for eval mode", file=sys.stderr)
            sys.exit(1)

        from inference.predictor import run_evaluation
        run_evaluation(cfg, args.checkpoint)


if __name__ == "__main__":
    main()