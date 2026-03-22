#!/usr/bin/env python3
"""Build the SigLIP reference gallery.

Run this once (with network access) before packaging the submission.
Outputs are written to models/final/siglip/ and copied to prompt_data/.

Usage:
    python build_gallery.py [--device cuda|cpu]
"""
from __future__ import annotations

import argparse
from inference.siglip_classifier import build_gallery

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SigLIP product gallery")
    parser.add_argument(
        "--device", default=None,
        help="torch device: 'cuda', 'cpu', etc. (default: auto-detect)"
    )
    args = parser.parse_args()

    summary = build_gallery(device=args.device)

    print("\nSummary dict:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
