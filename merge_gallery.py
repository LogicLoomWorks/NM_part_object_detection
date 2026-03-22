#!/usr/bin/env python3
"""Merge gallery .npy files into a single .npz archive.

Loads gallery/gallery_embeddings.npy and gallery/gallery_category_ids.npy,
saves them together into gallery/gallery_combined.npz with keys
'embeddings' and 'category_ids'.
"""
from pathlib import Path

import numpy as np

EMB_SRC = Path("gallery/gallery_embeddings.npy")
CAT_SRC = Path("gallery/gallery_category_ids.npy")
DST     = Path("gallery/gallery_combined.npz")


def main() -> None:
    print(f"Loading {EMB_SRC} ({EMB_SRC.stat().st_size / 1e6:.3f} MB) ...")
    embeddings = np.load(str(EMB_SRC), allow_pickle=False)

    print(f"Loading {CAT_SRC} ({CAT_SRC.stat().st_size / 1e6:.4f} MB) ...")
    category_ids = np.load(str(CAT_SRC), allow_pickle=False)

    print(f"  embeddings shape:   {embeddings.shape}  dtype={embeddings.dtype}")
    print(f"  category_ids shape: {category_ids.shape}  dtype={category_ids.dtype}")

    print(f"Saving to {DST} ...")
    np.savez(str(DST), embeddings=embeddings, category_ids=category_ids)

    size_mb = DST.stat().st_size / 1e6
    print(f"Saved. Output size: {size_mb:.3f} MB")


if __name__ == "__main__":
    main()
