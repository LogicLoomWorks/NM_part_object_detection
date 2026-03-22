#!/usr/bin/env python3
"""Quantize model weights from FP32 to FP16.

Section 1 — DEIMv2:
  Loads checkpoints/deimv2_best.pt, extracts the model state dict,
  converts all float32 tensors to float16, saves to
  checkpoints/deimv2_fp16.pt.

Section 2 — SigLIP:
  Loads siglip_weights/vision_model.safetensors, converts all float32
  tensors to float16, saves to siglip_weights/vision_model_fp16.safetensors.
  Prints combined submission weight total.
"""
from pathlib import Path

import torch
from safetensors.torch import load_file as safetensors_load
from safetensors.torch import save_file as safetensors_save

DEIMV2_SRC  = Path("checkpoints/deimv2_best.pt")
DEIMV2_DST  = Path("checkpoints/deimv2_fp16.pt")
SIGLIP_SRC  = Path("siglip_weights/vision_model.safetensors")
SIGLIP_DST  = Path("siglip_weights/vision_model_fp16.safetensors")
GALLERY_EMB = Path("gallery/gallery_embeddings.npy")
GALLERY_CAT = Path("gallery/gallery_category_ids.npy")


# ── Section 1: DEIMv2 ─────────────────────────────────────────────────────────

def quantize_deimv2() -> None:
    print(f"\n=== DEIMv2 ===")
    print(f"Loading {DEIMV2_SRC} ({DEIMV2_SRC.stat().st_size / 1e6:.1f} MB) ...")
    payload = torch.load(str(DEIMV2_SRC), map_location="cpu", weights_only=False)

    if isinstance(payload, dict) and "state_dict" in payload:
        print("Detected Lightning/trainer checkpoint — extracting 'state_dict' key.")
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        print("Detected raw state dict.")
        state_dict = payload
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(payload)}")

    print(f"State dict has {len(state_dict)} keys.")

    fp16_state = {}
    converted = skipped = 0
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
            fp16_state[k] = v.half()
            converted += 1
        else:
            fp16_state[k] = v
            skipped += 1

    print(f"Converted {converted} float32 tensors to float16.")
    print(f"Left unchanged: {skipped} tensors (non-float32 or non-tensor).")

    print(f"Saving to {DEIMV2_DST} ...")
    torch.save(fp16_state, str(DEIMV2_DST))

    size_mb = DEIMV2_DST.stat().st_size / 1e6
    print(f"Saved. Output size: {size_mb:.1f} MB")
    if size_mb < 240:
        print("OK — under 240 MB target.")
    else:
        print(f"WARNING — {size_mb:.1f} MB, above 240 MB target.")


# ── Section 2: SigLIP ─────────────────────────────────────────────────────────

def quantize_siglip() -> None:
    print(f"\n=== SigLIP ===")
    print(f"Loading {SIGLIP_SRC} ({SIGLIP_SRC.stat().st_size / 1e6:.1f} MB) ...")
    state_dict = safetensors_load(str(SIGLIP_SRC), device="cpu")

    print(f"State dict has {len(state_dict)} keys.")

    fp16_state = {}
    converted = skipped = 0
    for k, v in state_dict.items():
        if v.dtype == torch.float32:
            fp16_state[k] = v.half()
            converted += 1
        else:
            fp16_state[k] = v
            skipped += 1

    print(f"Converted {converted} float32 tensors to float16.")
    print(f"Left unchanged: {skipped} tensors (non-float32).")

    print(f"Saving to {SIGLIP_DST} ...")
    safetensors_save(fp16_state, str(SIGLIP_DST))

    size_mb = SIGLIP_DST.stat().st_size / 1e6
    print(f"Saved. Output size: {size_mb:.1f} MB")


# ── Combined total ─────────────────────────────────────────────────────────────

def print_combined_total() -> None:
    files = {
        DEIMV2_DST:  "checkpoints/deimv2_fp16.pt",
        SIGLIP_DST:  "siglip_weights/vision_model_fp16.safetensors",
        GALLERY_EMB: "gallery/gallery_embeddings.npy",
        GALLERY_CAT: "gallery/gallery_category_ids.npy",
    }
    print("\n=== Combined submission weight total ===")
    total = 0.0
    for path, label in files.items():
        mb = path.stat().st_size / 1e6
        total += mb
        print(f"  {label:<50}  {mb:>7.2f} MB")
    print(f"  {'TOTAL':<50}  {total:>7.2f} MB")
    limit = 420.0
    if total < limit:
        print(f"OK — {total:.2f} MB is under the {limit:.0f} MB limit.")
    else:
        print(f"WARNING — {total:.2f} MB exceeds the {limit:.0f} MB limit.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    quantize_deimv2()
    quantize_siglip()
    print_combined_total()


if __name__ == "__main__":
    main()
