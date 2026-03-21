# ============================================================
# models/__init__.py
# ============================================================
#
# PROBLEM: the old file did `from models.groundingdino.model import ...`
# at the top level.  That forced the *entire* model graph to load —
# including transformer.py, which tried to import transformers (HuggingFace)
# — just to do `from models.groundingdino.neck import build_neck`.
#
# FIX: use lazy __getattr__ so sub-modules can be imported independently
# without triggering the full model load.

from __future__ import annotations

__all__ = ["GroundingDINO", "build_model", "MODEL_REGISTRY"]

# Nothing is imported eagerly.  Importing `models.groundingdino.neck`
# or `models.groundingdino.backbone` directly will now work without side-effects.


def _build_groundingdino(cfg):
    from models.groundingdino.model import build_model  # noqa: PLC0415
    return build_model(cfg)


def _build_deimv2(cfg):
    from models.deimv2.model import build_model  # noqa: PLC0415
    return build_model(cfg)


def _build_sam(cfg):
    from models.sam.model import build_model  # noqa: PLC0415
    return build_model(cfg)


# Registry maps model name → build_model(cfg) factory.
# Each entry is a thin wrapper that defers the actual import until called,
# preserving the lazy-loading guarantee above.
MODEL_REGISTRY: dict[str, callable] = {
    "groundingdino": _build_groundingdino,
    "deimv2":        _build_deimv2,
    "sam":           _build_sam,
}


def __getattr__(name: str):
    if name in ("GroundingDINO", "build_model", "save_checkpoint", "load_checkpoint"):
        from models.groundingdino.model import (   # noqa: PLC0415
            GroundingDINO, build_model, save_checkpoint, load_checkpoint,
        )
        g = globals()
        g["GroundingDINO"]    = GroundingDINO
        g["build_model"]      = build_model
        g["save_checkpoint"]  = save_checkpoint
        g["load_checkpoint"]  = load_checkpoint
        return g[name]
    raise AttributeError(f"module 'models' has no attribute {name!r}")