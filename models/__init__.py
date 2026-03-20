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

__all__ = ["GroundingDINO", "build_model"]

# Nothing is imported eagerly.  Importing `models.groundingdino.neck`
# or `models.groundingdino.backbone` directly will now work without side-effects.

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