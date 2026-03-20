# models/groundingdino/__init__.py
#
# Lazy-load the heavy model symbols so that lightweight sub-modules
# (backbone, neck, transformer) can be imported in isolation — e.g.
# from a notebook cell or a unit test — without loading the full model.

from __future__ import annotations

__all__ = [
    "GroundingDINO",
    "build_model",
    "save_checkpoint",
    "load_checkpoint",
    "TimmBackbone",
]

_HEAVY = {"GroundingDINO", "build_model", "save_checkpoint", "load_checkpoint"}
_LIGHT = {"TimmBackbone"}


def __getattr__(name: str):
    if name in _HEAVY:
        from models.groundingdino.model import (   # noqa: PLC0415
            GroundingDINO, build_model, save_checkpoint, load_checkpoint,
        )
        g = globals()
        g["GroundingDINO"]   = GroundingDINO
        g["build_model"]     = build_model
        g["save_checkpoint"] = save_checkpoint
        g["load_checkpoint"] = load_checkpoint
        return g[name]

    if name in _LIGHT:
        from models.groundingdino.backbone import TimmBackbone  # noqa: PLC0415
        globals()["TimmBackbone"] = TimmBackbone
        return TimmBackbone

    raise AttributeError(f"module 'models.groundingdino' has no attribute {name!r}")