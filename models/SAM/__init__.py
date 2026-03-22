# models/sam/__init__.py
from __future__ import annotations

__all__ = ["build_model"]


def __getattr__(name: str):
    if name == "build_model":
        from models.sam.model import build_model
        globals()["build_model"] = build_model
        return build_model
    raise AttributeError(f"module 'models.sam' has no attribute {name!r}")
