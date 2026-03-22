# models/deimv2/__init__.py

from __future__ import annotations

__all__ = ["build_model"]


def __getattr__(name: str):
    if name == "build_model":
        from models.DEIMv2.model import build_model  # noqa: PLC0415
        globals()["build_model"] = build_model
        return build_model
    raise AttributeError(f"module 'models.deimv2' has no attribute {name!r}")
