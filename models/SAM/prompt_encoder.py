"""
models/sam/prompt_encoder.py

Replaces SAM's interactive prompts with a learned grid of point prompts.
Produces num_queries sparse embeddings by learning (num_queries, 2) point
coordinates in [0, 1] that are projected through SAM's prompt encoder PE layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LearnedGridPrompts(nn.Module):
    """Learned point-prompt embeddings for prompt-free SAM-based detection.

    Rather than accepting user-provided clicks or boxes, this module maintains
    a set of learnable 2-D coordinates in [0, 1] — one per detection query.
    At forward time those coordinates are embedded using SAM's positional
    encoding layer (``sam_prompt_encoder.pe_layer``), producing per-query
    sparse embeddings that drive the mask decoder.

    Args:
        num_queries: Number of detection queries (= number of point prompts).
        embed_dim:   SAM prompt embedding dimension (256 for vit_b).

    Note:
        ``sam_prompt_encoder`` is intentionally *not* stored in ``__init__``
        so this module stays lightweight and avoids circular dependencies.
        Pass it as a forward argument instead.
    """

    def __init__(self, num_queries: int, embed_dim: int) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # Learnable (x, y) coordinates, initialised on a regular grid in [0,1].
        # Shape: (num_queries, 2)
        coords = self._init_grid(num_queries)
        self.point_coords = nn.Parameter(coords)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _init_grid(n: int) -> torch.Tensor:
        """Initialise n points on a uniform 2-D grid spanning [0.1, 0.9]."""
        side = int(n ** 0.5) + 1
        xs = torch.linspace(0.1, 0.9, side)
        ys = torch.linspace(0.1, 0.9, side)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        pts = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)[:n]
        return pts  # (n, 2)

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(self, prompt_encoder: nn.Module) -> torch.Tensor:
        """Embed learned point coordinates through SAM's PE layer.

        Args:
            prompt_encoder: SAM's PromptEncoder (or any module exposing a
                ``pe_layer`` attribute that maps (N, 2) coords → embeddings).

        Returns:
            sparse_embeddings: ``(num_queries, 1, embed_dim)`` tensor —
                one embedding per query, shaped for the mask decoder.
        """
        # Clamp to (0, 1) so coordinates stay valid after gradient updates.
        coords = self.point_coords.clamp(0.0, 1.0)          # (Q, 2)

        # SAM's PositionEmbeddingRandom.forward_with_coords expects
        # coords in pixel space relative to image_size.  When accessed via
        # pe_layer.forward_with_coords the caller must supply image_size.
        # We instead use the raw sinusoidal call that many SAM variants expose.
        # Fall back to a simple Linear projection when the PE layer does not
        # have the expected API (e.g. timm fallback path).
        pe_layer = prompt_encoder.pe_layer if hasattr(prompt_encoder, "pe_layer") else None

        if pe_layer is not None and hasattr(pe_layer, "forward_with_coords"):
            # Standard segment_anything PromptEncoder path.
            # forward_with_coords: (coords, image_size) → (N, C)
            image_size = getattr(prompt_encoder, "input_image_size", (1024, 1024))
            # Scale [0,1] → pixel coords for SAM's encoder.
            pixel_coords = coords * torch.tensor(
                [image_size[1], image_size[0]],
                dtype=coords.dtype,
                device=coords.device,
            )
            # pe_layer.forward_with_coords expects shape (1, N, 2)
            emb = pe_layer.forward_with_coords(
                pixel_coords.unsqueeze(0), image_size
            )  # (1, Q, C)
            sparse = emb.squeeze(0).unsqueeze(1)           # (Q, 1, C)
        else:
            # Fallback: lightweight sinusoidal embedding of (x, y).
            sparse = self._sinusoidal_embed(coords)         # (Q, 1, C)

        return sparse

    def _sinusoidal_embed(self, coords: torch.Tensor) -> torch.Tensor:
        """Simple sinusoidal embedding of 2-D coordinates → (Q, 1, embed_dim)."""
        Q = coords.shape[0]
        d = self.embed_dim
        half = d // 4                               # each axis gets d/4 sin + d/4 cos
        temperature = 10_000.0

        dim_t = torch.arange(half, dtype=coords.dtype, device=coords.device)
        dim_t = temperature ** (dim_t / half)       # (half,)

        x = coords[:, 0:1] / dim_t                 # (Q, half)
        y = coords[:, 1:2] / dim_t                 # (Q, half)

        emb = torch.cat(
            [x.sin(), x.cos(), y.sin(), y.cos()], dim=1
        )                                           # (Q, 4*half)

        # Pad or trim to embed_dim.
        if emb.shape[1] < d:
            pad = torch.zeros(Q, d - emb.shape[1], device=coords.device, dtype=coords.dtype)
            emb = torch.cat([emb, pad], dim=1)
        else:
            emb = emb[:, :d]

        return emb.unsqueeze(1)                     # (Q, 1, embed_dim)
