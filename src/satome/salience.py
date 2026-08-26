"""Shared local-deviation salience primitives.

All functions return larger values for tokens or cells that disagree more
strongly with their local context. Scores are computed in float32 even when
the backbone hidden states use reduced precision.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _validate_hidden(hidden: Tensor) -> tuple[int, int, int]:
    if hidden.ndim != 3:
        raise ValueError(f"expected hidden states with shape [B, N, D], got {tuple(hidden.shape)}")
    batch, tokens, width = hidden.shape
    if batch <= 0 or tokens <= 0 or width <= 0:
        raise ValueError(f"hidden-state dimensions must be positive, got {tuple(hidden.shape)}")
    return batch, tokens, width


def _validate_kernel(kernel_size: int) -> None:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")


def local_salience_1d(hidden: Tensor, kernel_size: int = 11) -> Tensor:
    """Compute ``1 - cos(h_i, AvgPool_k(h)_i)`` on a token sequence.

    The edge behavior matches the experiment implementation: valid pooling is
    used in the interior and the first/last pooled context is replicated at
    sequence boundaries.
    """

    _, tokens, _ = _validate_hidden(hidden)
    _validate_kernel(kernel_size)
    if kernel_size > tokens:
        raise ValueError(f"kernel_size={kernel_size} exceeds token count {tokens}")

    normalized = F.normalize(hidden.float(), dim=-1)
    sequence = normalized.transpose(1, 2)
    pooled = F.avg_pool1d(sequence, kernel_size=kernel_size, stride=1)
    pad = kernel_size // 2
    if pad:
        pooled = torch.cat(
            (
                pooled[:, :, :1].expand(-1, -1, pad),
                pooled,
                pooled[:, :, -1:].expand(-1, -1, pad),
            ),
            dim=-1,
        )
    context = F.normalize(pooled.transpose(1, 2), dim=-1)
    return 1.0 - (normalized * context).sum(dim=-1)


def local_salience_2d(
    hidden: Tensor,
    grid_height: int,
    grid_width: int,
    kernel_size: int = 3,
) -> Tensor:
    """Compute local-deviation salience over a true 2D visual-token grid."""

    batch, tokens, width = _validate_hidden(hidden)
    _validate_kernel(kernel_size)
    if grid_height <= 0 or grid_width <= 0 or grid_height * grid_width != tokens:
        raise ValueError(
            f"grid shape {grid_height}x{grid_width} does not match token count {tokens}"
        )

    normalized = F.normalize(hidden.float(), dim=-1)
    image = normalized.view(batch, grid_height, grid_width, width).permute(0, 3, 1, 2).contiguous()
    context = F.avg_pool2d(
        image,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        count_include_pad=False,
    )
    context = context.permute(0, 2, 3, 1).reshape(batch, tokens, width)
    context = F.normalize(context, dim=-1)
    return 1.0 - (normalized * context).sum(dim=-1)


def _to_cells(features: Tensor, grid_height: int, grid_width: int) -> Tensor:
    batch, tokens, width = _validate_hidden(features)
    if grid_height % 2 or grid_width % 2 or grid_height * grid_width != tokens:
        raise ValueError(
            "2x2 cells require an even grid whose area matches the token count; "
            f"got {grid_height}x{grid_width} for {tokens} tokens"
        )
    grid = features.view(batch, grid_height, grid_width, width)
    return (
        grid.view(batch, grid_height // 2, 2, grid_width // 2, 2, width)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, (grid_height // 2) * (grid_width // 2), 4, width)
    )


def cell_pairwise_salience(features: Tensor, grid_height: int, grid_width: int) -> Tensor:
    """Average pairwise cosine dissimilarity within every 2x2 cell.

    This implements the paper's cell-protection score and accepts any feature
    signal chosen by an integration, such as a temporal hidden-state delta.
    """

    cells = F.normalize(_to_cells(features, grid_height, grid_width).float(), dim=-1)
    pairwise = torch.matmul(cells, cells.transpose(-1, -2))
    diagonal = pairwise.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    mean_unique_pair_cosine = (pairwise.sum(dim=(-1, -2)) - diagonal) / 12.0
    scores = 1.0 - mean_unique_pair_cosine
    return scores.view(features.shape[0], grid_height // 2, grid_width // 2)


def top_fraction_mask(scores: Tensor, fraction: float) -> Tensor:
    """Select a deterministic top fraction independently for each batch item."""

    if scores.ndim < 2:
        raise ValueError(f"scores must include batch and item dimensions, got {tuple(scores.shape)}")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")

    flat = scores.reshape(scores.shape[0], -1)
    count = int(round(flat.shape[1] * fraction))
    mask = torch.zeros_like(flat, dtype=torch.bool)
    if count:
        order = torch.argsort(flat, dim=-1, descending=True, stable=True)
        mask.scatter_(1, order[:, :count], True)
    return mask.reshape_as(scores)
