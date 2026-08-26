"""Backbone-independent 2x2 planning used by the T2I integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from .salience import (
    _to_cells,
    cell_isolation_salience,
    cell_pairwise_salience,
    top_fraction_mask,
)


_PAIR_OPTIONS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_DROP_OPTIONS = ((2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1))


@dataclass(frozen=True)
class CellMergePlan:
    """A fixed-size plan; integrations decide how to pack protected cells."""

    protected: Tensor
    keep_offsets: Tensor
    drop_offsets: Tensor
    owner_slots: Tensor
    reconstruction_error: Tensor
    protection_score: Tensor


def build_cell_merge_plan(
    hidden: Tensor,
    protection_features: Tensor,
    grid_height: int,
    grid_width: int,
    protect_ratio: float,
    protection_score: Literal["isolation", "mean_pairwise"] = "isolation",
) -> CellMergePlan:
    """Build the paper's protection and reconstruction-minimizing 4-to-2 plan.

    ``hidden`` determines the six-way keep/drop split and nearest destination.
    ``protection_features`` independently determines which cells remain 4-to-4.
    Keeping these inputs separate makes the temporal protection signal explicit.
    """

    if hidden.shape[:2] != protection_features.shape[:2]:
        raise ValueError(
            "hidden and protection_features must share batch/token dimensions, "
            f"got {tuple(hidden.shape)} and {tuple(protection_features.shape)}"
        )

    cells = F.normalize(_to_cells(hidden, grid_height, grid_width).float(), dim=-1)
    device = cells.device
    pair_options = torch.tensor(_PAIR_OPTIONS, device=device, dtype=torch.long)
    drop_options = torch.tensor(_DROP_OPTIONS, device=device, dtype=torch.long)

    kept = cells[:, :, pair_options, :]
    dropped = cells[:, :, drop_options, :]
    source_to_destination = torch.einsum("bcqsd,bcqkd->bcqsk", dropped, kept)
    nearest_cosine, owner_options = source_to_destination.max(dim=-1)
    errors = 1.0 - nearest_cosine.mean(dim=-1)
    best_option = errors.argmin(dim=-1)

    keep_offsets = pair_options[best_option]
    drop_offsets = drop_options[best_option]
    selected_owner = owner_options.gather(
        2,
        best_option.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2),
    ).squeeze(2)
    selected_error = errors.gather(2, best_option.unsqueeze(-1)).squeeze(-1)

    if protection_score == "isolation":
        protection_score_grid = cell_isolation_salience(
            protection_features,
            grid_height,
            grid_width,
        )
    elif protection_score == "mean_pairwise":
        protection_score_grid = cell_pairwise_salience(
            protection_features,
            grid_height,
            grid_width,
        )
    else:
        raise ValueError(
            "protection_score must be 'isolation' or 'mean_pairwise', "
            f"got {protection_score!r}"
        )
    protected_grid = top_fraction_mask(protection_score_grid, protect_ratio)

    return CellMergePlan(
        protected=protected_grid.reshape(hidden.shape[0], -1),
        keep_offsets=keep_offsets,
        drop_offsets=drop_offsets,
        owner_slots=selected_owner,
        reconstruction_error=selected_error,
        protection_score=protection_score_grid.reshape(hidden.shape[0], -1),
    )
