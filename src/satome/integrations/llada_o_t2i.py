"""Runtime support for the minimal LLaDA-o text-to-image integration."""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from typing import Any

import torch
from torch import Tensor

from ..config import T2IConfig
from ..t2i import CellMergePlan, build_cell_merge_plan


@dataclass(frozen=True)
class VelocityMapping:
    """Mapping from the packed visual sequence back to the full latent grid."""

    original_tokens: int
    kept_local: Tensor
    dropped_local: Tensor
    owner_kept_positions: Tensor
    owner_local: Tensor


@dataclass(frozen=True)
class PackedT2IInputs:
    """LLaDA-o query tensors after visual-token pruning."""

    sequence: Tensor
    vae_token_indexes: Tensor
    text_indexes: Tensor
    position_ids: Tensor | None
    query_indexes: Tensor | None
    sequence_lengths: Tensor
    cfg_text_position_ids: Tensor | None
    cfg_text_query_indexes: Tensor | None
    cfg_image_position_ids: Tensor | None
    cfg_image_query_indexes: Tensor | None


@dataclass(frozen=True)
class T2IStepStats:
    step: int
    signal: str
    protected_cells: int
    total_cells: int
    kept_tokens: int
    total_tokens: int


def _index_on(tensor: Tensor, indexes: Tensor) -> Tensor:
    return tensor[indexes.to(tensor.device)]


def _filter_optional(values: Tensor | None, keep_mask: Tensor, original_length: int) -> Tensor | None:
    if values is None or values.shape[0] != original_length:
        return values
    return values[keep_mask.to(values.device)]


def _contiguous_optional(values: Tensor | None, new_length: int, original_length: int) -> Tensor | None:
    if values is None or values.shape[0] != original_length:
        return values
    start = int(values[0].item())
    return torch.arange(start, start + new_length, dtype=values.dtype, device=values.device)


class LLaDAOT2IRuntime:
    """Stateful SaToMe controller attached to one LLaDA-o model instance."""

    def __init__(self, config: T2IConfig | None = None) -> None:
        self.config = config or T2IConfig()
        self.step = -1
        self.previous_latent: Tensor | None = None
        self.temporal_delta: Tensor | None = None
        self.fixed_temporal_mask: Tensor | None = None
        self.velocity_mapping: VelocityMapping | None = None
        self.history: list[T2IStepStats] = []

    @property
    def active(self) -> bool:
        return self.config.window_start <= self.step < self.config.window_end

    def reset(self, initial_latent: Tensor) -> None:
        self.step = -1
        self.previous_latent = initial_latent.detach().clone()
        self.temporal_delta = None
        self.fixed_temporal_mask = None
        self.velocity_mapping = None
        self.history.clear()

    def begin_step(self, step: int) -> None:
        self.step = int(step)
        self.velocity_mapping = None

    def observe_updated_latent(self, latent: Tensor) -> None:
        if self.previous_latent is not None:
            self.temporal_delta = (
                latent.detach().float()
                - self.previous_latent.to(device=latent.device, dtype=torch.float32)
            )
        self.previous_latent = latent.detach().clone()

    def _plan(self, hidden: Tensor, grid_height: int, grid_width: int) -> tuple[CellMergePlan, str]:
        if self.step < self.config.temporal_protection_end:
            signal = "temporal_latent_delta"
            if self.temporal_delta is None:
                protection_features = hidden
                force_protected = torch.ones(
                    (1, grid_height * grid_width // 4),
                    dtype=torch.bool,
                    device=hidden.device,
                )
            else:
                protection_features = self.temporal_delta.reshape(1, -1, self.temporal_delta.shape[-1])
                protection_features = protection_features.to(device=hidden.device, dtype=torch.float32)
                force_protected = self.fixed_temporal_mask
        else:
            signal = "current_llm_input"
            protection_features = hidden
            force_protected = None

        plan = build_cell_merge_plan(
            hidden,
            protection_features,
            grid_height,
            grid_width,
            self.config.protect_ratio,
            protection_score="isolation",
        )
        if signal == "temporal_latent_delta":
            if force_protected is None:
                self.fixed_temporal_mask = plan.protected.detach().clone()
            else:
                plan = CellMergePlan(
                    protected=force_protected.to(plan.protected.device),
                    keep_offsets=plan.keep_offsets,
                    drop_offsets=plan.drop_offsets,
                    owner_slots=plan.owner_slots,
                    reconstruction_error=plan.reconstruction_error,
                    protection_score=plan.protection_score,
                )
        return plan, signal

    def pack(
        self,
        *,
        sequence: Tensor,
        vae_token_indexes: Tensor,
        text_indexes: Tensor,
        position_ids: Tensor | None,
        query_indexes: Tensor | None,
        sequence_lengths: Tensor,
        cfg_text_position_ids: Tensor | None,
        cfg_text_query_indexes: Tensor | None,
        cfg_image_position_ids: Tensor | None,
        cfg_image_query_indexes: Tensor | None,
        has_cls_token: bool,
    ) -> PackedT2IInputs:
        """Prune dropped visual tokens and rewrite LLaDA-o query metadata."""

        if not self.active:
            return PackedT2IInputs(
                sequence,
                vae_token_indexes,
                text_indexes,
                position_ids,
                query_indexes,
                sequence_lengths,
                cfg_text_position_ids,
                cfg_text_query_indexes,
                cfg_image_position_ids,
                cfg_image_query_indexes,
            )

        cls_index = vae_token_indexes[:1] if has_cls_token else None
        grid_indexes = vae_token_indexes[1:] if has_cls_token else vae_token_indexes
        token_count = int(grid_indexes.numel())
        side = isqrt(token_count)
        if side * side != token_count or side % 2:
            raise ValueError(
                "SaToMe T2I expects one square, even-sided visual-token grid; "
                f"got {token_count} tokens"
            )

        grid_hidden = _index_on(sequence, grid_indexes).unsqueeze(0)
        plan, signal = self._plan(grid_hidden, side, side)
        device = grid_hidden.device
        cell_count = token_count // 4
        local_grid = (
            torch.arange(token_count, device=device)
            .view(side // 2, 2, side // 2, 2)
            .permute(0, 2, 1, 3)
            .reshape(cell_count, 4)
        )
        protected = plan.protected[0]
        keep_pairs = local_grid.gather(1, plan.keep_offsets[0])
        drop_pairs = local_grid.gather(1, plan.drop_offsets[0])
        owners = keep_pairs.gather(1, plan.owner_slots[0])
        merge_cells = (~protected).nonzero(as_tuple=False).flatten()

        protected_local = local_grid[protected].reshape(-1)
        compressed_local = keep_pairs[merge_cells].reshape(-1)
        kept_local = torch.cat((protected_local, compressed_local)).sort().values
        dropped_local = drop_pairs[merge_cells].reshape(-1)
        owner_local = owners[merge_cells].reshape(-1)

        kept_lookup = torch.empty(token_count, dtype=torch.long, device=device)
        kept_lookup[kept_local] = torch.arange(kept_local.numel(), device=device)
        owner_kept_positions = kept_lookup[owner_local]
        self.velocity_mapping = VelocityMapping(
            original_tokens=token_count,
            kept_local=kept_local.detach(),
            dropped_local=dropped_local.detach(),
            owner_kept_positions=owner_kept_positions.detach(),
            owner_local=owner_local.detach(),
        )

        original_length = int(sequence.shape[0])
        dropped_sequence_indexes = _index_on(grid_indexes, dropped_local)
        kept_sequence_indexes = _index_on(grid_indexes, kept_local)
        keep_mask = torch.ones(original_length, dtype=torch.bool, device=sequence.device)
        keep_mask[dropped_sequence_indexes.to(sequence.device)] = False
        new_position = torch.cumsum(keep_mask.to(torch.long), dim=0) - 1
        packed_sequence = sequence[keep_mask]

        packed_text_indexes = _index_on(new_position, text_indexes)
        packed_grid_indexes = _index_on(new_position, kept_sequence_indexes)
        if cls_index is not None:
            packed_cls_index = _index_on(new_position, cls_index)
            packed_vae_indexes = torch.cat((packed_cls_index, packed_grid_indexes))
        else:
            packed_vae_indexes = packed_grid_indexes

        new_length = int(packed_sequence.shape[0])
        packed_lengths = sequence_lengths.clone()
        packed_lengths[0] -= dropped_local.numel()
        self.history.append(
            T2IStepStats(
                step=self.step,
                signal=signal,
                protected_cells=int(protected.sum().item()),
                total_cells=cell_count,
                kept_tokens=int(kept_local.numel()),
                total_tokens=token_count,
            )
        )

        return PackedT2IInputs(
            sequence=packed_sequence,
            vae_token_indexes=packed_vae_indexes,
            text_indexes=packed_text_indexes,
            position_ids=_filter_optional(position_ids, keep_mask, original_length),
            query_indexes=_contiguous_optional(query_indexes, new_length, original_length),
            sequence_lengths=packed_lengths,
            cfg_text_position_ids=_filter_optional(
                cfg_text_position_ids, keep_mask, original_length
            ),
            cfg_text_query_indexes=_contiguous_optional(
                cfg_text_query_indexes, new_length, original_length
            ),
            cfg_image_position_ids=_filter_optional(
                cfg_image_position_ids, keep_mask, original_length
            ),
            cfg_image_query_indexes=_contiguous_optional(
                cfg_image_query_indexes, new_length, original_length
            ),
        )

    def unmerge_velocity(self, velocity: Tensor) -> Tensor:
        mapping = self.velocity_mapping
        if mapping is None:
            return velocity
        full = velocity.new_zeros((mapping.original_tokens, velocity.shape[-1]))
        kept = mapping.kept_local.to(velocity.device)
        dropped = mapping.dropped_local.to(velocity.device)
        owner_positions = mapping.owner_kept_positions.to(velocity.device)
        full[kept] = velocity
        full[dropped] = velocity[owner_positions]
        return full

    def restore_update_base(self, latent: Tensor) -> Tensor:
        mapping = self.velocity_mapping
        if mapping is None or self.config.alpha >= 1.0:
            return latent
        dropped = mapping.dropped_local.to(latent.device)
        owners = mapping.owner_local.to(latent.device)
        restored = latent.clone()
        restored[dropped] = (
            self.config.alpha * latent[dropped].float()
            + (1.0 - self.config.alpha) * latent[owners].float()
        ).to(latent.dtype)
        return restored


def enable_llada_o_t2i(model: Any, config: T2IConfig | None = None) -> LLaDAOT2IRuntime:
    """Attach SaToMe to a patched LLaDA-o model and return its runtime state."""

    runtime = LLaDAOT2IRuntime(config)
    model._satome_t2i_runtime = runtime
    return runtime
