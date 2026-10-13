"""CPU-only contract tests for the LLaDA-o T2I hook boundary.

These tests deliberately use a tiny deterministic fake backbone. They exercise
the same data-flow boundary as the upstream patch without loading a checkpoint,
VAE, CUDA kernels, or the transformer implementation.
"""

from __future__ import annotations

import pytest
import torch

from satome.config import T2IConfig
from satome.integrations.llada_o_t2i import LLaDAOT2IRuntime


def _visual_embedding(latent: torch.Tensor) -> torch.Tensor:
    """Small deterministic stand-in for VAE-to-LLM projection plus features."""

    return torch.cat((latent, latent.square()), dim=-1)


@pytest.mark.parametrize("has_cls_token", [False, True])
def test_fake_generation_loop_preserves_hook_contract(has_cls_token: bool) -> None:
    torch.manual_seed(123)
    token_count = 64
    latent_width = 4
    text_count = 3
    cls_count = int(has_cls_token)
    visual_start = text_count + cls_count
    sequence_length = visual_start + token_count
    visual_indexes = torch.arange(visual_start, sequence_length)
    vae_indexes = visual_indexes
    if has_cls_token:
        vae_indexes = torch.cat((torch.tensor([text_count]), visual_indexes))
    text_indexes = torch.arange(text_count)

    config = T2IConfig(
        total_steps=50,
        window_start=16,
        temporal_protection_end=25,
        window_end=45,
        protect_ratio=0.35,
        alpha=0.95,
    )
    runtime = LLaDAOT2IRuntime(config)
    latent = torch.randn(token_count, latent_width)
    runtime.reset(latent)

    inactive_identity_seen = False
    active_lengths: list[int] = []
    for step in range(49):
        runtime.begin_step(step)
        sequence = torch.zeros(sequence_length, latent_width * 2)
        sequence[text_indexes] = torch.arange(text_count, dtype=sequence.dtype).unsqueeze(-1)
        if has_cls_token:
            sequence[text_count] = -1.0
        sequence[visual_indexes] = _visual_embedding(latent)

        position_ids = torch.arange(sequence_length)
        query_indexes = torch.arange(100, 100 + sequence_length)
        packed = runtime.pack(
            sequence=sequence,
            vae_token_indexes=vae_indexes,
            text_indexes=text_indexes,
            position_ids=position_ids,
            query_indexes=query_indexes,
            sequence_lengths=torch.tensor([sequence_length], dtype=torch.int32),
            cfg_text_position_ids=position_ids.clone(),
            cfg_text_query_indexes=query_indexes.clone(),
            cfg_image_position_ids=position_ids.clone(),
            cfg_image_query_indexes=query_indexes.clone(),
            has_cls_token=has_cls_token,
        )

        if config.window_start <= step < config.window_end:
            active_lengths.append(int(packed.sequence.shape[0]))
            assert packed.sequence.shape[0] == sequence_length - 20
            assert packed.vae_token_indexes.numel() == 44 + cls_count
            assert packed.sequence_lengths.tolist() == [sequence_length - 20]
            assert packed.query_indexes.tolist() == list(
                range(100, 100 + sequence_length - 20)
            )

            packed_visual_indexes = (
                packed.vae_token_indexes[1:]
                if has_cls_token
                else packed.vae_token_indexes
            )
            fake_main_velocity = packed.sequence[packed_visual_indexes, :latent_width]
            fake_text_velocity = fake_main_velocity + 1.0
            fake_image_velocity = fake_main_velocity - 1.0
            full_main_velocity = runtime.unmerge_velocity(fake_main_velocity)
            full_text_velocity = runtime.unmerge_velocity(fake_text_velocity)
            full_image_velocity = runtime.unmerge_velocity(fake_image_velocity)
            assert full_main_velocity.shape == (token_count, latent_width)
            assert full_text_velocity.shape == (token_count, latent_width)
            assert full_image_velocity.shape == (token_count, latent_width)
            assert torch.isfinite(full_main_velocity).all()

            mapping = runtime.velocity_mapping
            assert mapping is not None
            torch.testing.assert_close(
                full_main_velocity[mapping.dropped_local],
                fake_main_velocity[mapping.owner_kept_positions],
            )
            restored = runtime.restore_update_base(latent)
            expected_restored = latent.clone()
            expected_restored[mapping.dropped_local] = (
                config.alpha * latent[mapping.dropped_local]
                + (1.0 - config.alpha) * latent[mapping.owner_local]
            )
            torch.testing.assert_close(restored, expected_restored)
            latent = restored - full_main_velocity * 0.01
        else:
            assert packed.sequence is sequence
            assert packed.vae_token_indexes is vae_indexes
            inactive_identity_seen = True
            latent = latent - 0.01

        runtime.observe_updated_latent(latent)

    assert inactive_identity_seen
    assert active_lengths == [sequence_length - 20] * 29
    assert [row.step for row in runtime.history] == list(range(16, 45))
    assert {row.protected_cells for row in runtime.history} == {6}
    assert {row.kept_tokens for row in runtime.history} == {44}


def test_fake_hook_rejects_non_square_visual_grid() -> None:
    runtime = LLaDAOT2IRuntime(
        T2IConfig(total_steps=3, window_start=1, temporal_protection_end=2, window_end=2)
    )
    runtime.reset(torch.randn(12, 4))
    runtime.begin_step(1)
    with pytest.raises(ValueError, match="square, even-sided"):
        runtime.pack(
            sequence=torch.randn(14, 8),
            vae_token_indexes=torch.arange(2, 14),
            text_indexes=torch.arange(2),
            position_ids=None,
            query_indexes=None,
            sequence_lengths=torch.tensor([14]),
            cfg_text_position_ids=None,
            cfg_text_query_indexes=None,
            cfg_image_position_ids=None,
            cfg_image_query_indexes=None,
            has_cls_token=False,
        )
