import torch

from satome.config import T2IConfig
from satome.integrations.llada_o_t2i import LLaDAOT2IRuntime


def _runtime() -> LLaDAOT2IRuntime:
    return LLaDAOT2IRuntime(
        T2IConfig(
            total_steps=4,
            window_start=1,
            temporal_protection_end=2,
            window_end=3,
            protect_ratio=0.25,
            alpha=0.75,
        )
    )


def test_pack_unmerge_and_soft_restoration() -> None:
    torch.manual_seed(7)
    runtime = _runtime()
    latent = torch.randn(16, 3)
    runtime.reset(latent)
    runtime.begin_step(0)
    runtime.observe_updated_latent(latent + torch.randn_like(latent) * 0.1)
    runtime.begin_step(1)

    sequence = torch.randn(18, 8)
    packed = runtime.pack(
        sequence=sequence,
        vae_token_indexes=torch.arange(2, 18),
        text_indexes=torch.arange(2),
        position_ids=torch.arange(18),
        query_indexes=torch.arange(30, 48),
        sequence_lengths=torch.tensor([18]),
        cfg_text_position_ids=torch.arange(18),
        cfg_text_query_indexes=torch.arange(50, 68),
        cfg_image_position_ids=None,
        cfg_image_query_indexes=None,
        has_cls_token=False,
    )

    assert packed.sequence.shape == (12, 8)
    assert packed.vae_token_indexes.numel() == 10
    assert packed.sequence_lengths.tolist() == [12]
    assert packed.query_indexes.tolist() == list(range(30, 42))
    assert runtime.history[-1].protected_cells == 1

    velocity = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    full_velocity = runtime.unmerge_velocity(velocity)
    mapping = runtime.velocity_mapping
    assert mapping is not None
    assert full_velocity.shape == (16, 3)
    torch.testing.assert_close(
        full_velocity[mapping.dropped_local],
        velocity[mapping.owner_kept_positions],
    )

    update_base = runtime.restore_update_base(latent)
    expected = 0.75 * latent[mapping.dropped_local] + 0.25 * latent[mapping.owner_local]
    torch.testing.assert_close(update_base[mapping.dropped_local], expected)


def test_temporal_mask_is_frozen_only_during_early_stage() -> None:
    torch.manual_seed(11)
    runtime = _runtime()
    latent = torch.randn(16, 3)
    runtime.reset(latent)
    runtime.observe_updated_latent(latent + torch.randn_like(latent))
    runtime.begin_step(1)
    first_plan, first_signal = runtime._plan(torch.randn(1, 16, 8), 4, 4)
    runtime.temporal_delta = torch.randn_like(runtime.temporal_delta)
    second_plan, second_signal = runtime._plan(torch.randn(1, 16, 8), 4, 4)

    assert first_signal == second_signal == "temporal_vae_delta"
    assert torch.equal(first_plan.protected, second_plan.protected)

    runtime.begin_step(2)
    _, late_signal = runtime._plan(torch.randn(1, 16, 8), 4, 4)
    assert late_signal == "current_llm_input"
