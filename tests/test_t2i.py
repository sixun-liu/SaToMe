import torch
import torch.nn.functional as F

from satome.config import I2TConfig, T2IConfig
from satome.salience import top_fraction_mask
from satome.t2i import build_cell_merge_plan


def test_reconstruction_plan_keeps_one_token_per_direction() -> None:
    hidden = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]]
    )
    plan = build_cell_merge_plan(
        hidden,
        protection_features=hidden,
        grid_height=2,
        grid_width=2,
        protect_ratio=0.0,
    )
    assert plan.keep_offsets[0, 0].tolist() == [0, 2]
    assert plan.drop_offsets[0, 0].tolist() == [1, 3]
    assert plan.owner_slots[0, 0].tolist() == [0, 1]
    torch.testing.assert_close(plan.reconstruction_error, torch.zeros_like(plan.reconstruction_error))
    assert not plan.protected.any()


def test_protect_ratio_one_marks_every_cell() -> None:
    hidden = torch.randn(1, 16, 8)
    plan = build_cell_merge_plan(hidden, hidden, 4, 4, protect_ratio=1.0)
    assert plan.protected.shape == (1, 4)
    assert plan.protected.all()


def test_unknown_protection_score_is_rejected() -> None:
    hidden = torch.randn(1, 4, 8)
    try:
        build_cell_merge_plan(
            hidden,
            hidden,
            2,
            2,
            protect_ratio=0.0,
            protection_score="unknown",  # type: ignore[arg-type]
        )
    except ValueError as exc:
        assert "protection_score" in str(exc)
    else:
        raise AssertionError("unknown protection score was accepted")


def test_default_retention_contract() -> None:
    assert T2IConfig().visual_retention == 0.675
    assert I2TConfig(retention=0.25).merge_ratio == 0.75


def test_default_4096_token_grid_protects_358_cells() -> None:
    scores = torch.arange(1024, dtype=torch.float32).reshape(1, 32, 32)
    protected = top_fraction_mask(scores, T2IConfig().protect_ratio)
    assert protected.sum().item() == 358
    retained_tokens = protected.sum().item() * 4 + (1024 - protected.sum().item()) * 2
    assert retained_tokens == 2764


def test_plan_matches_reference_equations() -> None:
    torch.manual_seed(42)
    hidden = torch.randn(1, 64, 32)
    temporal_delta = torch.randn(1, 64, 7)
    plan = build_cell_merge_plan(
        hidden,
        temporal_delta,
        8,
        8,
        protect_ratio=0.35,
        protection_score="isolation",
    )

    cell_local = (
        torch.arange(64).view(4, 2, 4, 2).permute(0, 2, 1, 3).reshape(16, 4)
    )
    owner_features = F.normalize(hidden[0, cell_local].float(), dim=-1)
    protect_features = F.normalize(temporal_delta[0, cell_local].float(), dim=-1)
    pair_options = torch.tensor(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    drop_options = torch.tensor(((2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1)))

    similarities = torch.einsum(
        "npdc,npkc->npdk",
        owner_features[:, drop_options],
        owner_features[:, pair_options],
    )
    nearest, owner_options = similarities.max(dim=-1)
    errors = 1.0 - nearest.mean(dim=-1)
    best = errors.argmin(dim=-1)
    expected_owners = owner_options[torch.arange(16), best]

    pairwise = protect_features @ protect_features.transpose(-1, -2)
    isolation = 1.0 - ((pairwise.sum(dim=-1) - 1.0) / 3.0).min(dim=-1).values
    expected_protected = torch.zeros(16, dtype=torch.bool)
    expected_protected[isolation.topk(round(16 * 0.35)).indices] = True

    assert torch.equal(plan.keep_offsets[0], pair_options[best])
    assert torch.equal(plan.drop_offsets[0], drop_options[best])
    assert torch.equal(plan.owner_slots[0], expected_owners)
    assert torch.equal(plan.protected[0], expected_protected)
