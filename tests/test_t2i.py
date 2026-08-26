import torch

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


def test_paper_default_retention_contract() -> None:
    assert T2IConfig().visual_retention == 0.675
    assert I2TConfig(retention=0.25).merge_ratio == 0.75


def test_default_4096_token_grid_protects_358_cells() -> None:
    scores = torch.arange(1024, dtype=torch.float32).reshape(1, 32, 32)
    protected = top_fraction_mask(scores, T2IConfig().protect_ratio)
    assert protected.sum().item() == 358
    retained_tokens = protected.sum().item() * 4 + (1024 - protected.sum().item()) * 2
    assert retained_tokens == 2764
