import pytest
import torch

from satome.salience import cell_pairwise_salience, local_salience_1d, local_salience_2d, top_fraction_mask


def test_uniform_sequence_has_zero_salience() -> None:
    hidden = torch.ones(2, 9, 4)
    scores = local_salience_1d(hidden, kernel_size=3)
    torch.testing.assert_close(scores, torch.zeros_like(scores), atol=1e-6, rtol=0)


def test_1d_outlier_is_more_salient_than_uniform_neighbors() -> None:
    hidden = torch.zeros(1, 9, 2)
    hidden[..., 0] = 1.0
    hidden[0, 4] = torch.tensor([0.0, 1.0])
    scores = local_salience_1d(hidden, kernel_size=3)
    assert scores[0, 4] > scores[0, 0]


def test_2d_salience_preserves_grid_shape() -> None:
    hidden = torch.ones(1, 16, 3)
    scores = local_salience_2d(hidden, grid_height=4, grid_width=4, kernel_size=3)
    assert scores.shape == (1, 16)
    torch.testing.assert_close(scores, torch.zeros_like(scores), atol=1e-6, rtol=0)


def test_cell_score_detects_directional_disagreement() -> None:
    uniform = torch.tensor([1.0, 0.0]).repeat(4, 1)
    mixed = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    features = torch.stack((uniform, mixed)).reshape(1, 8, 2)
    scores = cell_pairwise_salience(features, grid_height=2, grid_width=4)
    assert scores.shape == (1, 1, 2)
    assert scores[0, 0, 1] > scores[0, 0, 0]


def test_top_fraction_mask_is_deterministic_on_ties() -> None:
    scores = torch.tensor([[3.0, 2.0, 2.0, 1.0]])
    mask = top_fraction_mask(scores, 0.5)
    assert mask.tolist() == [[True, True, False, False]]


def test_even_kernel_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive odd"):
        local_salience_1d(torch.ones(1, 8, 2), kernel_size=2)
