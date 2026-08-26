"""Backbone-independent primitives for SaToMe."""

from .config import I2TConfig, T2IConfig
from .salience import cell_pairwise_salience, local_salience_1d, local_salience_2d, top_fraction_mask
from .t2i import CellMergePlan, build_cell_merge_plan

__all__ = [
    "CellMergePlan",
    "I2TConfig",
    "T2IConfig",
    "build_cell_merge_plan",
    "cell_pairwise_salience",
    "local_salience_1d",
    "local_salience_2d",
    "top_fraction_mask",
]
