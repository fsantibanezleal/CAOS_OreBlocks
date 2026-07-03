"""Block economics — the UPIT value model with per-block optimal destination.

A block's net value is the better of its two destinations, exactly the semantics of a MineLib
``.upit`` column (verified against newman1, where the published net value equals
max(value-if-wasted, value-if-processed)):

- waste:   ``-mining_cost * tonnage``
- ore:     ``(grade * recovery * price - processing_cost) * tonnage - mining_cost * tonnage``

The floating cutoff falls out of the max: a block is ore iff processing it beats wasting it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fields import Deposit

__all__ = ["Econ", "block_values", "is_ore", "cutoff_grade"]


@dataclass(frozen=True)
class Econ:
    """price: $/t of recovered metal · recovery: 0..1 · costs: $/t mined / $/t milled."""

    price: float = 9000.0
    recovery: float = 0.88
    mining_cost: float = 2.5
    processing_cost: float = 9.0

    def __post_init__(self) -> None:
        if not (0 < self.recovery <= 1):
            raise ValueError(f"recovery must be in (0,1], got {self.recovery}")
        if self.price <= 0 or self.mining_cost < 0 or self.processing_cost < 0:
            raise ValueError("price must be positive; costs non-negative")


def cutoff_grade(econ: Econ) -> float:
    """The grade above which processing beats wasting: processing_cost / (recovery * price)."""
    return econ.processing_cost / (econ.recovery * econ.price)


def block_values(dep: Deposit, econ: Econ) -> np.ndarray:
    """Net value per block (float64) at the optimal destination — the ``.upit`` column."""
    waste = -econ.mining_cost * dep.tonnage
    ore = (dep.grade * econ.recovery * econ.price - econ.processing_cost) * dep.tonnage + waste
    return np.maximum(waste, ore)


def is_ore(dep: Deposit, econ: Econ) -> np.ndarray:
    """Boolean per block: does processing beat wasting?"""
    return dep.grade > cutoff_grade(econ)
