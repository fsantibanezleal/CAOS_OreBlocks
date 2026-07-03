"""One-call synthetic MineLib-nature instances ("twins") with a stamped exact optimum.

``make_twin`` chains the whole pipeline — seeded deposit -> UPIT economics -> slope precedence ->
exact solve — and ``Twin.write`` emits the instance in exact MineLib format plus a meta sidecar
carrying the stamped optimum, so any UPIT solver can be validated against a known-by-construction
answer without touching licensed data. This is what CAOS PitForge consumes as its license-free
realistic mid-size instances, and what regression/scale tests generate on the fly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .economics import Econ, block_values
from .fields import Deposit, make_deposit
from .grid import BlockGrid
from .minelib_io import write_minelib
from .precedence import Precedence, build_precedence
from .upit import UpitResult, solve_upit

__all__ = ["Twin", "make_twin"]


@dataclass(frozen=True)
class Twin:
    name: str
    deposit: Deposit
    econ: Econ
    slope_deg: float
    values: np.ndarray
    precedence: Precedence
    upit: UpitResult

    def write(self, out_dir: str | Path) -> dict[str, Path]:
        """Write the MineLib triplet + meta (with the stamped exact optimum)."""
        return write_minelib(
            out_dir,
            self.name,
            self.deposit,
            self.values,
            self.precedence,
            meta_extra={
                "econ": {
                    "price": self.econ.price,
                    "recovery": self.econ.recovery,
                    "mining_cost": self.econ.mining_cost,
                    "processing_cost": self.econ.processing_cost,
                },
                "slope_deg": self.slope_deg,
                "stamped_optimum": self.upit.pit_value,
                "stamped_n_in_pit": self.upit.n_in_pit,
            },
        )


def make_twin(
    archetype: str,
    dims: tuple[int, int, int],
    seed: int = 1,
    *,
    econ: Econ | None = None,
    slope_deg: float = 45.0,
    peak_grade: float = 0.02,
    name: str | None = None,
) -> Twin:
    """Generate a full synthetic UPIT instance and solve it exactly (the stamped oracle)."""
    grid = BlockGrid(nx=dims[0], ny=dims[1], nz=dims[2])
    econ = econ or Econ()
    dep = make_deposit(grid, archetype, seed, peak_grade=peak_grade, name=name)
    values = block_values(dep, econ)
    prec = build_precedence(grid, slope_deg)
    upit = solve_upit(values, prec)
    return Twin(
        name=name or f"twin-{archetype}-{dims[0]}x{dims[1]}x{dims[2]}-s{seed}",
        deposit=dep,
        econ=econ,
        slope_deg=slope_deg,
        values=values,
        precedence=prec,
        upit=upit,
    )
