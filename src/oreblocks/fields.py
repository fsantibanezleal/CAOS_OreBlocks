"""Seeded synthetic deposits — geostatistically plausible grade fields on a block grid.

Clearly SYNTHETIC (no real drillholes), but built the way real deposits are described: a
deterministic grade trend (the geological shape) plus spatially-correlated noise (a box-smoothed
white field standing in for a variogram range), so downstream optimisers face non-trivial
ore/waste and slope trade-offs. Everything is seeded — byte-identical given (archetype, dims,
seed). The four archetypes mirror the CAOS PitForge teaching set:

- ``porphyry``  — a buried ellipsoidal high-grade shell (broad bowl pits)
- ``vein``      — a dipping tabular zone (narrow steep pits)
- ``layered``   — horizontal stratabound bands
- ``core_halo`` — a rich core inside a broad low-grade halo

Levels increase UPWARD (grid convention); the trend functions are written in depth fractions so
the shapes match their depth-down originals exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._rng import stream
from .grid import BlockGrid

__all__ = ["ARCHETYPES", "Deposit", "make_deposit"]

ARCHETYPES = ("porphyry", "vein", "layered", "core_halo")


@dataclass(frozen=True)
class Deposit:
    """A block model: grid + per-block grade (mass fraction), tonnage (t) and density (t/m3)."""

    grid: BlockGrid
    grade: np.ndarray
    tonnage: np.ndarray
    density: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.grid.n_blocks
        for name in ("grade", "tonnage", "density"):
            a = getattr(self, name)
            if a.shape != (n,):
                raise ValueError(f"{name} must be a flat array of {n} blocks, got {a.shape}")


def _smooth(vol: np.ndarray, passes: int) -> np.ndarray:
    """A few 3x3x3 box-blur passes (edge-corrected) give white noise a correlation length."""
    out = vol.astype(np.float64, copy=True)
    ones = np.ones_like(out)
    for _ in range(passes):
        acc = np.zeros_like(out)
        cnt = np.zeros_like(out)
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    src = out[
                        max(0, -dz) : out.shape[0] - max(0, dz),
                        max(0, -dy) : out.shape[1] - max(0, dy),
                        max(0, -dx) : out.shape[2] - max(0, dx),
                    ]
                    csrc = ones[
                        max(0, -dz) : out.shape[0] - max(0, dz),
                        max(0, -dy) : out.shape[1] - max(0, dy),
                        max(0, -dx) : out.shape[2] - max(0, dx),
                    ]
                    acc[
                        max(0, dz) : out.shape[0] - max(0, -dz),
                        max(0, dy) : out.shape[1] - max(0, -dy),
                        max(0, dx) : out.shape[2] - max(0, -dx),
                    ] += src
                    cnt[
                        max(0, dz) : out.shape[0] - max(0, -dz),
                        max(0, dy) : out.shape[1] - max(0, -dy),
                        max(0, dx) : out.shape[2] - max(0, -dx),
                    ] += csrc
        out = acc / cnt
    return out


def _trend(archetype: str, fx: np.ndarray, fy: np.ndarray, fdepth: np.ndarray) -> np.ndarray:
    """Relative grade shape in ~[0,1]; ``fdepth`` = 0 at the surface, 1 at the deepest level."""
    cx = fx - 0.5
    cy = fy - 0.5
    if archetype == "porphyry":
        r = np.sqrt(cx * cx + cy * cy + (fdepth - 0.45) ** 2)
        return np.maximum(0.0, 1.0 - np.abs(r - 0.22) / 0.28)
    if archetype == "vein":
        plane = cx * 0.8 + (fdepth - 0.5) * 0.6
        return np.maximum(0.0, 1.0 - np.abs(plane) / 0.12)
    if archetype == "layered":
        return 0.5 + 0.5 * np.cos(fdepth * np.pi * 4)
    if archetype == "core_halo":
        r = np.sqrt(cx * cx + cy * cy + (fdepth - 0.5) ** 2)
        return np.maximum(0.0, 1.0 - r / 0.45) ** 1.6
    raise ValueError(f"unknown archetype {archetype!r}; expected one of {ARCHETYPES}")


def make_deposit(
    grid: BlockGrid,
    archetype: str,
    seed: int = 1,
    *,
    peak_grade: float = 0.02,
    background: float = 0.001,
    density: float = 2.7,
    noise: float = 0.35,
    name: str | None = None,
) -> Deposit:
    """Build a seeded synthetic deposit on ``grid``. Grades are mass fractions in [0, 1]."""
    if archetype not in ARCHETYPES:
        raise ValueError(f"unknown archetype {archetype!r}; expected one of {ARCHETYPES}")
    rng = stream(seed, f"deposit:{archetype}")
    white = rng.random((grid.nz, grid.ny, grid.nx)) - 0.5
    corr = _smooth(white, passes=3)

    level = np.arange(grid.nz, dtype=np.float64)
    fdepth_1d = 1.0 - (level / (grid.nz - 1) if grid.nz > 1 else np.full(1, 0.5))
    fx_1d = np.arange(grid.nx, dtype=np.float64) / (grid.nx - 1) if grid.nx > 1 else np.full(1, 0.5)
    fy_1d = np.arange(grid.ny, dtype=np.float64) / (grid.ny - 1) if grid.ny > 1 else np.full(1, 0.5)
    fdepth, fy, fx = np.meshgrid(fdepth_1d, fy_1d, fx_1d, indexing="ij")

    shape = _trend(archetype, fx, fy, fdepth)
    grade = background + (peak_grade - background) * np.maximum(0.0, shape + noise * corr)
    grade = np.clip(grade, 0.0, 1.0).reshape(-1)

    block_tonnes = grid.block_volume * density
    return Deposit(
        grid=grid,
        grade=grade,
        tonnage=np.full(grid.n_blocks, block_tonnes),
        density=np.full(grid.n_blocks, density),
        meta={
            "archetype": archetype,
            "seed": seed,
            "peak_grade": peak_grade,
            "background": background,
            "noise": noise,
            "grade_unit": "mass fraction",
            "name": name or archetype,
            "synthetic": True,
        },
    )
