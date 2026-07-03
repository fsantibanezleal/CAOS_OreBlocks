"""Regular 3-D block grid with MineLib's level convention.

The vertical index is the LEVEL and increases UPWARD: level 0 is the deepest bench, level
``nz - 1`` is the surface bench. This is the convention published MineLib instances use (verified
against newman1: a block's predecessors sit one level ABOVE it), so everything oreblocks emits is
directly comparable with the published library. Viewers that draw depth-down (z=0 at the surface)
simply flip: ``z_down = (nz - 1) - level``.

Flat indexing is ``index = (level * ny + iy) * nx + ix`` — x fastest, then y, then level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BlockGrid"]


@dataclass(frozen=True)
class BlockGrid:
    """Grid dimensions + block size (metres). ``nz`` counts benches (levels, up)."""

    nx: int
    ny: int
    nz: int
    dx: float = 10.0
    dy: float = 10.0
    dz: float = 10.0

    def __post_init__(self) -> None:
        if min(self.nx, self.ny, self.nz) < 1:
            raise ValueError(f"grid dims must be >= 1, got {self.nx}x{self.ny}x{self.nz}")
        if min(self.dx, self.dy, self.dz) <= 0:
            raise ValueError("block size must be positive")

    @property
    def n_blocks(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def surface_level(self) -> int:
        return self.nz - 1

    @property
    def block_volume(self) -> float:
        return self.dx * self.dy * self.dz

    def index(self, ix: int, iy: int, level: int) -> int:
        """Flat index of the block at (ix, iy, level). Bounds-checked."""
        if not (0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= level < self.nz):
            raise IndexError(f"({ix},{iy},{level}) outside {self.nx}x{self.ny}x{self.nz}")
        return (level * self.ny + iy) * self.nx + ix

    def coords(self, i: int) -> tuple[int, int, int]:
        """(ix, iy, level) of flat index ``i``."""
        if not (0 <= i < self.n_blocks):
            raise IndexError(f"flat index {i} outside 0..{self.n_blocks - 1}")
        level, rem = divmod(i, self.nx * self.ny)
        iy, ix = divmod(rem, self.nx)
        return ix, iy, level

    def coord_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x, y, level) int32 arrays for every block, in flat order — the MineLib id order."""
        ii = np.arange(self.n_blocks, dtype=np.int64)
        level, rem = np.divmod(ii, self.nx * self.ny)
        iy, ix = np.divmod(rem, self.nx)
        return ix.astype(np.int32), iy.astype(np.int32), level.astype(np.int32)

    def level_slice(self, level: int) -> slice:
        """Flat-index slice covering one whole level."""
        if not (0 <= level < self.nz):
            raise IndexError(f"level {level} outside 0..{self.nz - 1}")
        per = self.nx * self.ny
        return slice(level * per, (level + 1) * per)
