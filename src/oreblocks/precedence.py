"""Slope precedence — which blocks must be removed ABOVE a block before it can be mined.

A pit wall stands at the overall slope angle θ (from horizontal). On a regular grid this is the
standard reduced pattern: arcs only to the (2rx+1)x(2ry+1) box one level ABOVE (level + 1, since
levels increase upward), with rx = round(dz / (dx tan θ)) clamped to >= 1; transitivity up the
levels reproduces the full cone exactly. θ = 45 deg with cubic blocks gives the classic 9-point
template.

The CSR layout matches MineLib's ``.prec`` semantics: for block b, ``plist[pstart[b]:pstart[b+1]]``
are the blocks that must be extracted BEFORE b.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .grid import BlockGrid

__all__ = ["Precedence", "slope_offsets", "build_precedence"]


@dataclass(frozen=True)
class Precedence:
    """CSR over predecessors: block b requires plist[pstart[b]:pstart[b+1]] mined first."""

    pstart: np.ndarray  # int64, length n_blocks + 1
    plist: np.ndarray  # int64

    @property
    def n_arcs(self) -> int:
        return int(self.plist.shape[0])

    def preds(self, b: int) -> np.ndarray:
        return self.plist[self.pstart[b] : self.pstart[b + 1]]


def slope_offsets(grid: BlockGrid, slope_deg: float) -> list[tuple[int, int]]:
    """(di, dj) horizontal offsets of the one-level-up template for the given wall angle."""
    if not (5.0 <= slope_deg <= 89.0):
        raise ValueError(f"slope angle {slope_deg} deg outside the sane range [5, 89]")
    t = math.tan(math.radians(slope_deg))
    rx = max(1, round(grid.dz / (grid.dx * t)))
    ry = max(1, round(grid.dz / (grid.dy * t)))
    return [(di, dj) for di in range(-rx, rx + 1) for dj in range(-ry, ry + 1)]


def build_precedence(grid: BlockGrid, slope_deg: float = 45.0) -> Precedence:
    """Build the CSR precedence for the whole grid (vectorised per offset)."""
    offsets = slope_offsets(grid, slope_deg)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    per = nx * ny
    n = grid.n_blocks

    ix = np.arange(nx, dtype=np.int64)
    iy = np.arange(ny, dtype=np.int64)
    gx, gy = np.meshgrid(ix, iy, indexing="xy")  # shape (ny, nx): gx[iy, ix] = ix
    gx = gx.reshape(-1)
    gy = gy.reshape(-1)

    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    for level in range(0, nz - 1):  # the top level has nothing above it
        base_src = level * per + gy * nx + gx
        for di, dj in offsets:
            jx = gx + di
            jy = gy + dj
            ok = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
            if not ok.any():
                continue
            src_parts.append(base_src[ok])
            dst_parts.append((level + 1) * per + jy[ok] * nx + jx[ok])

    if src_parts:
        src = np.concatenate(src_parts)
        dst = np.concatenate(dst_parts)
    else:  # single-level grid: no precedence at all
        src = np.empty(0, dtype=np.int64)
        dst = np.empty(0, dtype=np.int64)

    order = np.argsort(src, kind="stable")
    src = src[order]
    dst = dst[order]
    counts = np.bincount(src, minlength=n).astype(np.int64)
    pstart = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=pstart[1:])
    return Precedence(pstart=pstart, plist=dst)
