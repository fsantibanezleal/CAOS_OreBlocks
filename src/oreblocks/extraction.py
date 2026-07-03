"""Extraction state + loading faces — the bridge from a static pit to an OPERATING mine.

Given the exact pit (``in_pit``) and a mining progress fraction, benches are extracted top-down
(the only physical order): the state says which levels are fully out, which level is the active
bench, and which pit blocks remain. Loading FACES are seeded k-means clusters of the remaining
active-bench blocks — each face is a shovel position with its local mean grade, ore/waste split
and available tonnage. This is what a haulage simulator (e.g. minehaulsim) consumes to ground
truck cycles in geology: grade at face, bench elevation, ore vs waste destination.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._rng import stream
from .economics import Econ, cutoff_grade
from .fields import Deposit
from .grid import BlockGrid

__all__ = ["Face", "ExtractionState", "extraction_state", "loading_faces"]


@dataclass(frozen=True)
class Face:
    """One loading position on the active bench."""

    face_id: int
    level: int
    x: float  # centroid, block units
    y: float
    n_blocks: int
    tonnes: float
    mean_grade: float
    ore_fraction: float  # by tonnage, at the given cutoff
    elevation_m: float  # level * dz (bench floor height above the deepest level)


@dataclass(frozen=True)
class ExtractionState:
    """Benches above ``active_level`` are fully extracted (within the pit); the active bench is
    partially extracted per ``progress``. ``remaining`` marks in-pit blocks not yet mined."""

    active_level: int
    progress: float
    extracted: np.ndarray  # bool per block
    remaining: np.ndarray  # bool per block (in_pit & ~extracted)
    tonnes_extracted: float
    tonnes_remaining: float


def _pit_levels(grid: BlockGrid, in_pit: np.ndarray) -> list[int]:
    """Levels that contain pit blocks, ordered TOP-DOWN (mining order)."""
    per = grid.nx * grid.ny
    levels = [lv for lv in range(grid.nz) if in_pit[lv * per : (lv + 1) * per].any()]
    return list(reversed(levels))


def extraction_state(dep: Deposit, in_pit: np.ndarray, progress: float, seed: int = 1) -> ExtractionState:
    """Extract ``progress`` (0..1) of the pit TONNAGE top-down; the active bench is partial.

    Within the active bench, blocks leave in a seeded deterministic order (a centre-out sweep with
    jitter), so a given (pit, progress, seed) always yields the same state.
    """
    if not (0.0 <= progress <= 1.0):
        raise ValueError(f"progress must be in [0,1], got {progress}")
    grid = dep.grid
    in_pit = np.asarray(in_pit, dtype=bool)
    total = float(dep.tonnage[in_pit].sum())
    target = progress * total
    extracted = np.zeros(grid.n_blocks, dtype=bool)
    mined = 0.0
    active = grid.surface_level
    per = grid.nx * grid.ny
    rng = stream(seed, "extraction:order")

    for lv in _pit_levels(grid, in_pit):
        active = lv
        sl = slice(lv * per, (lv + 1) * per)
        idxs = np.nonzero(in_pit[sl])[0] + lv * per
        lv_tonnes = float(dep.tonnage[idxs].sum())
        if mined + lv_tonnes <= target or lv_tonnes == 0.0:
            extracted[idxs] = True
            mined += lv_tonnes
            if mined >= target:
                break
            continue
        # partial bench: deterministic centre-out order with seeded jitter
        ix = idxs % grid.nx
        iy = (idxs // grid.nx) % grid.ny
        cx = (grid.nx - 1) / 2.0
        cy = (grid.ny - 1) / 2.0
        key = np.hypot(ix - cx, iy - cy) + 0.25 * rng.random(idxs.shape[0])
        order = idxs[np.argsort(key, kind="stable")]
        csum = np.cumsum(dep.tonnage[order])
        need = target - mined
        k = 0 if need <= 0 else int(np.searchsorted(csum, need, side="left")) + 1
        take = order[: min(k, order.shape[0])]
        extracted[take] = True
        mined += float(dep.tonnage[take].sum())
        break

    remaining = in_pit & ~extracted
    return ExtractionState(
        active_level=active,
        progress=progress,
        extracted=extracted,
        remaining=remaining,
        tonnes_extracted=mined,
        tonnes_remaining=float(dep.tonnage[remaining].sum()),
    )


def loading_faces(
    dep: Deposit,
    state: ExtractionState,
    econ: Econ,
    n_faces: int = 3,
    seed: int = 1,
) -> list[Face]:
    """Cluster the remaining ACTIVE-bench blocks into ``n_faces`` seeded k-means faces."""
    if n_faces < 1:
        raise ValueError("n_faces must be >= 1")
    grid = dep.grid
    per = grid.nx * grid.ny
    lv = state.active_level
    sl = slice(lv * per, (lv + 1) * per)
    idxs = np.nonzero(state.remaining[sl])[0] + lv * per
    if idxs.shape[0] == 0:  # bench exhausted: fall back to the next level with remaining blocks
        rem_levels = [x for x in _pit_levels(grid, state.remaining) if state.remaining[x * per : (x + 1) * per].any()]
        if not rem_levels:
            return []
        lv = rem_levels[0]
        sl = slice(lv * per, (lv + 1) * per)
        idxs = np.nonzero(state.remaining[sl])[0] + lv * per

    ix = (idxs % grid.nx).astype(np.float64)
    iy = ((idxs // grid.nx) % grid.ny).astype(np.float64)
    pts = np.stack([ix, iy], axis=1)
    k = min(n_faces, idxs.shape[0])

    rng = stream(seed, f"faces:level{lv}")
    centroids = pts[rng.choice(pts.shape[0], size=k, replace=False)]
    assign = np.zeros(pts.shape[0], dtype=np.int64)
    for _ in range(12):  # Lloyd iterations — plenty for bench-scale point sets
        d2 = ((pts[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        assign = d2.argmin(axis=1)
        for c in range(k):
            sel = assign == c
            if sel.any():
                centroids[c] = pts[sel].mean(axis=0)

    cutoff = cutoff_grade(econ)
    faces: list[Face] = []
    for c in range(k):
        sel = idxs[assign == c]
        if sel.shape[0] == 0:
            continue
        t = dep.tonnage[sel]
        g = dep.grade[sel]
        tt = float(t.sum())
        ore_t = float(t[g > cutoff].sum())
        faces.append(
            Face(
                face_id=c,
                level=lv,
                x=float(centroids[c, 0]),
                y=float(centroids[c, 1]),
                n_blocks=int(sel.shape[0]),
                tonnes=tt,
                mean_grade=float((g * t).sum() / tt) if tt > 0 else 0.0,
                ore_fraction=ore_t / tt if tt > 0 else 0.0,
                elevation_m=lv * grid.dz,
            )
        )
    return faces
