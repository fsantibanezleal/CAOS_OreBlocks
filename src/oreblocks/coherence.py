"""Spatial coherence of a schedule: is each period one workable volume, or confetti?

A block-level schedule is free to pick blocks anywhere in the model, and it does. Chicoisne,
Espinoza, Goycoolea, Moreno and Rubio say so themselves in the Final Remarks of the paper that
introduced the algorithm (Operations Research 60(3):517-528, doi:10.1287/opre.1120.1050):

    "It is likely that the C-PIT solutions are such that blocks scheduled in a same time period are
    scattered throughout the mine. This might lead to schedules that require manual intervention by
    mining engineers to consider additional operational constraints [...] exacerbated by the fact
    that our minimal planning units are blocks rather than bench-phases."

A schedule with a high NPV and forty disconnected fragments per year is not a mine plan, and no NPV
chart shows the difference. This module measures it, so the honest number sits next to the flattering
one. Bai, Marcotte, Gamache, Gregory and Lapworth (J. S. Afr. Inst. Min. Metall. 118(5), 2018,
doi:10.17159/2411-9717/2018/v118n5a8) give the operational reason a minimum mining width matters:
equipment needs room, and they target about 100 m.

Connectivity is 6-neighbour (face-sharing) on the regular grid. Two blocks that touch only along an
edge or a corner are not the same mining front.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PeriodCoherence", "period_coherence", "schedule_coherence"]


@dataclass(frozen=True)
class PeriodCoherence:
    """Coherence of one period's mined increment."""

    period: int
    blocks: int
    components: int
    largest_component: int
    min_width_blocks: int

    @property
    def largest_share(self) -> float:
        return 0.0 if self.blocks == 0 else self.largest_component / self.blocks


def _label(coords: np.ndarray) -> tuple[int, np.ndarray]:
    """Union-find over 6-connectivity. ``coords`` is (m, 3) integer x, y, level."""
    m = coords.shape[0]
    if m == 0:
        return 0, np.zeros(0, dtype=np.int64)
    index = {(int(a), int(b), int(c)): i for i, (a, b, c) in enumerate(coords)}
    parent = np.arange(m, dtype=np.int64)

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = int(parent[root])
        while parent[i] != root:
            parent[i], i = root, int(parent[i])
        return root

    for i, (a, b, c) in enumerate(coords):
        for da, db, dc in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            j = index.get((int(a) + da, int(b) + db, int(c) + dc))
            if j is not None:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    roots = np.array([find(i) for i in range(m)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return int(labels.max()) + 1, labels


def _min_width(coords: np.ndarray) -> int:
    """Smallest run of consecutive mined blocks along x or y within a single level.

    A crude but honest proxy for the minimum mining width: if the narrowest place a shovel has to
    work is one block across, the plan is not operable whatever its NPV says.
    """
    if coords.shape[0] == 0:
        return 0
    best = None
    for axis, other in ((0, 1), (1, 0)):
        key: dict[tuple[int, int], list[int]] = {}
        for a, b, c in coords:
            key.setdefault((int(c), int(b if axis == 0 else a)), []).append(int(a if axis == 0 else b))
        for vals in key.values():
            vals.sort()
            run = 1
            for k in range(1, len(vals)):
                if vals[k] == vals[k - 1] + 1:
                    run += 1
                else:
                    best = run if best is None else min(best, run)
                    run = 1
            best = run if best is None else min(best, run)
        _ = other
    return int(best or 0)


def period_coherence(
    period_of_block: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    level: np.ndarray,
    period: int,
) -> PeriodCoherence:
    """Coherence of the blocks mined in one period."""
    sel = np.asarray(period_of_block) == period
    coords = np.stack([np.asarray(x)[sel], np.asarray(y)[sel], np.asarray(level)[sel]], axis=1)
    n_comp, labels = _label(coords)
    largest = 0 if n_comp == 0 else int(np.bincount(labels).max())
    return PeriodCoherence(
        period=period,
        blocks=int(coords.shape[0]),
        components=n_comp,
        largest_component=largest,
        min_width_blocks=_min_width(coords),
    )


def schedule_coherence(
    period_of_block: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    level: np.ndarray,
    n_periods: int,
) -> list[PeriodCoherence]:
    """Coherence for every period of a schedule, in period order."""
    return [period_coherence(period_of_block, x, y, level, t) for t in range(n_periods)]
