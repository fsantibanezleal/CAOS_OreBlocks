"""A compiled maximum-closure path, for the graphs the pure-Python one cannot finish.

The package's own Dinic is fine at the scale of a block model: `newman1` is 1,060 blocks and a
closure takes milliseconds. It is NOT fine on a TIME-EXPANDED graph, which is what the
Bienstock-Zuckerberg pricing problem lives on: a 10,976-block deposit over ten periods is 109,760
nodes and 972,904 arcs, and the interpreter loses. That is the entire reason the joint bound was
reported as "over budget" on every deposit twin, with Algorithm 4's looser certified bound used
instead.

`scipy.sparse.csgraph.maximum_flow` is compiled, and it takes INTEGER capacities. Scaling a float
objective to integers is normally where a bound quietly stops being a bound, so the rounding here is
directional and the error is stated rather than hoped about.

THE ROUNDING, and why the result is still an upper bound
--------------------------------------------------------
Round every node weight UP at scale ``s``::

    w'_b = ceil(s * w_b) / s   >=   w_b

For ANY closure ``C``, ``sum_{b in C} w'_b >= sum_{b in C} w_b``, so the maximum over closures
satisfies ``max' >= max``. The value this function reports is therefore an OVER-estimate of the true
maximum closure value, by at most ``n / s``, and it is exactly over-estimation that keeps the
Bienstock-Zuckerberg termination certificate valid: ``L(pi)`` is an upper bound on the optimum for
every dual vector ``pi``, and a pricing solve that UNDER-estimates ``L(pi)`` would report a bound
that is not one. An over-estimate is safe; an under-estimate is a wrong answer that looks right.

``s`` is as large as the solver's measured integer ceiling allows given the total positive mass,
which puts the slack around ``1e-5`` relative. That is small against the thing the joint bound is
there to measure (a tightening of order ``1e-4``) and it is NOT small enough to ignore, so it is
returned rather than assumed: the caller records it and the product prints it next to the bound.

The closure SET is recovered from the residual graph, the source side of the minimum cut, which is
the same construction the pure-Python path uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .precedence import Precedence

__all__ = ["ClosureResult", "has_fast_cut", "max_closure_fast"]

#: The largest total capacity `scipy.sparse.csgraph.maximum_flow` handles correctly.
#:
#: MEASURED, not assumed. It accepts an int64 matrix and returns a plausible number well past int32,
#: but the answer is WRONG: on a 10,976-block closure whose true value is 769,925,542, a total
#: capacity of 2**30 gives +3,210 (an over-estimate inside the rounding slack, as intended) and
#: 2**31 gives +15,491,856, which is the whole positive mass, meaning the flow came back as
#: essentially zero. The transition is exactly at 2**31, so something inside is int32. This ceiling
#: is one power of two below the observed failure, and `test_fastcut.py` re-measures it.
_TOTAL_CAPACITY_CEILING = 2**30


def has_fast_cut() -> bool:
    """True when the compiled max-flow is importable. It lives in the ``milp`` extra."""
    try:
        from scipy.sparse.csgraph import maximum_flow  # noqa: F401
    except Exception:  # pragma: no cover - exercised by the fallback path
        return False
    return True


@dataclass(frozen=True)
class ClosureResult:
    """A maximum closure, its value, and the price of having used integer arithmetic."""

    mask: np.ndarray  # (n,) bool
    value: float  # OVER-estimates the true maximum by at most `slack`
    slack: float  # absolute, = n_nodes / scale
    scale: float


def max_closure_fast(
    weights: np.ndarray,
    prec: Precedence,
    candidates: np.ndarray | None = None,
) -> ClosureResult:
    """Maximum closure over ``candidates`` using the compiled max-flow.

    Same semantics as ``upit.max_closure_within``: a closure means every predecessor of an included
    block is included. Blocks outside ``candidates`` are excluded and their arcs dropped.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow

    w = np.asarray(weights, dtype=np.float64)
    n = w.shape[0]
    cand = np.ones(n, dtype=bool) if candidates is None else np.asarray(candidates, dtype=bool)
    idx = np.nonzero(cand)[0]
    m = idx.shape[0]
    out = np.zeros(n, dtype=bool)
    if m == 0:
        return ClosureResult(mask=out, value=0.0, slack=0.0, scale=1.0)

    sub = w[idx]
    positive_mass = float(sub[sub > 0].sum())
    if positive_mass <= 0.0:
        return ClosureResult(mask=out, value=0.0, slack=0.0, scale=1.0)

    # The maximum flow can never exceed the total positive mass, so that IS the infinite capacity,
    # and the scale is whatever fits under the measured ceiling. Everything about the precision of
    # this call follows from one division.
    scale = float(_TOTAL_CAPACITY_CEILING) / max(1.0, positive_mass)

    caps = np.ceil(sub * scale).astype(np.int64)  # w' >= w, in scaled units
    src_cap = np.where(caps > 0, caps, 0)
    snk_cap = np.where(caps < 0, -caps, 0)
    inf = int(src_cap.sum()) + 1

    local = np.full(n, -1, dtype=np.int64)
    local[idx] = np.arange(m)
    s, t = m, m + 1

    # precedence arcs, restricted to the candidate set, vectorised
    lo = prec.pstart[idx]
    hi = prec.pstart[idx + 1]
    counts = (hi - lo).astype(np.int64)
    if counts.sum() > 0:
        tails = np.repeat(np.arange(m, dtype=np.int64), counts)
        offsets = np.arange(counts.sum(), dtype=np.int64) - np.repeat(
            np.cumsum(counts) - counts, counts
        )
        heads_global = prec.plist[np.repeat(lo, counts) + offsets]
        heads = local[heads_global]
        keep = heads >= 0
        tails, heads = tails[keep], heads[keep]
    else:
        tails = np.zeros(0, dtype=np.int64)
        heads = np.zeros(0, dtype=np.int64)

    has_src = src_cap > 0
    has_snk = snk_cap > 0
    rows = np.concatenate([
        np.full(int(has_src.sum()), s, dtype=np.int64),
        np.nonzero(has_snk)[0].astype(np.int64),
        tails,
    ])
    cols = np.concatenate([
        np.nonzero(has_src)[0].astype(np.int64),
        np.full(int(has_snk.sum()), t, dtype=np.int64),
        heads,
    ])
    data = np.concatenate([
        src_cap[has_src],
        snk_cap[has_snk],
        np.full(tails.shape[0], inf, dtype=np.int64),
    ])

    graph = csr_matrix((data, (rows, cols)), shape=(m + 2, m + 2), dtype=np.int64)
    graph.sum_duplicates()
    res = maximum_flow(graph, s, t)

    closure_scaled = int(src_cap.sum()) - int(res.flow_value)
    mask_local = _source_side(graph, res.flow, s, m + 2)
    out[idx] = mask_local[:m]

    return ClosureResult(
        mask=out,
        value=closure_scaled / scale,
        slack=m / scale,
        scale=scale,
    )


def _source_side(graph, flow, source: int, n_nodes: int) -> np.ndarray:
    """Nodes reachable from the source in the RESIDUAL graph: the source side of the minimum cut.

    Breadth-first by FRONTIER, in numpy. A node-at-a-time Python loop over a million residual arcs
    costs more than the compiled max-flow it exists to interpret, which turned a 6x win into a 0.7x
    loss on the first measurement.
    """

    residual = (graph - flow).tocsr()  # scipy's flow is antisymmetric, so this IS the residual
    residual.eliminate_zeros()
    indptr, indices, data = residual.indptr, residual.indices, residual.data
    positive = data > 0

    seen = np.zeros(n_nodes, dtype=bool)
    seen[source] = True
    frontier = np.array([source], dtype=np.int64)
    while frontier.size:
        counts = (indptr[frontier + 1] - indptr[frontier]).astype(np.int64)
        if counts.sum() == 0:
            break
        starts = np.repeat(indptr[frontier], counts)
        offsets = np.arange(counts.sum(), dtype=np.int64) - np.repeat(
            np.cumsum(counts) - counts, counts
        )
        slots = starts + offsets
        nxt = indices[slots[positive[slots]]]
        nxt = np.unique(nxt)
        nxt = nxt[~seen[nxt]]
        seen[nxt] = True
        frontier = nxt
    return seen
