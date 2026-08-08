"""The Bienstock-Zuckerberg decomposition, for real.

BZ solves the LP relaxation of a **General Precedence Constrained Problem** (GPCP)

.. code-block:: text

    Z = max   c'z
        s.t.  z_i <= z_j          for all (i, j) in I     (precedence)
              Hz <= h                                      (side constraints, e.g. capacities)
              z in {0,1}^n

It is the algorithm behind the large-scale mine-scheduling literature: Bienstock and Zuckerberg,
IPCO 2010, doi:10.1007/978-3-642-13036-6_1, and the study that makes it implementable, Munoz,
Espinoza, Goycoolea, Moreno, Queyranne and Rivera Letelier, Comput Optim Appl, 2017,
doi:10.1007/s10589-017-9946-1 (arXiv:1607.01104), which recasts it as **column generation** and is
the version this module follows.

**What it is worth, stated honestly.** The bound is exactly the LP bound: ``Z_BZ = Z_LP``, proven,
because ``{z : z_i <= z_j}`` is totally unimodular. BZ is a SPEED result at a scale where a general
LP solver produces nothing at all. Measured by its own authors on ``zuck_medium``: 40 seconds against
954,405 seconds for CPLEX 12.6, and CPLEX solved only the five smallest of fifteen instances.

**Why this implementation exists in a package that already has the critical multiplier algorithm.**
CMA is exact and fast but needs a SINGLE resource constraint per period; with two it is run once per
constraint and the best of those bounds is kept, which is certified but LOOSE. BZ handles an
arbitrary number of side constraints jointly, so it closes the part of a reported gap that belongs to
the bound rather than to the heuristic. On a two-capacity instance those are two different things and
only BZ can separate them.

**The mechanics.** Column generation whose restricted master runs over the LINEAR hull of the
precedence polytope (not the convex hull, which is what Dantzig-Wolfe uses and why it converges
slowly here), with generator matrices whose columns are **orthogonal 0-1 vectors**. Restricting to
the span of such a matrix is equivalent to EQUATING the variables inside each support, which is a
contraction: rows and variables collapse while the structure survives. The pricing problem is
``max (c - pi'H)'v`` over closures, which is a maximum closure, which is a minimum cut, so the whole
algorithm rides on the same max-flow this package already ships.

The refining step, from the paper: given the incumbent generator columns ``v^1..v^r`` and a new
pricing solution ``v``, the next matrix is the non-zero vectors of

.. code-block:: text

    {v^j AND v : j} union {v^j MINUS v : j} union {v MINUS (union_j v^j)}

which yields at most ``2r + 1`` columns and keeps them orthogonal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .upit import max_closure_within

__all__ = ["Gpcp", "BzResult", "solve_gpcp_lp"]


@dataclass
class Gpcp:
    """A general precedence constrained problem in the form BZ solves.

    ``prec_from``/``prec_to`` are parallel arrays: ``z[prec_from[k]] <= z[prec_to[k]]``, that is,
    ``prec_to[k]`` must be taken before ``prec_from[k]`` can be. ``H`` is stored row-sparse as a list
    of ``(indices, coefficients)`` because the capacity rows of a time-expanded scheduling problem are
    dense in blocks and empty everywhere else.
    """

    n: int
    c: np.ndarray  # (n,) objective
    prec_from: np.ndarray  # int64 (m,)
    prec_to: np.ndarray  # int64 (m,)
    h_rows: list[tuple[np.ndarray, np.ndarray]]  # side constraints, one (idx, coef) per row
    h_rhs: np.ndarray  # (R,)
    #: variables forced to 1 (release dates in the scheduling sense); rarely used here
    fixed_one: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @property
    def n_side(self) -> int:
        return len(self.h_rows)


@dataclass(frozen=True)
class BzResult:
    bound: float
    z: np.ndarray  # (n,) fractional solution attaining the bound
    iterations: int
    columns: int
    pricing_solves: int
    converged: bool
    gap_at_stop: float


def _closure_of(g: Gpcp, weights: np.ndarray) -> np.ndarray:
    """Maximum closure of ``weights`` over the precedence system, as a 0-1 mask.

    The package's closure solver takes predecessor lists in CSR, so the arcs are converted once here.
    A closure in that solver means: if block ``b`` is in, every predecessor of ``b`` is in, which is
    exactly ``z_b <= z_a`` for a predecessor ``a``.
    """
    return max_closure_within(weights, g._csr, np.ones(g.n, dtype=bool))  # noqa: SLF001


def _build_csr(g: Gpcp) -> None:
    from .precedence import Precedence

    counts = np.zeros(g.n, dtype=np.int64)
    np.add.at(counts, g.prec_from, 1)
    pstart = np.zeros(g.n + 1, dtype=np.int64)
    np.cumsum(counts, out=pstart[1:])
    plist = np.empty(int(pstart[-1]), dtype=np.int64)
    fill = pstart[:-1].copy()
    for k in range(g.prec_from.shape[0]):
        b = int(g.prec_from[k])
        plist[fill[b]] = g.prec_to[k]
        fill[b] += 1
    g._csr = Precedence(pstart=pstart, plist=plist)  # noqa: SLF001


def _refine(cols: list[np.ndarray], v: np.ndarray) -> list[np.ndarray]:
    """The BZ refining step: intersect, subtract, and add the remainder. Keeps columns orthogonal."""
    out: list[np.ndarray] = []
    union = np.zeros_like(v)
    for col in cols:
        union |= col
        inter = col & v
        if inter.any():
            out.append(inter)
        diff = col & ~v
        if diff.any():
            out.append(diff)
    rest = v & ~union
    if rest.any():
        out.append(rest)
    return out


def _elementary_basis(z: np.ndarray, tol: float = 1e-9) -> list[np.ndarray]:
    """Columns equating variables that share a value: the coarsification of an incumbent."""
    out: list[np.ndarray] = []
    vals = np.unique(np.round(z / tol) * tol)
    for val in vals:
        if abs(val) <= tol:
            continue
        mask = np.abs(z - val) <= tol
        if mask.any():
            out.append(mask.astype(np.uint8).astype(bool))
    return out


def _solve_master(
    g: Gpcp, cols: list[np.ndarray]
) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Restricted master over the span of ``cols``.

    With orthogonal 0-1 columns, restricting ``z`` to ``span(cols)`` means every variable inside a
    column's support takes that column's value, so the master is a SMALL linear program in one
    variable per column:

    - objective   ``max sum_q (sum_{i in supp q} c_i) lambda_q``
    - side rows   ``sum_q (sum_{i in supp q} H_ri) lambda_q <= h_r``
    - precedence  ``lambda_{q(i)} <= lambda_{q(j)}`` for every arc CROSSING two columns
    - bounds      ``0 <= lambda_q <= 1``; a variable in no column is pinned to 0

    Returns ``(objective, lambda, duals of the side rows)``.
    """
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix

    k = len(cols)
    if k == 0:
        return None
    owner = np.full(g.n, -1, dtype=np.int64)
    for q, col in enumerate(cols):
        owner[col] = q

    obj = np.array([float(g.c[col].sum()) for col in cols])

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    n_side = g.n_side
    for idx, coef in g.h_rows:
        row = np.zeros(k)
        own = owner[idx]
        keep = own >= 0
        np.add.at(row, own[keep], coef[keep])
        rows.append(row)
    rhs.extend(float(v) for v in g.h_rhs)

    # precedence between distinct columns, deduplicated
    oi = owner[g.prec_from]
    oj = owner[g.prec_to]
    cross = (oi != oj) & (oi >= 0)
    pairs = np.unique(np.stack([oi[cross], oj[cross]], axis=1), axis=0)
    for a, b in pairs:
        row = np.zeros(k)
        row[a] += 1.0
        if b >= 0:
            row[b] -= 1.0
        else:
            # the successor's column is absent, so its value is 0: lambda_a <= 0
            pass
        rows.append(row)
        rhs.append(0.0)

    a_ub = csr_matrix(np.array(rows)) if rows else None
    res = linprog(
        -obj,
        A_ub=a_ub,
        b_ub=np.array(rhs) if rows else None,
        bounds=[(0.0, 1.0)] * k,
        method="highs",
    )
    if not res.success:
        return None
    duals = np.zeros(n_side)
    if res.ineqlin is not None and n_side:
        duals = np.maximum(0.0, -np.asarray(res.ineqlin.marginals)[:n_side])
    return float(-res.fun), np.asarray(res.x), duals


def solve_gpcp_lp(
    g: Gpcp,
    *,
    max_iter: int = 200,
    tol: float = 1e-7,
    coarsify_at: int = 240,
    start_columns: list[np.ndarray] | None = None,
) -> BzResult:
    """Solve the LP relaxation of a GPCP by the Bienstock-Zuckerberg algorithm.

    Terminates on the certificate the paper gives: the pricing problem's value ``L(pi)`` is an upper
    bound on the integer optimum for every dual vector, so once the best pricing bound meets the
    master's value the LP is solved. There is no interval-width guesswork anywhere in the loop.
    """
    _build_csr(g)
    cols: list[np.ndarray] = list(start_columns) if start_columns else [np.ones(g.n, dtype=bool)]
    best_ub = np.inf
    pricing_solves = 0
    z = np.zeros(g.n)
    it = 0
    converged = False

    for it in range(1, max_iter + 1):  # noqa: B007 - `it` is reported in the result
        solved = _solve_master(g, cols)
        if solved is None:
            break
        z_lower, lam, pi = solved
        z = np.zeros(g.n)
        for q, col in enumerate(cols):
            z[col] = lam[q]

        # pricing: max (c - pi'H)'v over closures. L(pi) = that value + pi'h is an upper bound.
        weights = np.array(g.c, dtype=np.float64)
        for r, (idx, coef) in enumerate(g.h_rows):
            if pi[r] != 0.0:
                np.add.at(weights, idx, -pi[r] * coef)
        v = _closure_of(g, weights)
        pricing_solves += 1
        l_pi = float(weights[v].sum()) + float(pi @ g.h_rhs)
        best_ub = min(best_ub, l_pi)

        if best_ub - z_lower <= tol * max(1.0, abs(best_ub)):
            converged = True
            break

        new_cols = _refine(cols, v)
        if not new_cols or len(new_cols) == len(cols) and all(
            np.array_equal(a, b) for a, b in zip(new_cols, cols, strict=False)
        ):
            converged = True
            break
        cols = new_cols
        if len(cols) > coarsify_at:
            # coarsify onto the elementary basis of the incumbent, only after a strict improvement
            cols = _elementary_basis(z) or cols

    return BzResult(
        bound=float(best_ub if np.isfinite(best_ub) else 0.0),
        z=z,
        iterations=it,
        columns=len(cols),
        pricing_solves=pricing_solves,
        converged=converged,
        gap_at_stop=float(best_ub - (z @ g.c)) if np.isfinite(best_ub) else float("inf"),
    )
