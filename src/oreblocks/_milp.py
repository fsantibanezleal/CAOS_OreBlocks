"""One way to solve the package's integer programs, with a starting solution when there is one.

Every exact re-solve in this package (the sliding window, the C-PIT[D] and OPBSP-[D] local searches, the
exact OPBSP) is a 0-1 program handed to HiGHS. Until 0.6.1 each one called scipy's ``milp``, which takes
no starting point, so HiGHS opened every solve with whatever incumbent its own heuristics found. On most
windows that closes at the root in seconds. On a few it does not: measured on a downstream product's
14,400-block layered twin, the fifth window's root LP was 350.6 M and its first incumbent 128.9 M, and the
solver spent 54 minutes reaching 330.7 M and two hours more on the tree. Every caller here already holds
a feasible solution (the incumbent being improved, or one built in a pass over the candidates), so this
module passes it in.

``highspy`` (the HiGHS project's own Python interface) is used when it is installed, which
``oreblocks[milp]`` arranges since 0.6.2; it accepts the start through ``Highs.setSolution``. Without it
the solve falls back to scipy's ``milp`` and the start is only used as a floor. Either way the result
says which backend ran, because the two HiGHS builds are different versions and a reproducibility check
has to know which one it is comparing.

The contract every caller relies on: **the returned solution is never worse than the start.** A solver
that fails, or that stops on its gap with an incumbent below the start (it cannot with ``setSolution``,
but the fallback has no such guarantee), returns the start itself, flagged as such.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MilpResult", "milp_backend", "solve_binary_program"]


@dataclass(frozen=True)
class MilpResult:
    """What a solve returned. ``x`` is ``None`` only when there was no start and the solve failed."""

    x: np.ndarray | None
    objective: float  # of the MINIMISATION, c'x; +inf when x is None
    backend: str  # "highspy <version>" or "scipy <version>"
    status: str
    used_start: bool  # the returned x IS the start (the solver failed or did not beat it)
    start_objective: float | None = None
    start_rejected: bool = False  # a start was passed and failed the feasibility re-check
    nodes: int = -1  # branch-and-bound nodes the solver explored (-1 when the backend does not say)
    gap: float = float("nan")  # the solver's final relative MIP gap


def _highspy():
    try:
        import highspy
    except ImportError:  # pragma: no cover - exercised only without the extra
        return None
    return highspy


def milp_backend() -> str:
    """The backend :func:`solve_binary_program` will use, as it is written into results."""
    hs = _highspy()
    if hs is not None:
        h = hs.Highs()
        return f"highspy {h.versionMajor()}.{h.versionMinor()}.{h.versionPatch()}"
    import scipy

    return f"scipy {scipy.__version__}"


def _start_ok(a_csr, lo, hi, lb, ub, x0, tol: float = 1e-6) -> bool:
    """Feasibility of the start, checked here so a caller's bookkeeping slip cannot be passed off."""
    if x0 is None:
        return False
    if np.any(x0 < lb - tol) or np.any(x0 > ub + tol):
        return False
    ax = a_csr @ x0
    return bool(np.all(ax >= lo - tol * np.maximum(1.0, np.abs(lo)))
                and np.all(ax <= hi + tol * np.maximum(1.0, np.abs(hi))))


def solve_binary_program(
    c: np.ndarray,
    a_csr,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    lb: np.ndarray | float = 0.0,
    ub: np.ndarray | float = 1.0,
    mip_gap: float = 1e-4,
    time_limit: float | None = None,
    start: np.ndarray | None = None,
    strict_start: bool = False,
    node_limit: int | None = None,
    warm: bool = True,
) -> MilpResult:
    """Minimise ``c'x`` over ``lo <= A x <= hi``, ``lb <= x <= ub``, ``x`` integer.

    ``time_limit=None`` stops on the relative gap alone, so a bake lands in the same place on every
    machine. ``start`` must be feasible. An infeasible one means the caller built it wrong: with
    ``strict_start`` it raises, otherwise it is dropped, the solve runs cold, and ``start_rejected``
    says so, so a caller can count it rather than crash a day-long bake on a rounding edge.

    ``warm=False`` keeps the start as a FLOOR only: HiGHS solves cold and the start is returned when the
    solver ends below it. That is what the sliding window uses (0.6.3), because at a loose gap a warm
    start anchors the search: HiGHS proves the gap from the start at once and stops near it.

    ``node_limit`` stops the branch and bound after that many nodes and keeps the best incumbent (never
    below the start). It is a COUNT, not a clock, so unlike ``time_limit`` it lands in the same place on
    every machine; None (the default) stops on the gap alone.
    """
    n = int(c.shape[0])
    c = np.asarray(c, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    lb_v = np.broadcast_to(np.asarray(lb, dtype=np.float64), (n,)).copy()
    ub_v = np.broadcast_to(np.asarray(ub, dtype=np.float64), (n,)).copy()

    start_obj: float | None = None
    rejected = False
    if start is not None:
        start = np.asarray(start, dtype=np.float64)
        if not _start_ok(a_csr, lo, hi, lb_v, ub_v, start):
            if strict_start:
                raise ValueError("the starting solution passed to the MILP is infeasible")
            start, rejected = None, True
        else:
            start_obj = float(c @ start)

    hs = _highspy()
    backend = milp_backend()
    x: np.ndarray | None = None
    status = "not-run"
    nodes, gap = -1, float("nan")
    try:
        if hs is not None:
            x, status, nodes, gap = _solve_highspy(hs, c, a_csr, lo, hi, lb_v, ub_v, mip_gap, time_limit,
                                                   start if warm else None, node_limit)
        else:  # pragma: no cover - exercised only without the extra
            x, status = _solve_scipy(c, a_csr, lo, hi, lb_v, ub_v, mip_gap, time_limit, node_limit)
    except Exception as exc:  # noqa: BLE001 - a solver failure must never lose the start
        status = f"error: {type(exc).__name__}"
        x = None

    if x is not None:
        x = np.round(x)  # integral by construction; strip the solver's 1e-9 noise
        if not _start_ok(a_csr, lo, hi, lb_v, ub_v, x, tol=1e-5):
            status = f"{status} (solution failed the feasibility re-check)"
            x = None
    if x is not None:
        obj = float(c @ x)
        if start is None or obj <= start_obj + 1e-9 * max(1.0, abs(start_obj)):
            return MilpResult(x, obj, backend, status, False, start_obj, rejected, nodes, gap)
    if start is not None:
        return MilpResult(start.copy(), start_obj, backend, status, True, start_obj, rejected, nodes, gap)
    return MilpResult(None, float("inf"), backend, status, False, None, rejected, nodes, gap)


def _solve_highspy(hs, c, a_csr, lo, hi, lb, ub, mip_gap, time_limit, start, node_limit=None):
    csc = a_csr.tocsc()
    h = hs.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("mip_rel_gap", float(mip_gap))
    h.setOptionValue("presolve", "on")
    if time_limit is not None:
        h.setOptionValue("time_limit", float(time_limit))
    if node_limit is not None:
        h.setOptionValue("mip_max_nodes", int(node_limit))
    inf = hs.kHighsInf
    lp = hs.HighsLp()
    lp.num_col_ = int(c.shape[0])
    lp.num_row_ = int(a_csr.shape[0])
    lp.col_cost_ = c
    lp.col_lower_ = lb
    lp.col_upper_ = ub
    lp.row_lower_ = np.where(np.isfinite(lo), lo, -inf)
    lp.row_upper_ = np.where(np.isfinite(hi), hi, inf)
    lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
    lp.a_matrix_.start_ = csc.indptr.astype(np.int32)
    lp.a_matrix_.index_ = csc.indices.astype(np.int32)
    lp.a_matrix_.value_ = csc.data.astype(np.float64)
    lp.integrality_ = [hs.HighsVarType.kInteger] * lp.num_col_
    h.passModel(lp)
    if start is not None:
        sol = hs.HighsSolution()
        sol.col_value = list(map(float, start))
        h.setSolution(sol)
    h.run()
    st = h.getModelStatus()
    name = h.modelStatusToString(st)
    info = h.getInfo()
    # a primal solution exists whenever HiGHS found any feasible point (optimal, or stopped on a limit)
    nodes, gap = int(info.mip_node_count), float(info.mip_gap)
    if int(info.primal_solution_status) == 2:
        return np.asarray(h.getSolution().col_value, dtype=np.float64), name, nodes, gap
    return None, name, nodes, gap


def _solve_scipy(c, a_csr, lo, hi, lb, ub, mip_gap, time_limit, node_limit=None):  # pragma: no cover
    from scipy.optimize import Bounds, LinearConstraint, milp

    options: dict = {"mip_rel_gap": mip_gap, "presolve": True}
    if time_limit is not None:
        options["time_limit"] = time_limit
    if node_limit is not None:
        options["node_limit"] = int(node_limit)
    res = milp(
        c=c,
        constraints=LinearConstraint(a_csr, lo, hi),
        integrality=np.ones(c.shape[0]),
        bounds=Bounds(lb, ub),
        options=options,
    )
    if res.x is None:
        return None, str(res.message)
    return np.asarray(res.x), str(res.message)
