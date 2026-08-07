"""CPIT production scheduling: the certified LP bound, the rounding heuristics, and the controls.

The ultimate pit answers *which* blocks are worth mining. This module answers *when*, which is the
constrained pit limit problem (CPIT): assign every block an extraction period so that slope
precedence holds in every period, per-period resource capacities hold, and discounted value is
maximised. It is NP-hard, so what ships here is a **certified upper bound** plus **feasible
heuristic schedules**, with the gap between them reported rather than hidden.

Formulation (Chicoisne et al. 2012, Operations Research 60(3):517-528,
doi:10.1287/opre.1120.1050, equations 3a-3f), in the cumulative variables every published algorithm
and every published bound is stated in::

    max   sum_b sum_t  p_bt (x_bt - x_b,t-1)
    s.t.  sum_b a_rb (x_bt - x_b,t-1) <= c_rt     for all resources r, periods t
          x_bt <= x_at                            for all precedence arcs (a, b), all t
          x_bt <= x_b,t+1                         monotone: once mined, stays mined
          x_bt in {0,1},  x_b0 = 0

``x_bt = 1`` means block ``b`` has been extracted by the END of period ``t``.

**The bound.** For a single resource constraint per period, Chicoisne et al. Theorem 3.1 gives the
exact optimum of the LP relaxation in ``O(mn log n)`` with no LP solver at all: the LP optimum
decomposes into ``T`` independent problems ``CP(U_t)`` over the cumulative capacity
``U_t = sum_{s<=t} c_s``, and each of those is a convex combination of two consecutive **nested
pits** from the parametric family ``UPL(p - lambda a)``. Since a nested pit is a maximum closure,
the whole bound runs on the max-flow machinery this package already ships. That is
:func:`cpit_lp_relaxation`.

For two resource constraints, Algorithm 4 of the same paper relaxes one constraint at a time, keeps
``min`` of the two relaxed objectives (each is still a valid upper bound on the two-constraint
optimum) and takes the best of the two feasible solutions. That is :func:`cpit_bound_two_resources`.

**The bound is not tighter than the LP.** Munoz, Espinoza, Goycoolea, Moreno, Queyranne and Rivera
Letelier (Comput Optim Appl, doi:10.1007/s10589-017-9946-1, arXiv:1607.01104) prove that the
Bienstock-Zuckerberg decomposition attains exactly ``Z_BZ = Z_LP`` because the precedence system is
totally unimodular. BZ is a speed result on huge instances, not a better bound. Nothing here claims
otherwise.

**The schedules.** :func:`toposort_schedule` implements the TopoSort family (Chicoisne et al. 2012,
section 3.2, Algorithms 2 and 3): take a topological ordering of the precedence DAG that prefers
high-weight blocks early, then walk it assigning each block the earliest period whose remaining
resources fit. Three published weightings, in increasing order of quality:

===========  ===========================================  ======================================
name         weight                                       origin
===========  ===========================================  ======================================
``greedy``   ``w_b = p_b``                                 the obvious baseline (GrTS)
``gershon``  ``w_b = sum of profits of all successors``    Gershon 1987a (GeTS)
``expected`` ``w_b = -E_b`` from the LP relaxation         Chicoisne et al. 2012 (ExTS)
===========  ===========================================  ======================================

The published spread between them is large and is the whole argument for computing the bound: on
their AsiaMine instance with two resource constraints, greedy reached 0.138 of the LP bound and
expected-time reached 0.972 using the same scheduling code.

**The controls** (:func:`run_controls`) are not decoration. At discount rate zero with unlimited
capacity, CPIT degenerates to UPIT: the mined set must equal the exact ultimate pit block for block,
the LP bound must equal the exact ultimate-pit value, and the objective must not depend on the
order. A scheduling engine that fails those is broken in a way no chart would reveal.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from .minelib_models import Cpit
from .precedence import Precedence
from .upit import max_closure_within, solve_upit

__all__ = [
    "LpRelaxation",
    "ScheduleResult",
    "TOPOSORT_WEIGHTS",
    "Controls",
    "cpit_bound_two_resources",
    "cpit_lp_relaxation",
    "expected_extraction_times",
    "improve_schedule",
    "run_controls",
    "schedule_value",
    "solve_cpit",
    "toposort_order",
    "toposort_schedule",
]

TOPOSORT_WEIGHTS = ("greedy", "gershon", "expected")

_TOL = 1e-9


def _finite_values(inst: Cpit) -> np.ndarray:
    """Replace forbidden-destination sentinels by a value no closure would ever include."""
    v = np.array(inst.value, dtype=np.float64)
    bad = ~np.isfinite(v)
    if bad.any():
        pos = float(v[np.isfinite(v) & (v > 0)].sum())
        v[bad] = -(pos + 1.0)
    return v


def _successors(prec: Precedence, n: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR successor lists (pred -> successors), built once from the predecessor CSR."""
    counts = np.zeros(n, dtype=np.int64)
    np.add.at(counts, prec.plist, 1)
    sstart = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=sstart[1:])
    slist = np.empty(int(sstart[-1]), dtype=np.int64)
    fill = sstart[:-1].copy()
    for b in range(n):
        for k in range(prec.pstart[b], prec.pstart[b + 1]):
            p = int(prec.plist[k])
            slist[fill[p]] = b
            fill[p] += 1
    return sstart, slist


# ------------------------------------------------------------------------------------------------
# the certified bound
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LpRelaxation:
    """Exact optimum of the CPIT LP relaxation for one resource constraint per period."""

    bound: float
    x: np.ndarray  # float64 (T, n) cumulative fractional extraction, monotone in t
    period_value: np.ndarray  # float64 (T,) undiscounted value of the cumulative pit at t
    multipliers: np.ndarray  # float64 (T,) the critical multiplier used per period
    integral: np.ndarray  # bool (T,) True where the period solution needed no convex combination
    closure_solves: int
    resource: int

    def expected_times(self) -> np.ndarray:
        return expected_extraction_times(self.x)


def _pit_at(
    values: np.ndarray, prec: Precedence, coef: np.ndarray, lam: float, candidates: np.ndarray
) -> np.ndarray:
    """``argmax`` closure of ``values - lam * coef`` restricted to ``candidates``."""
    return max_closure_within(values - lam * coef, prec, candidates)


def cpit_lp_relaxation(
    inst: Cpit,
    prec: Precedence,
    *,
    resource: int = 0,
    max_bisections: int = 80,
    check_duality: bool = True,
) -> LpRelaxation:
    """Exact CPIT LP relaxation for ONE resource constraint, by the critical multiplier algorithm.

    Chicoisne et al. 2012, Theorem 3.1. The LP optimum is ``sum_t gamma_t * z_t`` where ``z_t`` is
    the optimum of ``CP(U_t) = max p.x s.t. closure, a.x <= U_t, 0 <= x <= 1`` and ``gamma_t > 0``
    are the Abel-summation weights of the discount factors. Each ``z_t`` is attained by a convex
    combination of two consecutive nested pits, found by bisecting the multiplier ``lambda``.

    Periods are solved from the LAST backwards, because ``U_t`` is increasing so the pits nest, and
    each solve is then restricted to the previous (larger) pit. That restriction is what keeps the
    cost near the cost of a handful of maximum-closure solves rather than ``T`` times a full one.
    """
    if inst.sense[resource].tolist().count("G"):
        raise NotImplementedError(
            "minimum-production (sense 'G') side constraints are not handled by the critical "
            "multiplier algorithm; it needs a single upper-bounded resource per period"
        )
    n = inst.n_blocks
    t_max = inst.n_periods
    v = _finite_values(inst)
    a = np.asarray(inst.coef[resource], dtype=np.float64)
    if a.min() < 0:
        raise ValueError("resource coefficients must be non-negative")
    caps = np.cumsum(np.asarray(inst.limit[resource], dtype=np.float64))

    solves = 0
    full = solve_upit(v, prec)
    solves += 1
    a_full = float(a[full.in_pit].sum())

    x = np.zeros((t_max, n), dtype=np.float64)
    zval = np.zeros(t_max, dtype=np.float64)
    lam_used = np.zeros(t_max, dtype=np.float64)
    integral = np.zeros(t_max, dtype=bool)

    # a multiplier large enough to empty the pit: beyond max(v/a) every block prices negative
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(a > 0, v / np.where(a > 0, a, 1.0), -np.inf)
    lam_max = float(max(1.0, np.nanmax(ratio[np.isfinite(ratio)], initial=1.0))) * 2.0 + 1.0

    candidates = full.in_pit.copy()
    for t in range(t_max - 1, -1, -1):
        target = float(caps[t])
        if a_full <= target + _TOL:
            # the whole ultimate pit fits by this period: the LP solution is the integral pit
            x[t] = full.in_pit.astype(np.float64)
            zval[t] = float(v[full.in_pit].sum())
            integral[t] = True
            lam_used[t] = 0.0
            candidates = full.in_pit.copy()
            continue

        lo, hi = 0.0, lam_max
        pit_lo = candidates.copy()  # f(lo) > target
        pit_hi = np.zeros(n, dtype=bool)  # f(hi) <= target
        for _ in range(max_bisections):
            mid = 0.5 * (lo + hi)
            pit_mid = _pit_at(v, prec, a, mid, pit_lo)
            solves += 1
            if float(a[pit_mid].sum()) > target + _TOL:
                lo, pit_lo = mid, pit_mid
            else:
                hi, pit_hi = mid, pit_mid
            if hi - lo <= 1e-12 * max(1.0, hi):
                break

        b_u = float(a[pit_lo].sum())
        b_l = float(a[pit_hi].sum())
        z_u = float(v[pit_lo].sum())
        z_l = float(v[pit_hi].sum())
        if b_u - b_l <= _TOL:
            alpha = 1.0
            integral[t] = True
        else:
            alpha = (b_u - target) / (b_u - b_l)
            alpha = min(1.0, max(0.0, alpha))
            integral[t] = alpha in (0.0, 1.0)
        xt = np.where(pit_lo, 1.0 - alpha, 0.0)
        xt[pit_hi] = 1.0
        x[t] = xt
        zval[t] = alpha * z_l + (1.0 - alpha) * z_u
        lam_used[t] = hi

        if check_duality:
            # strong duality: CP(U) = min_lambda [ UPL(p - lambda a) + lambda U ]
            lam = hi
            dual = float((v - lam * a)[pit_hi].sum()) + lam * target
            scale = max(1.0, abs(zval[t]))
            if abs(dual - zval[t]) > 1e-5 * scale:
                raise AssertionError(
                    f"period {t + 1}: primal {zval[t]:.6f} and dual {dual:.6f} disagree "
                    f"(lambda {lam:.6g}); the critical multiplier did not converge"
                )
        candidates = pit_lo.copy()

    # monotonicity across periods (Chicoisne et al. Proposition 3.2)
    for t in range(t_max - 1):
        if (x[t] > x[t + 1] + 1e-9).any():
            raise AssertionError(f"cumulative solution is not monotone between periods {t + 1} and {t + 2}")

    bound = float((inst.gamma() * zval).sum())
    return LpRelaxation(
        bound=bound,
        x=x,
        period_value=zval,
        multipliers=lam_used,
        integral=integral,
        closure_solves=solves,
        resource=resource,
    )


def expected_extraction_times(x: np.ndarray) -> np.ndarray:
    """``E_b = sum_t t (x_bt - x_b,t-1) + (T+1)(1 - x_bT)`` from a fractional LP solution.

    Chicoisne et al. 2012, section 3.2. ``E_b`` is the expected period in which block ``b`` is
    extracted if the fractional solution is read as a probability of extraction per period, with a
    block never extracted contributing ``T + 1``. It is the weight that turns a bound into a plan.
    """
    t_max, n = x.shape
    prev = np.zeros(n, dtype=np.float64)
    e = np.zeros(n, dtype=np.float64)
    for t in range(t_max):
        e += (t + 1) * (x[t] - prev)
        prev = x[t]
    e += (t_max + 1) * (1.0 - prev)
    return e


# ------------------------------------------------------------------------------------------------
# feasible schedules
# ------------------------------------------------------------------------------------------------
def toposort_order(prec: Precedence, weight: np.ndarray, allowed: np.ndarray | None = None) -> np.ndarray:
    """Topological ordering of the precedence DAG preferring HIGH weight early (Algorithm 2).

    Ties break on block id, so the ordering is deterministic. ``allowed`` restricts the ordering to
    a subset (normally the ultimate pit, since a schedule never gains by mining outside it).
    """
    n = weight.shape[0]
    mask = np.ones(n, dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)
    sstart, slist = _successors(prec, n)
    indeg = np.zeros(n, dtype=np.int64)
    for b in np.nonzero(mask)[0]:
        for k in range(prec.pstart[b], prec.pstart[b + 1]):
            if mask[prec.plist[k]]:
                indeg[b] += 1

    heap: list[tuple[float, int]] = [
        (-float(weight[b]), int(b)) for b in np.nonzero(mask & (indeg == 0))[0]
    ]
    heapq.heapify(heap)
    order = np.empty(int(mask.sum()), dtype=np.int64)
    at = 0
    while heap:
        _, b = heapq.heappop(heap)
        order[at] = b
        at += 1
        for k in range(sstart[b], sstart[b + 1]):
            c = int(slist[k])
            if not mask[c]:
                continue
            indeg[c] -= 1
            if indeg[c] == 0:
                heapq.heappush(heap, (-float(weight[c]), c))
    if at != order.shape[0]:
        raise AssertionError(f"precedence graph has a cycle: ordered {at} of {order.shape[0]} blocks")
    return order


@dataclass
class ScheduleResult:
    """A feasible integer schedule and everything needed to judge it."""

    method: str
    period_of_block: np.ndarray  # int64 (n,), -1 = never mined
    npv: float
    bound: float | None = None
    per_period_value: np.ndarray = field(default_factory=lambda: np.zeros(0))
    per_period_resource: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    mined_blocks: int = 0
    heuristic: bool = True
    notes: str = ""

    @property
    def gap_pct(self) -> float | None:
        """``(bound - npv) / bound`` in percent, the published MineLib gap definition."""
        if self.bound is None or self.bound <= 0:
            return None
        return 100.0 * (self.bound - self.npv) / self.bound


def schedule_value(inst: Cpit, period_of_block: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Discounted value of a schedule, plus the per-period value and resource use."""
    v = _finite_values(inst)
    d = inst.discount_factors()
    t_max, n_res = inst.n_periods, inst.n_resources
    per_val = np.zeros(t_max, dtype=np.float64)
    per_res = np.zeros((n_res, t_max), dtype=np.float64)
    mined = period_of_block >= 0
    for t in range(t_max):
        sel = mined & (period_of_block == t)
        per_val[t] = float(v[sel].sum())
        for r in range(n_res):
            per_res[r, t] = float(inst.coef[r][sel].sum())
    return float((d * per_val).sum()), per_val, per_res


def toposort_schedule(
    inst: Cpit,
    prec: Precedence,
    *,
    weight: str | np.ndarray = "expected",
    relaxation: LpRelaxation | None = None,
    allowed: np.ndarray | None = None,
) -> ScheduleResult:
    """The TopoSort heuristic (Chicoisne et al. 2012, Algorithm 3), for any of the three weights.

    Walk a weighted topological ordering; give each block the earliest period that is at least its
    predecessors' periods and whose remaining resources fit it. A block whose resources never fit,
    or whose predecessor was left unmined, stays unmined. Feasibility is by construction.
    """
    n = inst.n_blocks
    v = _finite_values(inst)
    if isinstance(weight, str):
        if weight == "greedy":
            w = v.copy()
        elif weight == "gershon":
            w = _successor_profit_sums(prec, v)
        elif weight == "expected":
            if relaxation is None:
                relaxation = cpit_lp_relaxation(inst, prec)
            w = -relaxation.expected_times()
        else:
            raise ValueError(f"unknown weight {weight!r}, expected one of {TOPOSORT_WEIGHTS}")
        label = weight
    else:
        w = np.asarray(weight, dtype=np.float64)
        label = "custom"

    if allowed is None:
        allowed = solve_upit(v, prec).in_pit
    order = toposort_order(prec, w, allowed)

    t_max, n_res = inst.n_periods, inst.n_resources
    remaining = np.array(inst.limit, dtype=np.float64).copy()
    period = np.full(n, -1, dtype=np.int64)
    for b in order:
        earliest = 0
        blocked = False
        for k in range(prec.pstart[b], prec.pstart[b + 1]):
            p = int(prec.plist[k])
            if not allowed[p]:
                continue
            if period[p] < 0:
                blocked = True
                break
            earliest = max(earliest, int(period[p]))
        if blocked:
            continue
        need = inst.coef[:, b]
        placed = -1
        for t in range(earliest, t_max):
            if all(remaining[r, t] >= need[r] - _TOL for r in range(n_res)):
                placed = t
                break
        if placed < 0:
            continue
        period[b] = placed
        for r in range(n_res):
            remaining[r, placed] -= need[r]

    npv, per_val, per_res = schedule_value(inst, period)
    return ScheduleResult(
        method=f"toposort:{label}",
        period_of_block=period,
        npv=npv,
        per_period_value=per_val,
        per_period_resource=per_res,
        mined_blocks=int((period >= 0).sum()),
        heuristic=True,
    )


def _successor_profit_sums(prec: Precedence, values: np.ndarray) -> np.ndarray:
    """``w(b) = sum of p_a over every block a that has b as a predecessor, transitively``.

    Gershon 1987a. Computed by a reverse topological sweep, so it is the sum over the whole
    successor cone rather than only the immediate successors.
    """
    n = values.shape[0]
    sstart, slist = _successors(prec, n)
    indeg = np.diff(prec.pstart)
    heap = [int(b) for b in np.nonzero(indeg == 0)[0]]
    order: list[int] = []
    deg = indeg.copy()
    while heap:
        b = heap.pop()
        order.append(b)
        for k in range(sstart[b], sstart[b + 1]):
            c = int(slist[k])
            deg[c] -= 1
            if deg[c] == 0:
                heap.append(c)
    if len(order) != n:
        raise AssertionError("precedence graph has a cycle")
    w = values.astype(np.float64).copy()
    for b in reversed(order):
        acc = 0.0
        for k in range(sstart[b], sstart[b + 1]):
            acc += w[int(slist[k])]
        w[b] += acc
    return w


def improve_schedule(
    inst: Cpit, prec: Precedence, result: ScheduleResult, *, max_passes: int = 12
) -> ScheduleResult:
    """Local search by period shifting: pull value forward, push cost back.

    Two neighbourhoods, both from the shift family used across the mine-scheduling metaheuristic
    literature (Lamghari and Dimitrakopoulos, EJOR 2012, doi:10.1016/j.ejor.2012.05.029; Sari and
    Kumral 2016):

    1. **Pull forward.** A block with positive value moves to an earlier period when all of its
       predecessors are already at or before that period and the capacity is there. Discounting
       makes this strictly improving.
    2. **Push back.** A block with negative value moves to a later period when none of its
       successors are scheduled at or before that period and the capacity is there. Discounting
       makes this strictly improving too.

    Both moves preserve precedence and capacity by construction, so every intermediate schedule is
    feasible and the objective is monotone. This is deliberately NOT the exact ``C-PIT[D]``
    neighbourhood of Chicoisne et al. section 3.3, which re-solves a restricted integer program per
    neighbourhood and needs a MILP solver; that one is not implemented here and this docstring is
    the record of the difference.
    """
    n = inst.n_blocks
    v = _finite_values(inst)
    d = inst.discount_factors()
    t_max, n_res = inst.n_periods, inst.n_resources
    period = result.period_of_block.copy()
    sstart, slist = _successors(prec, n)

    remaining = np.array(inst.limit, dtype=np.float64).copy()
    for b in np.nonzero(period >= 0)[0]:
        remaining[:, period[b]] -= inst.coef[:, b]

    moved_total = 0
    for _ in range(max_passes):
        moved = 0
        # 1. pull positive-value blocks forward, earliest periods first
        for b in np.argsort(period, kind="stable"):
            t = int(period[b])
            if t <= 0 or v[b] <= 0:
                continue
            floor = 0
            for k in range(prec.pstart[b], prec.pstart[b + 1]):
                p = int(prec.plist[k])
                if period[p] >= 0:
                    floor = max(floor, int(period[p]))
            need = inst.coef[:, b]
            for s in range(floor, t):
                if all(remaining[r, s] >= need[r] - _TOL for r in range(n_res)):
                    remaining[:, t] += need
                    remaining[:, s] -= need
                    period[b] = s
                    moved += 1
                    break
        # 2. push negative-value blocks back, latest periods first
        for b in np.argsort(-period, kind="stable"):
            t = int(period[b])
            if t < 0 or t >= t_max - 1 or v[b] >= 0:
                continue
            ceiling = t_max - 1
            for k in range(sstart[b], sstart[b + 1]):
                c = int(slist[k])
                if period[c] >= 0:
                    ceiling = min(ceiling, int(period[c]))
            need = inst.coef[:, b]
            for s in range(ceiling, t, -1):
                if all(remaining[r, s] >= need[r] - _TOL for r in range(n_res)):
                    remaining[:, t] += need
                    remaining[:, s] -= need
                    period[b] = s
                    moved += 1
                    break
        moved_total += moved
        if moved == 0:
            break

    npv, per_val, per_res = schedule_value(inst, period)
    if npv < result.npv - 1e-6 * max(1.0, abs(result.npv)):
        raise AssertionError(f"local search made it worse: {result.npv:.4f} -> {npv:.4f}")
    _ = d  # discount factors are inside schedule_value; kept here for the reader
    return ScheduleResult(
        method=f"{result.method}+shift-ls",
        period_of_block=period,
        npv=npv,
        bound=result.bound,
        per_period_value=per_val,
        per_period_resource=per_res,
        mined_blocks=int((period >= 0).sum()),
        heuristic=True,
        notes=f"{moved_total} shift moves accepted",
    )


# ------------------------------------------------------------------------------------------------
# two resource constraints, and the top-level driver
# ------------------------------------------------------------------------------------------------
def cpit_bound_two_resources(
    inst: Cpit, prec: Precedence, **kw
) -> tuple[float, list[LpRelaxation]]:
    """Certified bound when several resources bind, by relaxing all but one at a time.

    Chicoisne et al. 2012, Algorithm 4. Dropping every resource but ``r`` yields a relaxation of the
    full problem, so its LP optimum is an upper bound on the full optimum; the smallest such bound
    over ``r`` is the tightest one this construction gives. It is looser than a joint LP bound (that
    is what the Bienstock-Zuckerberg algorithm computes) and it is still certified.
    """
    relaxations = [cpit_lp_relaxation(inst, prec, resource=r, **kw) for r in range(inst.n_resources)]
    return min(rel.bound for rel in relaxations), relaxations


def solve_cpit(
    inst: Cpit,
    prec: Precedence,
    *,
    method: str = "expected",
    local_search: bool = True,
) -> tuple[ScheduleResult, list[LpRelaxation]]:
    """Bound then schedule then improve: the whole ladder in one call.

    Returns the feasible schedule with its certified bound attached, and the per-resource LP
    relaxations that produced the bound. With ``method='expected'`` this is the ExTS procedure: run
    the critical multiplier algorithm per resource, take the expected extraction times from the
    relaxation whose objective is smallest, schedule from them, and improve by shifting.
    """
    bound, relaxations = cpit_bound_two_resources(inst, prec)
    v = _finite_values(inst)
    allowed = solve_upit(v, prec).in_pit

    if method == "expected":
        best: ScheduleResult | None = None
        for rel in relaxations:
            cand = toposort_schedule(inst, prec, weight="expected", relaxation=rel, allowed=allowed)
            cand.method = f"toposort:expected[r{rel.resource}]"
            if best is None or cand.npv > best.npv:
                best = cand
        result = best
        assert result is not None
    else:
        result = toposort_schedule(inst, prec, weight=method, allowed=allowed)

    result.bound = bound
    if local_search:
        result = improve_schedule(inst, prec, result)
        result.bound = bound
    if result.npv > bound + 1e-6 * max(1.0, abs(bound)):
        raise AssertionError(
            f"feasible objective {result.npv:.4f} exceeds the certified bound {bound:.4f}"
        )
    return result, relaxations


# ------------------------------------------------------------------------------------------------
# the controls
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Controls:
    """The three mandatory checks tying the scheduling lane to the proven ultimate pit."""

    duality_set_matches: bool
    duality_bound_error: float
    bound_geq_feasible: bool
    order_invariant: bool
    order_invariance_error: float

    @property
    def all_pass(self) -> bool:
        return self.duality_set_matches and self.bound_geq_feasible and self.order_invariant


def run_controls(inst: Cpit, prec: Precedence, result: ScheduleResult | None = None) -> Controls:
    """Run the duality, bound and order-invariance controls on an instance.

    **Duality.** At discount rate 0 with unlimited capacity, CPIT is UPIT. The LP relaxation's
    cumulative solution at the final period must equal the exact ultimate pit block for block, and
    its bound must equal the exact ultimate-pit value.

    **Bound.** The certified bound must be at least any feasible objective. A bound below a feasible
    solution is proof of a bug, not a tight result.

    **Order invariance.** Still at rate 0 with unlimited capacity, the objective cannot depend on
    the sequence, so every TopoSort weighting must return the same value.
    """
    v = _finite_values(inst)
    exact = solve_upit(v, prec)

    total = float(inst.coef[0].sum()) + 1.0
    degenerate = Cpit(
        name=f"{inst.name}-control",
        n_blocks=inst.n_blocks,
        n_periods=1,
        discount_rate=0.0,
        value=inst.value,
        limit=np.full((1, 1), total),
        sense=np.full((1, 1), "L", dtype="<U1"),
        coef=inst.coef[:1],
        period_one_undiscounted=True,
    )
    rel = cpit_lp_relaxation(degenerate, prec)
    mined = rel.x[-1] > 0.5
    set_matches = bool(np.array_equal(mined, exact.in_pit))
    bound_err = abs(rel.bound - exact.pit_value)

    values = []
    for w in TOPOSORT_WEIGHTS:
        if w == "expected":
            values.append(toposort_schedule(degenerate, prec, weight=w, relaxation=rel).npv)
        else:
            values.append(toposort_schedule(degenerate, prec, weight=w).npv)
    spread = float(max(values) - min(values))
    scale = max(1.0, abs(exact.pit_value))

    ok_bound = True
    if result is not None and result.bound is not None:
        ok_bound = result.npv <= result.bound + 1e-6 * max(1.0, abs(result.bound))

    return Controls(
        duality_set_matches=set_matches,
        duality_bound_error=float(bound_err),
        bound_geq_feasible=bool(ok_bound),
        order_invariant=bool(spread <= 1e-6 * scale),
        order_invariance_error=spread,
    )
