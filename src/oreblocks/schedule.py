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


class _ParametricPits:
    """The nested family ``x(lambda) = argmax closure of (v - lambda a)``, solved lazily and CACHED.

    The break-points of the family do not depend on the period; only the target capacity ``U_t``
    does. Solving each period from scratch therefore re-derives the same pits ``T`` times, which is
    the single largest cost in the whole algorithm. This cache is shared across periods, and because
    the pits nest (a larger multiplier prices more blocks negative, so the pit can only shrink) every
    new solve is restricted to the smallest already-known pit that must contain it.

    Entries are kept sorted by ``lambda`` ascending, so capacity is non-increasing down the list.
    """

    def __init__(self, values: np.ndarray, coef: np.ndarray, prec: Precedence, full: np.ndarray) -> None:
        self.v = values
        self.a = coef
        self.prec = prec
        self.lams: list[float] = [0.0]
        self.pits: list[np.ndarray] = [full.copy()]
        self.caps: list[float] = [float(coef[full].sum())]
        self.vals: list[float] = [float(values[full].sum())]
        self.solves = 0

    def _insert(self, lam: float, pit: np.ndarray) -> int:
        i = 0
        while i < len(self.lams) and self.lams[i] < lam:
            i += 1
        if i < len(self.lams) and self.lams[i] == lam:
            return i
        self.lams.insert(i, lam)
        self.pits.insert(i, pit)
        self.caps.insert(i, float(self.a[pit].sum()))
        self.vals.insert(i, float(self.v[pit].sum()))
        return i

    def at(self, lam: float) -> int:
        """Index of the (possibly newly solved) entry for this multiplier."""
        for i, existing in enumerate(self.lams):
            if existing == lam:
                return i
        # the pit at lam is contained in the pit at any smaller multiplier: restrict to the nearest
        candidates = self.pits[0]
        for i, existing in enumerate(self.lams):
            if existing < lam:
                candidates = self.pits[i]
            else:
                break
        pit = max_closure_within(self.v - lam * self.a, self.prec, candidates)
        self.solves += 1
        return self._insert(lam, pit)

    def extend_until_below(self, target: float, lam_hint: float) -> None:
        """Make sure at least one known entry has capacity at or below ``target``."""
        lam = max(lam_hint, 1.0)
        while self.caps[-1] > target:
            self.at(lam)
            lam *= 4.0
            if lam > 1e18:
                raise AssertionError("no multiplier empties the pit; check the resource coefficients")

    def _brackets(self, target: float) -> tuple[int, int]:
        u = max(j for j in range(len(self.caps)) if self.caps[j] >= target)
        low = min(j for j in range(len(self.caps)) if self.caps[j] <= target)
        return u, low

    def solve_cp(self, target: float, *, max_refine: int = 80, tol: float = 1e-9) -> tuple[int, int, float, float]:
        """Optimum of ``CP(target) = max v.x`` over closures with ``a.x <= target``.

        Returns ``(u, l, alpha, z)`` where the optimal fractional solution is
        ``alpha * pit[l] + (1 - alpha) * pit[u]`` and ``z`` is its value.

        The stopping rule is the certificate itself, not a guess at how many bisections are enough.
        Strong duality gives ``CP(U) = min_lambda [ UPL(v - lambda a) + lambda U ]``, so the search
        refines until the primal estimate and that dual expression agree. When they do, the two
        bracketing pits are consecutive break-points; when they do not, they are not, whatever the
        interval width says.
        """
        u, low = self._brackets(target)
        for _ in range(max_refine):
            if u >= low:
                return u, u, 1.0, self.vals[u]
            b_u, b_l = self.caps[u], self.caps[low]
            alpha = 1.0 if b_u - b_l <= _TOL else min(1.0, max(0.0, (b_u - target) / (b_u - b_l)))
            z = alpha * self.vals[low] + (1.0 - alpha) * self.vals[u]
            lam = self.lams[low]
            dual = float((self.v - lam * self.a)[self.pits[low]].sum()) + lam * target
            if abs(dual - z) <= tol * max(1.0, abs(z)):
                return u, low, alpha, z
            lo, hi = self.lams[u], self.lams[low]
            if hi - lo <= 1e-13 * max(1.0, hi):
                return u, low, alpha, z
            self.at(0.5 * (lo + hi))
            u, low = self._brackets(target)
        u, low = self._brackets(target)
        b_u, b_l = self.caps[u], self.caps[low]
        alpha = 1.0 if b_u - b_l <= _TOL else min(1.0, max(0.0, (b_u - target) / (b_u - b_l)))
        return u, low, alpha, alpha * self.vals[low] + (1.0 - alpha) * self.vals[u]


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

    full = solve_upit(v, prec)
    a_full = float(a[full.in_pit].sum())
    family = _ParametricPits(v, a, prec, full.in_pit)

    x = np.zeros((t_max, n), dtype=np.float64)
    zval = np.zeros(t_max, dtype=np.float64)
    lam_used = np.zeros(t_max, dtype=np.float64)
    integral = np.zeros(t_max, dtype=bool)

    # a multiplier large enough to empty the pit: beyond max(v/a) every block prices negative
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(a > 0, v / np.where(a > 0, a, 1.0), -np.inf)
    lam_hint = float(max(1.0, np.nanmax(ratio[np.isfinite(ratio)], initial=1.0)))

    for t in range(t_max - 1, -1, -1):
        target = float(caps[t])
        if a_full <= target + _TOL:
            # the whole ultimate pit fits by this period: the LP solution is the integral pit
            x[t] = full.in_pit.astype(np.float64)
            zval[t] = float(v[full.in_pit].sum())
            integral[t] = True
            lam_used[t] = 0.0
            continue

        family.extend_until_below(target, lam_hint)
        u, low, alpha, z = family.solve_cp(target, max_refine=max_bisections)
        pit_u, pit_l = family.pits[u], family.pits[low]
        integral[t] = u == low or alpha in (0.0, 1.0)
        xt = np.where(pit_u, 1.0 - alpha, 0.0)
        xt[pit_l] = 1.0
        x[t] = xt
        zval[t] = z
        lam_used[t] = family.lams[low]

        if check_duality:
            # strong duality: CP(U) = min_lambda [ UPL(p - lambda a) + lambda U ]
            lam = family.lams[low]
            dual = float((v - lam * a)[pit_l].sum()) + lam * target
            scale = max(1.0, abs(zval[t]))
            if abs(dual - zval[t]) > 1e-5 * scale:
                raise AssertionError(
                    f"period {t + 1}: primal {zval[t]:.6f} and dual {dual:.6f} disagree "
                    f"(lambda {lam:.6g}); the critical multiplier did not converge"
                )
    solves = family.solves + 1

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
    bound: bool = True,
) -> tuple[ScheduleResult, list[LpRelaxation]]:
    """Bound then schedule then improve: the whole ladder in one call.

    Returns the feasible schedule with its certified bound attached, and the per-resource LP
    relaxations that produced the bound. With ``method='expected'`` this is the ExTS procedure: run
    the critical multiplier algorithm per resource, take the expected extraction times from the
    relaxation whose objective is smallest, schedule from them, and improve by shifting.

    ``bound=False`` SKIPS the certified bound and returns a schedule with ``bound=None``. The bound
    costs a parametric family of maximum closures per resource, hundreds of them, while a schedule
    from a combinatorial weight costs one closure; a caller that needs many schedules and no bound
    pays a factor of a hundred for a number it discards. The uncertainty ensemble is exactly that
    caller: it re-solves once per realisation and compares NPVs, and it never reads the bound. Left
    on by default, this turned a thirteen-case bake into a four-hour run that finished one case.
    ``method='expected'`` needs the relaxation by definition and is rejected here rather than
    silently downgraded.
    """
    if not bound and method == "expected":
        raise ValueError("method='expected' needs the LP relaxation, so it needs bound=True")
    if bound:
        certified, relaxations = cpit_bound_two_resources(inst, prec)
    else:
        certified, relaxations = None, []
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

    result.bound = certified
    if local_search:
        result = improve_schedule(inst, prec, result)
        result.bound = certified
    if certified is not None and result.npv > certified + 1e-6 * max(1.0, abs(certified)):
        raise AssertionError(
            f"feasible objective {result.npv:.4f} exceeds the certified bound {certified:.4f}"
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


# ------------------------------------------------------------------------------------------------
# the joint bound: CPIT as a General Precedence Constrained Problem, solved by Bienstock-Zuckerberg
# ------------------------------------------------------------------------------------------------
def cpit_to_gpcp(inst: Cpit, prec: Precedence):
    """Time-expand CPIT into the GPCP form the Bienstock-Zuckerberg algorithm solves.

    The cumulative variable ``x_bt`` becomes node ``b * T + t``. Three families of arcs and one family
    of side constraints, all transcribed from Chicoisne et al. 2012 equations (3a)-(3f):

    - **precedence in every period**  ``x_bt <= x_at`` for each arc and each ``t``
    - **monotonicity**                ``x_bt <= x_b,t+1``
    - **capacity**  ``sum_b a_rb (x_bt - x_b,t-1) <= c_rt``, which in the z-space has a POSITIVE
      coefficient on ``(b, t)`` and a NEGATIVE one on ``(b, t-1)``. Side rows with negative entries
      are exactly why this needs BZ rather than a second parametric closure.

    The objective in the cumulative variables is ``sum_t gamma_t (x_t . v)`` by Abel summation, the
    same identity the critical multiplier algorithm uses.
    """
    from .bz import Gpcp

    n, t_max = inst.n_blocks, inst.n_periods
    v = _finite_values(inst)
    g_w = inst.gamma()

    c = np.empty(n * t_max, dtype=np.float64)
    for t in range(t_max):
        c[t::t_max] = g_w[t] * v

    pf: list[np.ndarray] = []
    pt: list[np.ndarray] = []
    owner = np.repeat(np.arange(n, dtype=np.int64), np.diff(prec.pstart))
    for t in range(t_max):
        pf.append(owner * t_max + t)
        pt.append(prec.plist.astype(np.int64) * t_max + t)
    blocks = np.arange(n, dtype=np.int64)
    for t in range(t_max - 1):
        pf.append(blocks * t_max + t)
        pt.append(blocks * t_max + t + 1)

    h_rows: list[tuple[np.ndarray, np.ndarray]] = []
    h_rhs: list[float] = []
    for r in range(inst.n_resources):
        a = np.asarray(inst.coef[r], dtype=np.float64)
        nz = np.nonzero(a)[0]
        for t in range(t_max):
            idx = nz * t_max + t
            coef = a[nz]
            if t > 0:
                idx = np.concatenate([idx, nz * t_max + (t - 1)])
                coef = np.concatenate([coef, -a[nz]])
            h_rows.append((idx.astype(np.int64), coef))
            h_rhs.append(float(inst.limit[r][t]))

    return Gpcp(
        n=n * t_max,
        c=c,
        prec_from=np.concatenate(pf).astype(np.int64),
        prec_to=np.concatenate(pt).astype(np.int64),
        h_rows=h_rows,
        h_rhs=np.array(h_rhs, dtype=np.float64),
    )


def cpit_bz_bound(inst: Cpit, prec: Precedence, **kw):
    """The JOINT LP bound over all resources at once, by Bienstock-Zuckerberg.

    Use it where :func:`cpit_bound_two_resources` is loose. That function relaxes all but one
    resource and keeps the smallest of the resulting bounds, which is certified and, on an instance
    where two capacities both bind, materially weaker. The difference between the two numbers is the
    part of a reported gap that belongs to the BOUND rather than to the heuristic, and separating
    those is the only reason to run this.

    Needs scipy for the restricted master. Install ``oreblocks[milp]``.
    """
    from .bz import solve_gpcp_lp

    return solve_gpcp_lp(cpit_to_gpcp(inst, prec), **kw)


# ------------------------------------------------------------------------------------------------
# the sliding time window heuristic, which is what industry actually runs
# ------------------------------------------------------------------------------------------------
def sliding_window_schedule(
    inst: Cpit,
    prec: Precedence,
    *,
    window: int = 3,
    fix: int = 1,
    allowed: np.ndarray | None = None,
    relaxation: LpRelaxation | None = None,
    cand_max: int = 600,
    cover: float = 1.6,
    mip_gap: float = 1e-3,
    time_limit: float | None = None,
) -> ScheduleResult:
    """Cullenbine, Wood and Newman, Optimization Letters, 2011, doi:10.1007/s11590-011-0306-2.

    Enforce every constraint inside a window of ``window`` periods, aggregate everything after it
    into a single tail, fix the first ``fix`` periods of the answer, slide, repeat.

    THE WINDOW HAS TO MATTER, and in the first implementation it did not. That version scheduled each
    window with the same greedy TopoSort the SOTA rung uses, then undid every placement past the
    frozen prefix and returned its capacity. Since a placement consumes only its own period's
    capacity, nothing inside the window could influence the frozen prefix, and the answer came out
    bit-identical for ``window`` of 1, 2, 3, 5, 8 and T on every instance tried: zero blocks moved.
    The rung was a plain one-period-at-a-time greedy carrying a citation for a look-ahead method.

    What makes this a look-ahead is that the sub-problem is solved JOINTLY over the window with the
    rest of the horizon present as an aggregated tail, so a block worth taking in period 1 only
    because of what it unlocks in period 3 is visible to the solver. That needs an integer program
    per slide, so this rung needs scipy (``oreblocks[milp]``).

    The variables are CUMULATIVE, ``y[i][j] = 1`` when block ``i`` is mined by the end of slot ``j``,
    for the same reason the exact local search uses them: precedence becomes one row of two entries
    per arc per slot instead of a prefix sum, and the objective follows by Abel summation. Written
    with per-slot assignment variables first, the constraint matrix grew quadratically in the window
    and one slide took ninety seconds.

    Two approximations, both stated rather than buried:

    - **The tail is optimistic.** Periods at or beyond ``start + window`` collapse into one pseudo
      period whose capacity is their total and whose discount factor is that of the FIRST of them.
      That over-values tail production, which is the standard relaxation for this heuristic: it keeps
      the tail from dominating the window while still making the window aware a horizon exists.
    - **The candidate set is sized by TONNAGE**, not by a constant: the frontier plus the best of
      the undecided pool by value density, taken until the cumulative extraction resource covers
      ``cover`` times the window's own capacity, and never more than ``cand_max``. A flat cap is
      the wrong shape here, and measurably so: 150 blocks starved the sub-problem and cut the
      objective from 39.7 M to 10.4 M on a 1008-block twin, because the frontier alone could not
      reach a period's limit. The published method solves the full model per window; a pure-Python
      caller cannot, and a cap that is named beats a horizon that is silently one period.
    """
    from scipy.optimize import LinearConstraint, milp
    from scipy.sparse import coo_matrix

    n, t_max, n_res = inst.n_blocks, inst.n_periods, inst.n_resources
    v = _finite_values(inst)
    disc = inst.discount_factors()
    coef = np.asarray(inst.coef, dtype=np.float64).reshape(n_res, n)
    limit = np.asarray(inst.limit, dtype=np.float64).reshape(n_res, t_max)

    if allowed is None:
        allowed = solve_upit(v, prec).in_pit
    allowed = np.asarray(allowed, dtype=bool)

    window = max(1, int(window))
    fix = max(1, min(int(fix), window))

    period = np.full(n, -1, dtype=np.int64)
    decided = np.zeros(n, dtype=bool)
    used = np.zeros((n_res, t_max))
    density = np.where(coef[0] > 0, v / np.maximum(coef[0], 1e-9), v)

    start = 0
    while start < t_max:
        stop = min(start + window, t_max)
        has_tail = stop < t_max
        n_slot = (stop - start) + (1 if has_tail else 0)

        pool = np.nonzero(allowed & ~decided)[0]
        if pool.size == 0:
            break

        undecided = ~decided
        frontier = [
            int(b)
            for b in pool
            if not np.any(allowed[prec.plist[prec.pstart[b] : prec.pstart[b + 1]]]
                          & undecided[prec.plist[prec.pstart[b] : prec.pstart[b + 1]]])
        ]
        # Size the candidate set by TONNAGE, not by a constant. It has to be able to FILL the
        # window's capacity or the sub-problem is starved and the slide mines almost nothing: a flat
        # cap of 150 blocks cut the objective from 39.7 M to 10.4 M on a 1008-block twin, because the
        # frontier alone could not reach a period's limit. Take the best of the pool by value density
        # until the cumulative extraction resource covers `cover` windows' worth, then stop.
        if pool.size > cand_max:
            order = pool[np.argsort(-density[pool])]
            need = cover * float(limit[0, start:stop].sum() + (limit[0, stop:].sum() if has_tail else 0.0))
            take = int(np.searchsorted(np.cumsum(coef[0, order]), need) + 1)
            want = int(min(max(take, len(frontier)), order.size))
            if want > cand_max:
                # REFUSE rather than starve. Capping here silently returns a schedule that mined 550
                # blocks of 14,400 because the sub-problem could never reach a period's limit, and a
                # starved answer that still looks like a schedule is exactly the failure this method
                # was rewritten to remove. The caller decides: raise the cap and pay, or skip the rung
                # and say so.
                raise ValueError(
                    f"sliding window needs {want} candidate blocks to fill the capacity of periods "
                    f"{start}..{stop} and cand_max is {cand_max}. Raise cand_max (the sub-problem is "
                    f"a MILP over cand_max x {n_slot} binaries) or skip this rung"
                )
            take = want
            cand = np.unique(np.concatenate([np.asarray(frontier, dtype=np.int64), order[:take]]))
        else:
            cand = pool
        if cand.size == 0:
            start += fix
            continue

        pos = {int(b): i for i, b in enumerate(cand)}
        k = int(cand.size)
        n_var = k * n_slot

        def slot_period(j: int, _start=start, _stop=stop) -> int:
            """The real period a slot stands for; the tail answers with its FIRST period."""
            return _start + j if _start + j < _stop else _stop

        # Abel summation over the cumulative variables: mining at slot j is worth
        # v * disc[period(j)], and a block mined by j but not by j-1 was mined AT j.
        c = np.zeros(n_var)
        for i, b in enumerate(cand):
            for j in range(n_slot):
                nxt = disc[slot_period(j + 1)] if j + 1 < n_slot else 0.0
                c[i * n_slot + j] = -(v[b] * (disc[slot_period(j)] - nxt))

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        lo: list[float] = []
        hi: list[float] = []
        r = 0

        def add(entries, low, high, _rows=rows, _cols=cols, _data=data, _lo=lo, _hi=hi):
            nonlocal r
            for jj, val in entries:
                _rows.append(r)
                _cols.append(jj)
                _data.append(val)
            _lo.append(low)
            _hi.append(high)
            r += 1

        # cumulative: y[i][j] <= y[i][j+1]
        for i in range(k):
            for j in range(n_slot - 1):
                add([(i * n_slot + j, 1.0), (i * n_slot + j + 1, -1.0)], -np.inf, 0.0)

        for i, b in enumerate(cand):
            blocked = False
            earliest = start
            for kk in range(prec.pstart[b], prec.pstart[b + 1]):
                a = int(prec.plist[kk])
                if not allowed[a]:
                    continue
                if a in pos:
                    j0 = pos[a]
                    for j in range(n_slot):
                        add([(i * n_slot + j, 1.0), (j0 * n_slot + j, -1.0)], -np.inf, 0.0)
                elif decided[a]:
                    earliest = max(earliest, int(period[a]))
                else:
                    blocked = True
                    break
            if blocked:
                add([(i * n_slot + n_slot - 1, 1.0)], -np.inf, 0.0)
                continue
            for j in range(n_slot):
                if slot_period(j) < earliest:
                    add([(i * n_slot + j, 1.0)], -np.inf, 0.0)

        # capacity: what is mined AT slot j is y[.][j] - y[.][j-1]
        for rr in range(n_res):
            for j in range(n_slot):
                t = slot_period(j)
                cap = (
                    float(limit[rr, stop:].sum())
                    if (has_tail and j == n_slot - 1)
                    else float(limit[rr, t] - used[rr, t])
                )
                entries = []
                for i, b in enumerate(cand):
                    a_rb = float(coef[rr, b])
                    if a_rb == 0.0:
                        continue
                    entries.append((i * n_slot + j, a_rb))
                    if j > 0:
                        entries.append((i * n_slot + j - 1, -a_rb))
                if entries:
                    add(entries, -np.inf, max(0.0, cap))

        a_mat = coo_matrix((data, (rows, cols)), shape=(r, n_var)).tocsr()
        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(a_mat, np.array(lo), np.array(hi)),
                integrality=np.ones(n_var),
                bounds=(0, 1),
                options=_window_options(time_limit, mip_gap),
            )
        except Exception:  # noqa: BLE001 - a failed slide must not lose the schedule so far
            res = None

        if res is not None and res.success and res.x is not None:
            y = np.asarray(res.x).reshape(k, n_slot) > 0.5
            for i, b in enumerate(cand):
                hit = np.nonzero(y[i])[0]
                if hit.size == 0:
                    continue
                j = int(hit[0])
                t = start + j
                # FIX only the frozen prefix; everything else is reconsidered on the next slide
                if j < (stop - start) and t < start + fix:
                    period[int(b)] = t
                    decided[int(b)] = True
                    used[:, t] += coef[:, int(b)]

        start += fix

    npv, per_val, per_res = schedule_value(inst, period)
    return ScheduleResult(
        method=f"sliding-window(w={window},f={fix})",
        period_of_block=period,
        npv=npv,
        per_period_value=per_val,
        per_period_resource=per_res,
        mined_blocks=int((period >= 0).sum()),
        heuristic=True,
        notes=(
            f"window {window}, {fix} period(s) fixed per slide, horizon beyond the window aggregated "
            f"into one optimistic tail; candidate set capped at {cand_max} blocks per slide"
        ),
    )


def _window_options(time_limit: float | None, mip_gap: float) -> dict:
    """Solver options, with the wall clock omitted when the caller wants a reproducible bake."""
    options: dict = {"mip_rel_gap": mip_gap, "presolve": True}
    if time_limit is not None:
        options["time_limit"] = time_limit
    return options
