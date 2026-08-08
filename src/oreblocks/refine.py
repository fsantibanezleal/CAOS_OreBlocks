"""Refinement: the exact C-PIT[D] local search, cutoff-grade policy, and operational smoothing.

Three things a schedule needs after a rounding heuristic has produced it, each from a different part
of the literature and each with a different honest status.

**1. The exact C-PIT[D] local search** (Chicoisne, Espinoza, Goycoolea, Moreno and Rubio, Operations
Research 60(3):517-528, 2012, doi:10.1287/opre.1120.1050, section 3.3). Fix every block outside a
small set ``D`` at its incumbent period and re-solve the restricted problem EXACTLY. Their three
neighbourhood constructions, chosen with equal probability:

1. a random scheduled block ``a``, plus a connected subset of its predecessors ``B-(a)``,
2. the same with successors ``B+(a)``,
3. a random scheduled block ``a`` at period ``t``, plus only blocks scheduled in ``t-1, t, t+1``.

This is the rung the shift local search is explicitly NOT. It needs a MILP solver, so it lives behind
the ``oreblocks[milp]`` extra, and it is what closes the last few percent: the authors measured their
heuristic at 0.937 to 0.986 of the LP bound before local search and 0.955 to 0.997 after an hour.

**2. Lane's cutoff-grade policy** (Lane 1988, *The Economic Definition of Ore*). The cutoff that
maximises NPV is not the break-even cutoff: it carries an OPPORTUNITY COST, because processing a
marginal tonne delays everything behind it. The three limiting cutoffs (mine, mill, market) and the
balancing cutoffs between them are computed here from the same economics the schedule uses, so the
number a CPIT instance folds into its block values can be shown next to the number Lane's theory says
it should be.

**3. Minimum mining width** (Bai, Marcotte, Gamache, Gregory and Lapworth, J. S. Afr. Inst. Min.
Metall. 118(5), 2018, doi:10.17159/2411-9717/2018/v118n5a8). Conventional methods produce pushbacks
with "narrow benches and pit bottom, irregular boundaries, and multiple separated components", and
manual post-modification "destroys value and violates resource constraints". The smoothing here is an
adaptive opening: a mined cell whose bench neighbourhood is thinner than the target width is deferred
to the period of its majority neighbour. It COSTS NPV, and reporting that cost is the point: an
operable plan is worth less on paper than an inoperable one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .minelib_models import Cpit
from .precedence import Precedence
from .schedule import (  # noqa: PLC2701
    ScheduleResult,
    _finite_values,
    _successors,
    schedule_value,
)

__all__ = [
    "CutoffPolicy",
    "SmoothingReport",
    "enforce_min_width",
    "exact_local_search",
    "lane_cutoffs",
]


# ------------------------------------------------------------------------------------------------
# 1. the exact C-PIT[D] neighbourhood
# ------------------------------------------------------------------------------------------------
def _neighbourhood(
    rng: np.random.Generator,
    prec: Precedence,
    sstart: np.ndarray,
    slist: np.ndarray,
    period: np.ndarray,
    d_max: int,
) -> np.ndarray:
    scheduled = np.nonzero(period >= 0)[0]
    if scheduled.size == 0:
        return np.zeros(0, dtype=np.int64)
    a = int(rng.choice(scheduled))
    mode = int(rng.integers(0, 3))
    out = {a}

    if mode == 0 or mode == 1:
        frontier = [a]
        while frontier and len(out) < d_max:
            b = frontier.pop()
            nbrs = (
                prec.plist[prec.pstart[b] : prec.pstart[b + 1]]
                if mode == 0
                else slist[sstart[b] : sstart[b + 1]]
            )
            for c in nbrs:
                c = int(c)
                if c not in out and period[c] >= 0:
                    out.add(c)
                    frontier.append(c)
                    if len(out) >= d_max:
                        break
    else:
        t = int(period[a])
        near = np.nonzero((period >= max(0, t - 1)) & (period <= t + 1))[0]
        if near.size > d_max:
            near = rng.choice(near, size=d_max, replace=False)
        out.update(int(v) for v in near)

    return np.array(sorted(out), dtype=np.int64)


def exact_local_search(
    inst: Cpit,
    prec: Precedence,
    result: ScheduleResult,
    *,
    d_max: int = 220,
    rounds: int = 24,
    seed: int = 7,
    time_limit: float = 20.0,
    mip_gap: float = 1e-5,
) -> ScheduleResult:
    """Chicoisne et al. section 3.3: re-solve a restricted C-PIT EXACTLY, repeatedly.

    Every accepted move is a proven improvement of the restricted problem, and the objective is
    monotone by construction because the incumbent is always feasible for the restricted model. Needs
    scipy (``oreblocks[milp]``); without it, use :func:`oreblocks.improve_schedule`, which is a shift
    neighbourhood and a strictly weaker one.
    """
    from scipy.optimize import LinearConstraint, milp
    from scipy.sparse import coo_matrix

    n, t_max, n_res = inst.n_blocks, inst.n_periods, inst.n_resources
    v = _finite_values(inst)
    disc = inst.discount_factors()
    sstart, slist = _successors(prec, n)
    rng = np.random.default_rng(seed)

    period = result.period_of_block.copy()
    accepted = 0

    for _ in range(rounds):
        dset = _neighbourhood(rng, prec, sstart, slist, period, d_max)
        if dset.size < 4:
            continue
        pos = {int(b): i for i, b in enumerate(dset)}
        k = dset.size
        n_var = k * t_max

        # capacity already consumed by the FIXED blocks, per resource and period
        used = np.zeros((n_res, t_max))
        for b in range(n):
            if period[b] >= 0 and int(b) not in pos:
                used[:, period[b]] += inst.coef[:, b]

        # y_it = 1 if the i-th free block is mined by the end of period t (cumulative)
        c = np.zeros(n_var)
        for i, b in enumerate(dset):
            for t in range(t_max):
                nxt = disc[t + 1] if t + 1 < t_max else 0.0
                c[i * t_max + t] = -(v[b] * (disc[t] - nxt))

        rows, cols, data, lo, hi = [], [], [], [], []
        r = 0

        def add(entries, low, high, _rows=rows, _cols=cols, _data=data, _lo=lo, _hi=hi):
            nonlocal r
            for j, val in entries:
                _rows.append(r)
                _cols.append(j)
                _data.append(val)
            _lo.append(low)
            _hi.append(high)
            r += 1

        for i in range(k):
            for t in range(t_max - 1):
                add([(i * t_max + t, 1.0), (i * t_max + t + 1, -1.0)], -np.inf, 0.0)

        for i, b in enumerate(dset):
            for kk in range(prec.pstart[b], prec.pstart[b + 1]):
                a = int(prec.plist[kk])
                if a in pos:
                    j = pos[a]
                    for t in range(t_max):
                        add([(i * t_max + t, 1.0), (j * t_max + t, -1.0)], -np.inf, 0.0)
                else:
                    # a is FIXED: b cannot be mined before a's period, and if a is unmined nor can b
                    pa = int(period[a])
                    if pa < 0:
                        for t in range(t_max):
                            add([(i * t_max + t, 1.0)], -np.inf, 0.0)
                        break
                    for t in range(pa):
                        add([(i * t_max + t, 1.0)], -np.inf, 0.0)
            # successors that are FIXED cap b from above
            for kk in range(sstart[b], sstart[b + 1]):
                sblk = int(slist[kk])
                if sblk in pos or period[sblk] < 0:
                    continue
                ps = int(period[sblk])
                add([(i * t_max + ps, 1.0)], 1.0, np.inf)  # b must be gone by then

        for rr in range(n_res):
            for t in range(t_max):
                entries = []
                for i, b in enumerate(dset):
                    q = float(inst.coef[rr][b])
                    if not q:
                        continue
                    entries.append((i * t_max + t, q))
                    if t > 0:
                        entries.append((i * t_max + t - 1, -q))
                if entries:
                    add(entries, -np.inf, float(inst.limit[rr][t] - used[rr, t]))

        if r == 0:
            continue
        a_mat = coo_matrix((data, (rows, cols)), shape=(r, n_var)).tocsr()
        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(a_mat, np.array(lo), np.array(hi)),
                integrality=np.ones(n_var),
                bounds=(0, 1),
                options={"time_limit": time_limit, "mip_rel_gap": mip_gap, "presolve": True},
            )
        except Exception:  # noqa: BLE001 - a solver failure must never lose the incumbent
            continue
        if not res.success or res.x is None:
            continue

        y = np.asarray(res.x).reshape(k, t_max)
        cand = period.copy()
        for i, b in enumerate(dset):
            hit = np.nonzero(y[i] > 0.5)[0]
            cand[b] = int(hit[0]) if hit.size else -1
        npv_new, _, _ = schedule_value(inst, cand)
        npv_old, _, _ = schedule_value(inst, period)
        if npv_new > npv_old + 1e-9 * max(1.0, abs(npv_old)):
            period = cand
            accepted += 1

    npv, per_val, per_res = schedule_value(inst, period)
    if npv < result.npv - 1e-6 * max(1.0, abs(result.npv)):
        raise AssertionError(f"exact local search made it worse: {result.npv} -> {npv}")
    return ScheduleResult(
        method=f"{result.method}+cpitD-ls",
        period_of_block=period,
        npv=npv,
        bound=result.bound,
        per_period_value=per_val,
        per_period_resource=per_res,
        mined_blocks=int((period >= 0).sum()),
        heuristic=True,
        notes=f"{accepted} of {rounds} exact C-PIT[D] re-solves improved the incumbent",
    )


# ------------------------------------------------------------------------------------------------
# 2. Lane's cutoff-grade policy
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CutoffPolicy:
    """Lane's three limiting cutoffs plus the balancing ones, and the break-even for comparison."""

    break_even: float
    mine_limiting: float
    mill_limiting: float
    market_limiting: float
    balancing_mine_mill: float
    balancing_mill_market: float
    opportunity_cost_per_period: float
    note: str = ""

    @property
    def optimum(self) -> float:
        """Lane's rule: the effective optimum is the MIDDLE value of the three limiting cutoffs."""
        return float(np.median([self.mine_limiting, self.mill_limiting, self.market_limiting]))


def lane_cutoffs(
    *,
    price: float,
    selling_cost: float,
    mining_cost: float,
    processing_cost: float,
    fixed_cost_per_year: float,
    recovery: float,
    mine_capacity: float,
    mill_capacity: float,
    market_capacity: float,
    discount_rate: float,
    remaining_npv: float,
) -> CutoffPolicy:
    """Lane's cutoff grades, computed from the same economics a CPIT instance folds into its values.

    The break-even cutoff makes a tonne of ore pay for its own processing. Lane's insight is that this
    is the WRONG cutoff whenever a capacity binds, because processing a marginal tonne consumes a
    scarce hour and pushes every profitable tonne behind it one step further into the discount. The
    opportunity cost ``F = d * NPV_remaining + fixed`` is what turns break-even into the economic
    cutoff, and it is why the optimum FALLS as a deposit is depleted.

    Every argument is per year and per tonne in consistent units; ``recovery`` is a fraction and
    ``remaining_npv`` is the discounted value still ahead of the operation.
    """
    margin = (price - selling_cost) * recovery
    if margin <= 0:
        raise ValueError("price net of selling cost times recovery must be positive")

    opportunity = discount_rate * remaining_npv + fixed_cost_per_year

    break_even = processing_cost / margin
    # Lane's MINE-limiting cutoff IS the break-even cutoff, and that is a result rather than an
    # oversight: when the shovels are the bottleneck the scarce hour is a MINING hour, and ore and
    # waste consume it identically, so the opportunity cost cancels out of the ore-versus-waste
    # comparison entirely.
    mine_lim = break_even
    mill_lim = (processing_cost + opportunity / max(mill_capacity, 1e-9)) / margin
    market_lim = (processing_cost + 0.0) / max(
        margin - opportunity / max(market_capacity, 1e-9) / max(recovery, 1e-9), 1e-9
    )

    return CutoffPolicy(
        break_even=float(break_even),
        mine_limiting=float(mine_lim),
        mill_limiting=float(mill_lim),
        market_limiting=float(market_lim),
        balancing_mine_mill=float(0.5 * (mine_lim + mill_lim)),
        balancing_mill_market=float(0.5 * (mill_lim + market_lim)),
        opportunity_cost_per_period=float(opportunity),
        note=(
            "Lane 1988. The mine-limiting cutoff carries no processing opportunity cost because the "
            "binding hour is a mining hour; the mill-limiting one carries the whole of it; the "
            "market-limiting one charges it against the margin instead of the cost. The optimum is "
            "the MIDDLE of the three."
        ),
    )


# ------------------------------------------------------------------------------------------------
# 3. minimum mining width
# ------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SmoothingReport:
    moved_blocks: int
    npv_before: float
    npv_after: float
    min_width_before: int
    min_width_after: int
    #: how many mined blocks sit in a run narrower than the target. The MINIMUM is dominated by a
    #: handful of pathological cells that no smoothing can absorb (an isolated block whose neighbours
    #: are all unmined has nothing to vote for it), so the count is the number that actually moves.
    below_target_before: int = 0
    below_target_after: int = 0
    target_width: int = 0

    @property
    def below_target_reduction_pct(self) -> float:
        if self.below_target_before == 0:
            return 0.0
        return 100.0 * (self.below_target_before - self.below_target_after) / self.below_target_before

    @property
    def npv_cost_pct(self) -> float:
        return 0.0 if self.npv_before == 0 else 100.0 * (self.npv_before - self.npv_after) / abs(self.npv_before)


def enforce_min_width(
    inst: Cpit,
    result: ScheduleResult,
    x: np.ndarray,
    y: np.ndarray,
    level: np.ndarray,
    prec: Precedence,
    *,
    target_width: int = 3,
    passes: int = 3,
) -> tuple[ScheduleResult, SmoothingReport]:
    """Adaptive opening: absorb slivers into the period of their majority bench neighbour.

    A mined cell whose run of same-period neighbours along a bench is shorter than ``target_width``
    is a place no shovel can work. It is moved to the period the majority of its 4-neighbours on that
    bench belong to, provided precedence still holds. Capacity is NOT re-imposed, so the result is
    reported as an OPERABILITY view of the plan rather than as a feasible replacement for it, and the
    NPV it costs is the honest price of operability.
    """
    period = result.period_of_block.copy()
    nx = int(x.max()) + 1
    ny = int(y.max()) + 1
    at = {}
    for b in range(period.shape[0]):
        at[(int(x[b]), int(y[b]), int(level[b]))] = b

    def run_len(b: int) -> int:
        best = 10**9
        for axis in (0, 1):
            run = 1
            for step in (-1, 1):
                k = 1
                while True:
                    dx, dy = (step * k, 0) if axis == 0 else (0, step * k)
                    nb = at.get((int(x[b]) + dx, int(y[b]) + dy, int(level[b])))
                    if nb is None or period[nb] != period[b]:
                        break
                    run += 1
                    k += 1
            best = min(best, run)
        return best

    sstart, slist = _successors(prec, period.shape[0])
    npv_before, _, _ = schedule_value(inst, period)
    runs_before = [run_len(int(b)) for b in np.nonzero(period >= 0)[0]]
    width_before = min(runs_before, default=0)
    narrow_before = sum(1 for r in runs_before if r < target_width)
    moved = 0

    for _ in range(passes):
        changed = 0
        for b in np.nonzero(period >= 0)[0]:
            if run_len(int(b)) >= target_width:
                continue
            votes: dict[int, int] = {}
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = at.get((int(x[b]) + dx, int(y[b]) + dy, int(level[b])))
                if nb is None or period[nb] < 0:
                    continue
                votes[int(period[nb])] = votes.get(int(period[nb]), 0) + 1
            if not votes:
                continue
            want = max(votes, key=lambda t: votes[t])
            if want == period[b]:
                continue
            # Precedence cuts BOTH ways. Moving a block earlier can overtake a predecessor; moving
            # it later can be overtaken by a successor already scheduled ahead of the new period.
            # Checking only predecessors passes a plan that mines a block before the rock above it.
            ok = all(
                period[int(p)] <= want
                for p in prec.plist[prec.pstart[b] : prec.pstart[b + 1]]
                if period[int(p)] >= 0
            ) and all(
                period[int(c)] >= want
                for c in slist[sstart[int(b)] : sstart[int(b) + 1]]
                if period[int(c)] >= 0
            )
            if not ok:
                continue
            period[b] = want
            changed += 1
            moved += 1
        if changed == 0:
            break

    npv_after, per_val, per_res = schedule_value(inst, period)
    runs_after = [run_len(int(b)) for b in np.nonzero(period >= 0)[0]]
    width_after = min(runs_after, default=0)
    narrow_after = sum(1 for r in runs_after if r < target_width)
    smoothed = ScheduleResult(
        method=f"{result.method}+min-width({target_width})",
        period_of_block=period,
        npv=npv_after,
        bound=result.bound,
        per_period_value=per_val,
        per_period_resource=per_res,
        mined_blocks=int((period >= 0).sum()),
        heuristic=True,
        notes=f"{moved} slivers absorbed; operability view, capacity not re-imposed",
    )
    _ = (nx, ny)
    return smoothed, SmoothingReport(
        moved_blocks=moved,
        npv_before=float(npv_before),
        npv_after=float(npv_after),
        min_width_before=int(width_before),
        min_width_after=int(width_after),
        below_target_before=int(narrow_before),
        below_target_after=int(narrow_after),
        target_width=int(target_width),
    )
