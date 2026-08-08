"""PCPSP and OPBSP: when the model CHOOSES where each block goes.

CPIT fixes the destination before it runs, folded into a single net value per block. That is the
reduction the whole industry performs when it computes a cutoff grade ahead of scheduling, and it is
what makes CPIT a *different problem*, not a simplification of the same one. Jelvez, Morales and
Nancel-Penard, MPES 2018, doi:10.1007/978-3-319-99220-4_18, state it exactly:

    "The Precedence Constrained Production Scheduling Problem (PCPSP) extends the last one mainly by
    considering multiple possible destinations for the blocks (therefore the model decides which one
    is the optimal choice) and respecting general side constraints, such as blending."

Two things live here.

**OPBSP**, the fully binary formulation of the same problem (Jelvez et al. section 2.2): a block goes
to exactly one destination, so ``x_bdt`` is binary rather than fractional. Their equations (3)-(10).
OPBSP solutions are feasible for PCPSP, so an OPBSP objective can be scored against a published PCPSP
bound. Solved exactly here with HiGHS through ``scipy.optimize.milp`` for instances small enough, and
by a destination-aware TopoSort otherwise.

**The destination-aware heuristic**, which is the interesting one, because it makes the cutoff grade
an OUTPUT. A block is sent to the plant only while plant capacity remains and the plant is worth more
than the dump for that block; when the plant is full the same block goes to waste, so the effective
cutoff moves with the schedule instead of being a number decided in advance. That is visible on
screen as blocks changing destination when the mill capacity slider moves, which is the whole reason
the distinction matters to a viewer.

What is NOT here: a stockpile. An inventory whose reclaimed grade is the blend of what is inside makes
the model bilinear; the published linear models fix that grade as a parameter and search over it
(Rezakhah, Moreno and Newman, doi:10.1016/j.cor.2019.02.001), and its value is fragile, falling 37 and
69 percent at 5 and 10 percent annual degradation (Rezakhah and Newman,
doi:10.1016/j.cor.2018.11.009). Blending and other general side constraints are read from a ``.pcpsp``
file and not solved; an instance that declares them is not being solved as posed, and that is stated
rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .minelib_models import Pcpsp
from .precedence import Precedence
from .schedule import _finite_values, toposort_order  # noqa: PLC2701
from .upit import solve_upit

__all__ = ["DestinationSchedule", "destination_toposort", "solve_opbsp_exact"]


@dataclass
class DestinationSchedule:
    """A schedule that also decides WHERE each block goes."""

    method: str
    period_of_block: np.ndarray  # int64 (n,), -1 = never mined
    destination_of_block: np.ndarray  # int64 (n,), -1 = never mined
    npv: float
    mined_blocks: int
    exact: bool
    #: the effective cutoff that came OUT of the schedule, per period: the lowest grade sent to the
    #: plant in that period. An input in CPIT, a result here.
    effective_cutoff: np.ndarray
    notes: str = ""


def _discount(inst: Pcpsp) -> np.ndarray:
    expo = np.arange(inst.n_periods, dtype=np.float64)
    if not inst.period_one_undiscounted:
        expo = expo + 1.0
    return 1.0 / (1.0 + inst.discount_rate) ** expo


def destination_toposort(
    inst: Pcpsp,
    prec: Precedence,
    *,
    grade: np.ndarray | None = None,
    weight: np.ndarray | None = None,
) -> DestinationSchedule:
    """Destination-aware TopoSort: the cutoff grade becomes an output of the schedule.

    Walk a weighted topological ordering as in CPIT, but at each block choose the destination that
    maximises discounted value among those whose remaining resources still fit. When the plant is
    full the block falls to the dump, so the effective cutoff rises in exactly the periods where
    processing binds. Feasibility is by construction for the resource rows; general side constraints
    (blending) are NOT enforced here.
    """
    n, t_max, n_dest = inst.n_blocks, inst.n_periods, inst.n_destinations
    d = _discount(inst)
    _, best_val = inst.best_destination()
    cpit_like = inst.to_cpit()
    allowed = solve_upit(_finite_values(cpit_like), prec).in_pit

    if weight is None:
        weight = best_val.astype(np.float64)
    order = toposort_order(prec, weight, allowed)

    remaining = np.array(inst.limit, dtype=np.float64).copy()
    period = np.full(n, -1, dtype=np.int64)
    dest = np.full(n, -1, dtype=np.int64)
    npv = 0.0

    for b in order:
        earliest, blocked = 0, False
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

        placed, chosen, best = -1, -1, -np.inf
        for t in range(earliest, t_max):
            for dd in range(n_dest):
                if inst.forbidden[b, dd]:
                    continue
                need = [inst.coef[r][b, dd] for r in range(inst.n_resources)]
                if any(remaining[r, t] < need[r] - 1e-9 for r in range(inst.n_resources)):
                    continue
                val = d[t] * float(inst.value[b, dd])
                if val > best:
                    best, placed, chosen = val, t, dd
            if placed >= 0:
                break  # earliest feasible period wins; within it the best destination
        if placed < 0:
            continue
        period[b], dest[b] = placed, chosen
        npv += best
        for r in range(inst.n_resources):
            remaining[r, placed] -= inst.coef[r][b, chosen]

    # the effective cutoff: the lowest grade actually sent to a processing destination per period
    cutoff = np.full(t_max, np.nan)
    if grade is not None:
        proc = int(np.argmax([inst.coef[min(1, inst.n_resources - 1)][:, dd].sum() for dd in range(n_dest)]))
        for t in range(t_max):
            sel = (period == t) & (dest == proc)
            if sel.any():
                cutoff[t] = float(grade[sel].min())

    return DestinationSchedule(
        method="pcpsp-destination-toposort",
        period_of_block=period,
        destination_of_block=dest,
        npv=float(npv),
        mined_blocks=int((period >= 0).sum()),
        exact=False,
        effective_cutoff=cutoff,
        notes="destination chosen per block against the remaining capacity; the cutoff is an output",
    )


def solve_opbsp_exact(
    inst: Pcpsp,
    prec: Precedence,
    *,
    max_variables: int = 40_000,
    time_limit: float = 300.0,
    mip_gap: float = 1e-4,
    grade: np.ndarray | None = None,
) -> DestinationSchedule | None:
    """OPBSP solved EXACTLY as a mixed-integer program (Jelvez et al. 2018, equations 3-10).

    Returns ``None`` when the instance is larger than ``max_variables`` rather than pretending a
    heuristic answer is an exact one. Needs scipy (``oreblocks[milp]``).

    Variables ``z_bdt in {0,1}``: block ``b`` is extracted in period ``t`` and sent to destination
    ``d``. One binary family carries both decisions, so nothing is linearised and nothing is
    approximated::

        max   sum_{b,d,t}  discount[t] * value[b,d] * z_bdt
        s.t.  sum_{d,t} z_bdt <= 1                                          (mine a block at most once)
              sum_{d,s<=t} z_bds  <=  sum_{d,s<=t} z_ads   for (a pred of b), all t   (precedence)
              sum_{b,d} coef[r][b,d] * z_bdt <= limit[r][t]                 (capacity)
              z_bdt = 0 wherever the destination is forbidden

    The precedence row is the cumulative form: "by period t, b has been mined at most as much as a
    has", which is exactly ``x_bt <= x_at`` written over the destination-indexed binaries.
    """
    from scipy.optimize import LinearConstraint, milp
    from scipy.sparse import coo_matrix

    n, t_max, n_dest = inst.n_blocks, inst.n_periods, inst.n_destinations
    n_var = n * n_dest * t_max
    if n_var > max_variables:
        return None

    d = _discount(inst)
    vid = lambda b, dd, t: (b * n_dest + dd) * t_max + t  # noqa: E731

    c = np.zeros(n_var)
    forbidden = np.zeros(n_var, dtype=bool)
    for b in range(n):
        for dd in range(n_dest):
            bad = bool(inst.forbidden[b, dd])
            for t in range(t_max):
                j = vid(b, dd, t)
                c[j] = -(d[t] * float(inst.value[b, dd])) if not bad else 0.0
                forbidden[j] = bad

    rows, cols, data, lo, hi = [], [], [], [], []
    r = 0

    def add(entries, low, high, _rows=rows, _cols=cols, _data=data, _lo=lo, _hi=hi):
        nonlocal r
        for j, v in entries:
            _rows.append(r)
            _cols.append(j)
            _data.append(v)
        _lo.append(low)
        _hi.append(high)
        r += 1

    for b in range(n):
        add([(vid(b, dd, t), 1.0) for dd in range(n_dest) for t in range(t_max)], -np.inf, 1.0)

    for b in range(n):
        for k in range(prec.pstart[b], prec.pstart[b + 1]):
            a = int(prec.plist[k])
            for t in range(t_max):
                entries = [(vid(b, dd, s), 1.0) for dd in range(n_dest) for s in range(t + 1)]
                entries += [(vid(a, dd, s), -1.0) for dd in range(n_dest) for s in range(t + 1)]
                add(entries, -np.inf, 0.0)

    for rr in range(inst.n_resources):
        coef = inst.coef[rr]
        for t in range(t_max):
            entries = []
            for b in range(n):
                for dd in range(n_dest):
                    q = float(coef[b, dd])
                    if q:
                        entries.append((vid(b, dd, t), q))
            if entries:
                add(entries, -np.inf, float(inst.limit[rr][t]))

    a_mat = coo_matrix((data, (rows, cols)), shape=(r, n_var)).tocsr()
    ub = np.where(forbidden, 0.0, 1.0)
    res = milp(
        c=c,
        constraints=LinearConstraint(a_mat, np.array(lo), np.array(hi)),
        integrality=np.ones(n_var),
        bounds=(np.zeros(n_var), ub),
        options={"time_limit": time_limit, "mip_rel_gap": mip_gap, "presolve": True},
    )
    if not res.success or res.x is None:
        return None

    x = np.asarray(res.x)
    period = np.full(n, -1, dtype=np.int64)
    dest = np.full(n, -1, dtype=np.int64)
    npv = 0.0
    for b in range(n):
        for dd in range(n_dest):
            for t in range(t_max):
                if x[vid(b, dd, t)] > 0.5:
                    period[b], dest[b] = t, dd
                    npv += d[t] * float(inst.value[b, dd])
                    break
            if period[b] >= 0:
                break

    cutoff = np.full(t_max, np.nan)
    if grade is not None and n_dest > 1:
        for t in range(t_max):
            sel = (period == t) & (dest == 1)
            if sel.any():
                cutoff[t] = float(grade[sel].min())

    return DestinationSchedule(
        method="opbsp-exact(milp)",
        period_of_block=period,
        destination_of_block=dest,
        npv=float(npv),
        mined_blocks=int((period >= 0).sum()),
        exact=True,
        effective_cutoff=cutoff,
        notes=f"HiGHS MILP, {n_var} binaries, {r} rows, mip_rel_gap {mip_gap}",
    )
