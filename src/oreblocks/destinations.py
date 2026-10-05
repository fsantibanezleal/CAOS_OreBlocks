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

from dataclasses import dataclass, replace

import numpy as np

from .minelib_models import Pcpsp
from .precedence import Precedence
from .schedule import _finite_values, toposort_order  # noqa: PLC2701
from .upit import solve_upit

__all__ = [
    "DestinationSchedule",
    "PcpspBound",
    "destination_toposort",
    "exact_destination_local_search",
    "lift_to_pcpsp",
    "pcpsp_lagrangian_bound",
    "pcpsp_lp_bound",
    "pcpsp_schedule_value",
    "restrict_destinations",
    "solve_opbsp_exact",
]


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


def _processing_destination(inst: Pcpsp) -> int:
    """The destination that consumes the processing resource: the plant.

    With one resource there is no processing row to read, and destination 1 is taken as the plant,
    which is the MineLib convention for a two-destination file (0 waste, 1 process).
    """
    if inst.n_resources < 2:
        return min(1, inst.n_destinations - 1)
    row = min(1, inst.n_resources - 1)
    return int(np.argmax([float(inst.coef[row][:, dd].sum()) for dd in range(inst.n_destinations)]))


def _effective_cutoff(inst: Pcpsp, period: np.ndarray, dest: np.ndarray, grade: np.ndarray | None) -> np.ndarray:
    cutoff = np.full(inst.n_periods, np.nan)
    if grade is None:
        return cutoff
    proc = _processing_destination(inst)
    for t in range(inst.n_periods):
        sel = (period == t) & (dest == proc)
        if sel.any():
            cutoff[t] = float(np.asarray(grade)[sel].min())
    return cutoff


def pcpsp_schedule_value(inst: Pcpsp, period: np.ndarray, dest: np.ndarray) -> tuple[float, np.ndarray]:
    """Discounted value of a destination schedule, and its resource use per resource and period.

    Raises if a mined block has no destination or a forbidden one: a plan that does either is not a
    plan, and summing a forbidden sentinel value is how a 5e19 lands in an NPV.
    """
    d = _discount(inst)
    use = np.zeros((inst.n_resources, inst.n_periods))
    npv = 0.0
    for b in np.nonzero(period >= 0)[0]:
        dd = int(dest[b])
        if dd < 0 or inst.forbidden[b, dd]:
            raise ValueError(f"block {b} is mined with an invalid destination {dd}")
        t = int(period[b])
        npv += d[t] * float(inst.value[b, dd])
        use[:, t] += inst.coef[:, b, dd]
    return float(npv), use


def _check_feasible(inst: Pcpsp, prec: Precedence, period: np.ndarray, dest: np.ndarray) -> None:
    _, use = pcpsp_schedule_value(inst, period, dest)
    limit = np.asarray(inst.limit, dtype=np.float64)
    if (use > limit * (1 + 1e-9) + 1e-6).any():
        raise AssertionError("destination schedule exceeds a capacity")
    for b in np.nonzero(period >= 0)[0]:
        for k in range(prec.pstart[b], prec.pstart[b + 1]):
            a = int(prec.plist[k])
            if period[a] < 0 or period[a] > period[b]:
                raise AssertionError(f"block {b} is mined before its predecessor {a}")


def destination_toposort(
    inst: Pcpsp,
    prec: Precedence,
    *,
    grade: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    allowed: np.ndarray | None = None,
) -> DestinationSchedule:
    """Destination-aware TopoSort: the cutoff grade becomes an output of the schedule.

    Walk a weighted topological ordering as in CPIT. For each destination of a block find the
    EARLIEST period, not before any predecessor, whose remaining resources take the block at that
    destination; then keep the destination whose discounted value at its own earliest period is the
    largest. An ore block therefore WAITS for the plant when the plant is worth more a period later
    than the dump is now, and falls to the dump when it is not, so the effective cutoff moves with
    the schedule instead of being a number decided in advance. Feasibility for the resource rows is
    by construction; general side constraints (blending) are NOT enforced here.

    Pass the LP expected extraction times as ``weight=-E_b`` for the ExTS ordering. The default is
    the best-destination value, which makes this greedy TopoSort with a destination choice; before
    0.6.0 that default, together with a rule that took the first period where ANY destination fitted
    (so ore went to the dump the moment the plant was full), produced plans below the
    fixed-destination CPIT plan on every case a downstream product tried, although choosing the
    destination is the richer problem.
    """
    n, t_max, n_dest = inst.n_blocks, inst.n_periods, inst.n_destinations
    d = _discount(inst)
    _, best_val = inst.best_destination()
    if allowed is None:
        cpit_like = inst.to_cpit()
        allowed = solve_upit(_finite_values(cpit_like), prec).in_pit

    if weight is None:
        weight = best_val.astype(np.float64)
    order = toposort_order(prec, weight, allowed)

    remaining = np.array(inst.limit, dtype=np.float64).copy()
    period = np.full(n, -1, dtype=np.int64)
    dest = np.full(n, -1, dtype=np.int64)

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
        for dd in range(n_dest):
            if inst.forbidden[b, dd]:
                continue
            need = inst.coef[:, b, dd]
            for t in range(earliest, t_max):
                if all(remaining[r, t] >= need[r] - 1e-9 for r in range(inst.n_resources)):
                    val = d[t] * float(inst.value[b, dd])
                    if val > best:
                        best, placed, chosen = val, t, dd
                    break  # the earliest period for THIS destination
        if placed < 0:
            continue
        period[b], dest[b] = placed, chosen
        remaining[:, placed] -= inst.coef[:, b, chosen]

    npv, _ = pcpsp_schedule_value(inst, period, dest)
    return DestinationSchedule(
        method="pcpsp-destination-toposort",
        period_of_block=period,
        destination_of_block=dest,
        npv=float(npv),
        mined_blocks=int((period >= 0).sum()),
        exact=False,
        effective_cutoff=_effective_cutoff(inst, period, dest, grade),
        notes="per destination the earliest period that fits, then the best discounted destination",
    )


def lift_to_pcpsp(
    inst: Pcpsp, prec: Precedence, period_of_block: np.ndarray, *, grade: np.ndarray | None = None
) -> DestinationSchedule:
    """A fixed-destination (CPIT) plan read as a PCPSP plan: every mined block at its best destination.

    ``Pcpsp.to_cpit`` builds the CPIT instance from exactly these destinations, values and resource
    coefficients, so a plan feasible for that CPIT is feasible here with the same objective. A PCPSP
    method that starts from it cannot end below the CPIT plan, which is the property the richer
    problem has to show.
    """
    best_dest, _ = inst.best_destination()
    period = np.asarray(period_of_block, dtype=np.int64).copy()
    dest = np.where(period >= 0, best_dest, -1).astype(np.int64)
    _check_feasible(inst, prec, period, dest)
    npv, _ = pcpsp_schedule_value(inst, period, dest)
    return DestinationSchedule(
        method="cpit-lifted",
        period_of_block=period,
        destination_of_block=dest,
        npv=npv,
        mined_blocks=int((period >= 0).sum()),
        exact=False,
        effective_cutoff=_effective_cutoff(inst, period, dest, grade),
        notes="a fixed-destination plan with every block at its a-priori best destination",
    )


def restrict_destinations(inst: Pcpsp, dest: np.ndarray) -> Pcpsp:
    """The same instance with each block allowed only the destination ``dest`` names (-1: all kept).

    With ``dest = PcpspBound.preferred_destination()`` this is the RE-CUT: the cutoff the PCPSP
    relaxation chooses, fixed, after which ``to_cpit()`` gives a CPIT whose plans are PCPSP plans of
    the original instance with the same value. That CPIT is then scheduled by the full CPIT machinery
    (its own critical-multiplier relaxations, ExTS, the sliding window). It is the cutoff-then-schedule
    sequence of practice, with the cutoff taken from the relaxation instead of decided in advance:
    the relaxation knows the opportunity cost of the plant, and a comparison of a block's two values
    does not (a marginal ore block's plant value is positive and its dump value negative, so that
    comparison always sends it to the plant, even when the plant tonnage it takes is worth more to the
    richer ore below it).
    """
    dest = np.asarray(dest, dtype=np.int64)
    if dest.shape != (inst.n_blocks,):
        raise ValueError(f"need one destination per block, got shape {dest.shape}")
    forbidden = inst.forbidden.copy()
    chosen = np.nonzero(dest >= 0)[0]
    if (dest[chosen] >= inst.n_destinations).any() or inst.forbidden[chosen, dest[chosen]].any():
        raise ValueError("a block is restricted to a destination it may not use")
    forbidden[chosen, :] = True
    forbidden[chosen, dest[chosen]] = False
    return replace(inst, name=f"{inst.name}-recut", forbidden=forbidden)


def exact_destination_local_search(
    inst: Pcpsp,
    prec: Precedence,
    start: DestinationSchedule,
    *,
    d_max: int = 160,
    rounds: int = 16,
    seed: int = 11,
    time_limit: float | None = None,
    mip_gap: float = 1e-4,
    grade: np.ndarray | None = None,
) -> DestinationSchedule:
    """OPBSP-[D]: the exact restricted re-solve of Chicoisne et al. section 3.3, with destinations.

    The C-PIT[D] neighbourhoods (a block and a connected set of its predecessors, the same with
    successors, or the blocks of three consecutive periods), but the restricted model decides the
    PERIOD and the DESTINATION of every free block jointly, with binary destinations as in OPBSP
    (Jelvez, Morales and Nancel-Penard 2018, section 2.2). Fixed blocks keep both and keep their
    resources. Variables per free block ``i``: ``y_it`` (mined by the end of ``t``) and ``z_idt``
    (mined in ``t`` and sent to ``d``), linked by ``sum_d z_idt = y_it - y_i,t-1``. Every accepted
    move is a proven improvement of the restricted problem, so the objective is monotone and the
    result is never below ``start``.

    ``time_limit=None`` stops each re-solve on the relative MIP gap alone, so a bake is reproducible
    from (instance, seed) and not from the speed of the machine.
    """
    from scipy.optimize import LinearConstraint, milp
    from scipy.sparse import coo_matrix

    from .refine import _neighbourhood  # noqa: PLC2701 - the same neighbourhoods as C-PIT[D]
    from .schedule import _successors  # noqa: PLC2701

    n, t_max, n_dest, n_res = inst.n_blocks, inst.n_periods, inst.n_destinations, inst.n_resources
    disc = _discount(inst)
    sstart, slist = _successors(prec, n)
    rng = np.random.default_rng(seed)
    value = np.where(inst.forbidden, 0.0, np.nan_to_num(inst.value, neginf=0.0, posinf=0.0))

    period = start.period_of_block.copy()
    dest = start.destination_of_block.copy()
    _check_feasible(inst, prec, period, dest)
    npv_now, _ = pcpsp_schedule_value(inst, period, dest)
    accepted = 0

    for _ in range(rounds):
        dset = _neighbourhood(rng, prec, sstart, slist, period, d_max)
        if dset.size < 4:
            continue
        pos = {int(b): i for i, b in enumerate(dset)}
        k = int(dset.size)
        ny = k * t_max
        n_var = ny + k * n_dest * t_max

        def yv(i: int, t: int, _t_max: int = t_max) -> int:
            return i * _t_max + t

        def zv(i: int, dd: int, t: int, _ny: int = ny, _t_max: int = t_max) -> int:
            return _ny + (i * n_dest + dd) * _t_max + t

        used = np.zeros((n_res, t_max))
        free = np.zeros(n, dtype=bool)
        free[dset] = True
        for b in np.nonzero((period >= 0) & ~free)[0]:
            used[:, period[b]] += inst.coef[:, b, dest[b]]

        c = np.zeros(n_var)
        lb = np.zeros(n_var)
        ub = np.ones(n_var)
        for i, b in enumerate(dset):
            for dd in range(n_dest):
                for t in range(t_max):
                    if inst.forbidden[b, dd]:
                        ub[zv(i, dd, t)] = 0.0
                    else:
                        c[zv(i, dd, t)] = -disc[t] * float(value[b, dd])

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        lo: list[float] = []
        hi: list[float] = []

        def add(entries, low, high, _rows=rows, _cols=cols, _data=data, _lo=lo, _hi=hi):
            r = len(_lo)
            for j, val in entries:
                _rows.append(r)
                _cols.append(j)
                _data.append(val)
            _lo.append(low)
            _hi.append(high)

        for i, b in enumerate(dset):
            for t in range(t_max):
                if t + 1 < t_max:
                    add([(yv(i, t), 1.0), (yv(i, t + 1), -1.0)], -np.inf, 0.0)
                link = [(zv(i, dd, t), 1.0) for dd in range(n_dest)] + [(yv(i, t), -1.0)]
                if t > 0:
                    link.append((yv(i, t - 1), 1.0))
                add(link, 0.0, 0.0)
            for kk in range(prec.pstart[b], prec.pstart[b + 1]):
                a = int(prec.plist[kk])
                if a in pos:
                    for t in range(t_max):
                        add([(yv(i, t), 1.0), (yv(pos[a], t), -1.0)], -np.inf, 0.0)
                elif period[a] < 0:
                    ub[yv(i, 0) : yv(i, 0) + t_max] = 0.0
                else:
                    for t in range(int(period[a])):
                        ub[yv(i, t)] = 0.0
            for kk in range(sstart[b], sstart[b + 1]):
                cblk = int(slist[kk])
                if cblk in pos or period[cblk] < 0:
                    continue
                lb[yv(i, int(period[cblk]))] = 1.0
        if (lb > ub).any():
            continue

        for rr in range(n_res):
            for t in range(t_max):
                entries = [
                    (zv(i, dd, t), float(inst.coef[rr, b, dd]))
                    for i, b in enumerate(dset)
                    for dd in range(n_dest)
                    if inst.coef[rr, b, dd]
                ]
                if entries:
                    add(entries, -np.inf, float(inst.limit[rr][t] - used[rr, t]))

        a_mat = coo_matrix((data, (rows, cols)), shape=(len(lo), n_var)).tocsr()
        options: dict = {"mip_rel_gap": mip_gap, "presolve": True}
        if time_limit is not None:
            options["time_limit"] = time_limit
        try:
            res = milp(
                c=c,
                constraints=LinearConstraint(a_mat, np.array(lo), np.array(hi)),
                integrality=np.ones(n_var),
                bounds=(lb, ub),
                options=options,
            )
        except Exception:  # noqa: BLE001 - a solver failure must never lose the incumbent
            continue
        if not res.success or res.x is None:
            continue

        x = np.asarray(res.x)
        cand_p, cand_d = period.copy(), dest.copy()
        for i, b in enumerate(dset):
            cand_p[b], cand_d[b] = -1, -1
            for t in range(t_max):
                hit = [dd for dd in range(n_dest) if x[zv(i, dd, t)] > 0.5]
                if hit:
                    cand_p[b], cand_d[b] = t, hit[0]
                    break
        try:
            _check_feasible(inst, prec, cand_p, cand_d)
        except AssertionError:
            continue
        npv_new, _ = pcpsp_schedule_value(inst, cand_p, cand_d)
        if npv_new > npv_now + 1e-9 * max(1.0, abs(npv_now)):
            period, dest, npv_now = cand_p, cand_d, npv_new
            accepted += 1

    if npv_now < start.npv - 1e-6 * max(1.0, abs(start.npv)):
        raise AssertionError(f"destination local search made it worse: {start.npv} -> {npv_now}")
    return DestinationSchedule(
        method=f"{start.method}+opbspD-ls",
        period_of_block=period,
        destination_of_block=dest,
        npv=float(npv_now),
        mined_blocks=int((period >= 0).sum()),
        exact=False,
        effective_cutoff=_effective_cutoff(inst, period, dest, grade),
        notes=f"{accepted} of {rounds} exact OPBSP-[D] re-solves improved the incumbent",
    )


@dataclass(frozen=True)
class PcpspBound:
    """The PCPSP LP relaxation: an upper bound on every destination schedule of the instance.

    With ``solution=True`` it also carries what the relaxation says about the plan, the way the CPIT
    relaxation seeds ExTS: each block's expected extraction time ``E_b`` (``T + 1`` for a block the
    LP never mines) and the share of the mined part of each block that the LP sends to each
    destination.
    """

    bound: float
    seconds: float
    n_variables: int
    n_rows: int
    status: str
    expected_time: np.ndarray | None = None  # float64 (n,)
    destination_share: np.ndarray | None = None  # float64 (n, D), rows sum to 1 where the LP mines
    #: "highs-lp" (the LP solved directly) or "lagrangian" (its dual by closures; see
    #: ``pcpsp_lagrangian_bound``), so a reader knows which number they are looking at
    method: str = "highs-lp"
    #: Lagrangian only: cutting-plane iterations, the estimated distance to the dual optimum (relative),
    #: and the absolute rounding slack of the compiled closure, by which the bound may exceed the LP
    iterations: int = 0
    gap_estimate: float = 0.0
    slack: float = 0.0

    def preferred_destination(self) -> np.ndarray:
        """The destination the relaxation sends most of each block to; -1 where it mines none of it."""
        if self.destination_share is None:
            raise ValueError("solve the bound with solution=True to read its destinations")
        mined = self.destination_share.sum(axis=1) > 0
        return np.where(mined, self.destination_share.argmax(axis=1), -1).astype(np.int64)


def pcpsp_lp_bound(
    inst: Pcpsp,
    prec: Precedence,
    *,
    max_rows: int = 1_500_000,
    time_limit: float | None = None,
    solution: bool = False,
) -> PcpspBound | None:
    """Solve the PCPSP LP relaxation with HiGHS. ``None`` when the model is above ``max_rows``.

    The MineLib PCPSP model (Espinoza, Goycoolea, Moreno and Newman 2013,
    doi:10.1007/s10479-012-1258-3) with every integrality relaxed: cumulative extraction ``x_bt`` in
    [0, 1], monotone in ``t`` and closed under precedence in every period, destination fractions
    ``y_bdt >= 0`` with ``sum_d y_bdt = x_bt - x_b,t-1``, and per-period resource rows over ``y``.
    Every block of the file is kept: no ultimate-pit reduction is applied, because the bound must
    hold for the problem as stated and that reduction is proven for CPIT, not assumed here for PCPSP.

    The value is a bound to the solver's optimality tolerances; it is reported with its status, and
    a schedule above it is a defect of one of the two.
    """
    import time

    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix

    n, t_max, n_dest, n_res = inst.n_blocks, inst.n_periods, inst.n_destinations, inst.n_resources
    nx = n * t_max
    n_var = nx + n * n_dest * t_max
    n_rows = n * (t_max - 1) + int(prec.n_arcs) * t_max + n * t_max + n_res * t_max
    if n_rows > max_rows:
        return None
    disc = _discount(inst)
    value = np.where(inst.forbidden, 0.0, np.nan_to_num(inst.value, neginf=0.0, posinf=0.0))

    blocks = np.arange(n, dtype=np.int64)
    c = np.zeros(n_var)
    ub = np.ones(n_var)
    for dd in range(n_dest):
        bad = np.nonzero(inst.forbidden[:, dd])[0]
        for t in range(t_max):
            c[nx + (blocks * n_dest + dd) * t_max + t] = -disc[t] * value[:, dd]
            ub[nx + (bad * n_dest + dd) * t_max + t] = 0.0

    ub_rows: list[np.ndarray] = []
    ub_cols: list[np.ndarray] = []
    ub_data: list[np.ndarray] = []
    ub_rhs: list[np.ndarray] = []
    row = 0
    # monotone: x_bt - x_b,t+1 <= 0
    for t in range(t_max - 1):
        rws = np.arange(row, row + n)
        ub_rows += [rws, rws]
        ub_cols += [blocks * t_max + t, blocks * t_max + t + 1]
        ub_data += [np.ones(n), -np.ones(n)]
        row += n
    ub_rhs.append(np.zeros(n * (t_max - 1)))
    # precedence in every period: x_bt - x_at <= 0
    owner = np.repeat(blocks, np.diff(prec.pstart))
    pred = prec.plist.astype(np.int64)
    m = owner.shape[0]
    for t in range(t_max):
        rws = np.arange(row, row + m)
        ub_rows += [rws, rws]
        ub_cols += [owner * t_max + t, pred * t_max + t]
        ub_data += [np.ones(m), -np.ones(m)]
        row += m
    ub_rhs.append(np.zeros(m * t_max))
    # resources: sum_{b,d} q_rbd y_bdt <= c_rt
    for rr in range(n_res):
        for t in range(t_max):
            for dd in range(n_dest):
                q = np.asarray(inst.coef[rr][:, dd], dtype=np.float64)
                nz = np.nonzero(q)[0]
                ub_rows.append(np.full(nz.size, row))
                ub_cols.append(nx + (nz * n_dest + dd) * t_max + t)
                ub_data.append(q[nz])
            ub_rhs.append(np.array([float(inst.limit[rr][t])]))
            row += 1
    a_ub = coo_matrix(
        (np.concatenate(ub_data), (np.concatenate(ub_rows), np.concatenate(ub_cols))), shape=(row, n_var)
    ).tocsr()
    b_ub = np.concatenate(ub_rhs)

    # linking: sum_d y_bdt - x_bt + x_b,t-1 = 0
    eq_rows: list[np.ndarray] = []
    eq_cols: list[np.ndarray] = []
    eq_data: list[np.ndarray] = []
    erow = 0
    for t in range(t_max):
        rws = np.arange(erow, erow + n)
        for dd in range(n_dest):
            eq_rows.append(rws)
            eq_cols.append(nx + (blocks * n_dest + dd) * t_max + t)
            eq_data.append(np.ones(n))
        eq_rows.append(rws)
        eq_cols.append(blocks * t_max + t)
        eq_data.append(-np.ones(n))
        if t > 0:
            eq_rows.append(rws)
            eq_cols.append(blocks * t_max + t - 1)
            eq_data.append(np.ones(n))
        erow += n
    a_eq = coo_matrix(
        (np.concatenate(eq_data), (np.concatenate(eq_rows), np.concatenate(eq_cols))), shape=(erow, n_var)
    ).tocsr()

    options: dict = {"presolve": True}
    if time_limit is not None:
        options["time_limit"] = time_limit
    t0 = time.perf_counter()
    res = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=np.zeros(erow),
        bounds=np.stack([np.zeros(n_var), ub], 1),
        method="highs",
        options=options,
    )
    seconds = time.perf_counter() - t0
    if res.status != 0:
        return PcpspBound(
            bound=float("nan"), seconds=seconds, n_variables=n_var, n_rows=row + erow, status=str(res.message)
        )
    expected = share = None
    if solution:
        x = np.clip(np.asarray(res.x[:nx]).reshape(n, t_max), 0.0, 1.0)
        y = np.clip(np.asarray(res.x[nx:]).reshape(n, n_dest, t_max), 0.0, None)
        prev = np.concatenate([np.zeros((n, 1)), x[:, :-1]], axis=1)
        expected = ((x - prev) * np.arange(1, t_max + 1)).sum(axis=1) + (t_max + 1) * (1.0 - x[:, -1])
        sent = y.sum(axis=2)
        total = sent.sum(axis=1, keepdims=True)
        share = np.divide(sent, total, out=np.zeros_like(sent), where=total > 1e-9)
    return PcpspBound(
        bound=float(-res.fun), seconds=seconds, n_variables=n_var, n_rows=row + erow, status="optimal",
        expected_time=expected, destination_share=share,
    )


def _time_expanded(prec: Precedence, n: int, t_max: int) -> Precedence:
    """Node ``t * n + b`` is ``x_bt``. It requires ``(a, t)`` for every predecessor ``a`` of ``b``
    (precedence in every period) and ``(b, t + 1)`` for ``t < T - 1`` (mined by ``t`` means mined by
    ``t + 1``), so a closure of this graph is a monotone, precedence-feasible cumulative schedule."""
    deg = np.diff(prec.pstart).astype(np.int64)
    counts = np.concatenate([deg + (1 if t < t_max - 1 else 0) for t in range(t_max)])
    starts = np.zeros(n * t_max + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    plist = np.empty(int(starts[-1]), dtype=np.int64)
    within = np.arange(int(deg.sum()), dtype=np.int64) - np.repeat(np.cumsum(deg) - deg, deg)
    for t in range(t_max):
        base = starts[t * n:(t + 1) * n]
        plist[np.repeat(base, deg) + within] = np.asarray(prec.plist, dtype=np.int64) + t * n
        if t < t_max - 1:
            plist[base + deg] = np.arange(n, dtype=np.int64) + (t + 1) * n
    return Precedence(pstart=starts, plist=plist)


def pcpsp_lagrangian_bound(
    inst: Pcpsp,
    prec: Precedence,
    *,
    max_iter: int = 400,
    tol: float = 1e-6,
    stall: int = 40,
) -> PcpspBound:
    """The PCPSP LP bound through its Lagrangian dual: closures instead of one huge LP.

    Dualise the ``R T`` capacity rows with multipliers ``mu_rt >= 0``. What is left is, for each block
    and period, the choice of the destination worth most at those prices, and a precedence-closed,
    monotone cumulative schedule, which is a maximum closure on the time-expanded graph:

    .. code-block:: text

        g_bt(mu) = max_d ( disc_t v_bd - sum_r mu_rt q_rbd )
        L(mu)    = sum_rt mu_rt c_rt + max over closures of sum_bt (g_bt - g_b,t+1) x_bt

    ``L(mu)`` is an upper bound on every destination schedule for EVERY ``mu >= 0``, and its minimum is
    the PCPSP LP value (the inner problem is a closure, totally unimodular, so the dual has no gap with
    the LP). The closure is the compiled one, which rounds weights UP, so each ``L(mu)`` can only
    over-estimate: the reported bound is valid at any iteration, and exceeds the LP by at most the
    returned ``slack`` once converged.

    ``mu`` is driven by a cutting-plane method in the ``R T`` multipliers with a box trust region
    around the best point (a serious step doubles the box, a null step shrinks it), stopping when the
    model's lower estimate is within ``max(tol |bound|, 2 slack)`` of the best value with the box not
    binding, after ``stall`` iterations without a serious step, or at ``max_iter``.

    The relaxed schedule at the best ``mu`` is returned as one-hot ``destination_share`` and
    ``expected_time``: the destination each block takes at those prices, with the plant's opportunity
    cost priced in, which is what the re-cut fixes (``restrict_destinations``).

    Why it exists: the direct LP (``pcpsp_lp_bound``) has about ``n T`` precedence rows times the arc
    density; on a 14,400-block, ten-period deposit that is 1.44 million rows, and HiGHS did not finish
    two such instances in six and a half hours. Here the same bound took minutes, and on a 6,912-block
    instance it lands 4.8 parts per million above the exact LP (316,476,932 against 316,475,407).
    """
    import time

    from scipy.optimize import linprog

    from .fastcut import max_closure_fast

    t0 = time.perf_counter()
    n, t_max, n_dest, n_res = inst.n_blocks, inst.n_periods, inst.n_destinations, inst.n_resources
    disc = _discount(inst)
    coef = np.asarray(inst.coef, dtype=np.float64)  # (R, n, D)
    cap = np.asarray(inst.limit, dtype=np.float64)  # (R, T)
    value = np.where(inst.forbidden, -np.inf, np.nan_to_num(inst.value, neginf=-np.inf, posinf=0.0))
    usable = np.isfinite(value).any(axis=1)  # a block with every destination forbidden is never mined
    graph = _time_expanded(prec, n, t_max)
    big = float(np.abs(np.where(np.isfinite(value), value, 0.0)).sum() * max(1.0, float(disc.max()))) + 1.0

    def evaluate(mu: np.ndarray):
        g = disc[:, None, None] * value[None, :, :] - np.einsum("rt,rbd->tbd", mu, coef)
        g = np.where(np.isfinite(g), g, -np.inf)
        best_d = np.argmax(g, axis=2)  # (T, n)
        gbest = np.take_along_axis(g, best_d[:, :, None], axis=2)[:, :, 0]
        gbest = np.where(usable[None, :], gbest, -big)
        w = gbest - np.vstack([gbest[1:], np.zeros((1, n))])
        res = max_closure_fast(w.ravel(), graph)
        x = res.mask.reshape(t_max, n)
        inc = x & ~np.vstack([np.zeros((1, n), dtype=bool), x[:-1]])
        use = np.zeros((n_res, t_max))
        for t in range(t_max):
            bs = np.nonzero(inc[t])[0]
            if bs.size:
                use[:, t] = coef[:, bs, best_d[t, bs]].sum(axis=1)
        return float((mu * cap).sum() + res.value), cap - use, inc, best_d, float(res.slack)

    # trust-region scale per resource: a value per unit of resource that no multiplier needs to exceed
    finite_v = np.where(np.isfinite(value), np.abs(value), 0.0)
    def _smallest_positive(row: np.ndarray) -> float:
        pos = row[row > 0]
        return float(pos.min()) if pos.size else 1.0

    peak = float((disc[0] * finite_v).max())
    scale = np.array([peak / max(1e-12, _smallest_positive(coef[r])) for r in range(n_res)])
    center = np.zeros((n_res, t_max))
    best_val, sg, inc, best_d, slack = evaluate(center)
    best = (best_val, center.copy(), inc, best_d)
    cuts = [(center.copy(), best_val, sg)]
    delta = np.repeat(scale[:, None], t_max, axis=1) * 0.05
    model = -np.inf
    since_serious = 0
    iterations = 0
    status = "iteration limit"
    nv = n_res * t_max + 1
    for _ in range(max_iter):
        iterations += 1
        a_ub = np.zeros((len(cuts), nv))
        b_ub = np.zeros(len(cuts))
        for k, (muk, lk, sk) in enumerate(cuts):
            a_ub[k, :-1] = sk.ravel()
            a_ub[k, -1] = -1.0
            b_ub[k] = -(lk - float((sk * muk).sum()))
        lo = np.maximum(0.0, center - delta).ravel()
        hi = (center + delta).ravel()
        cvec = np.zeros(nv)
        cvec[-1] = 1.0
        mres = linprog(cvec, A_ub=a_ub, b_ub=b_ub, bounds=list(zip(lo, hi, strict=True)) + [(None, None)],
                       method="highs")
        if mres.x is None:
            status = f"master LP failed: {mres.message}"
            break
        mu = mres.x[:-1].reshape(n_res, t_max)
        model = float(mres.x[-1])
        val, sg, inc, best_d, s_now = evaluate(mu)
        slack = max(slack, s_now)
        cuts.append((mu.copy(), val, sg))
        if val < best[0] - 0.1 * max(best[0] - model, 0.0):
            best = (val, mu.copy(), inc, best_d)
            center = mu.copy()
            delta = delta * 2.0
            since_serious = 0
        else:
            delta = delta * 0.7
            since_serious += 1
        on_edge = bool(np.any(np.isclose(mu.ravel(), hi) & (hi > lo + 1e-12)))
        if best[0] - model <= max(tol * abs(best[0]), 2.0 * slack) and not on_edge:
            status = "converged"
            break
        if since_serious >= stall:
            status = "stalled"
            break

    _, _, inc, best_d = best
    expected = np.full(n, t_max + 1.0)
    share = np.zeros((n, n_dest))
    for t in range(t_max):
        bs = np.nonzero(inc[t])[0]
        expected[bs] = t + 1.0
        share[bs, best_d[t, bs]] = 1.0
    return PcpspBound(
        bound=float(best[0]), seconds=time.perf_counter() - t0, n_variables=n * t_max,
        n_rows=int(graph.plist.shape[0]), status=status, expected_time=expected, destination_share=share,
        method="lagrangian", iterations=iterations,
        gap_estimate=float((best[0] - model) / abs(best[0])) if best[0] else 0.0, slack=slack,
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

    cutoff = _effective_cutoff(inst, period, dest, grade)

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
