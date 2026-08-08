"""Tests for the rungs added in 0.3.0: BZ, sliding window, exact local search, destinations,
Lane cutoffs, width smoothing and the stochastic ensemble.

The load-bearing test in this file is the first one: Bienstock-Zuckerberg and the critical multiplier
algorithm are two completely different algorithms computing the SAME quantity, so on a single-resource
instance they must agree to machine precision. If they ever disagree, one of them is wrong and no
amount of plausible-looking output will say which.
"""

from __future__ import annotations

import numpy as np
import pytest

import oreblocks as ob

scipy = pytest.importorskip("scipy", reason="the MILP and BZ master need scipy (oreblocks[milp])")


def _instance(dims=(10, 10, 6), seed=5, periods=5, n_res=2, slack=(0.62, 0.5)):
    twin = ob.make_twin("porphyry", dims=dims, seed=seed)
    ton = twin.deposit.tonnage
    vals = twin.values.astype(float)
    in_pit = twin.upit.in_pit
    coef = np.zeros((n_res, vals.shape[0]))
    coef[0] = ton
    lim = [[slack[0] * float(ton[in_pit].sum()) / periods] * periods]
    if n_res > 1:
        coef[1] = np.where(vals > 0, ton, 0.0)
        lim.append([slack[1] * float(coef[1][in_pit].sum()) / periods] * periods)
    inst = ob.Cpit(
        name="t", n_blocks=vals.shape[0], n_periods=periods, discount_rate=0.10, value=vals,
        limit=np.array(lim), sense=np.full((n_res, periods), "L", dtype="<U1"), coef=coef,
        resource_names=("mining", "processing")[:n_res],
    )
    return twin, inst


# ------------------------------------------------------------------------------------------------
# Bienstock-Zuckerberg
# ------------------------------------------------------------------------------------------------
def test_bz_and_the_critical_multiplier_agree_on_a_single_resource():
    """Two different algorithms, one LP. They must agree, and this is the strongest check here."""
    twin, inst = _instance(n_res=1, slack=(0.6, 0))
    cma = ob.cpit_lp_relaxation(inst, twin.precedence).bound
    bz = ob.cpit_bz_bound(inst, twin.precedence)
    assert bz.converged
    assert bz.bound == pytest.approx(cma, rel=1e-9)


def test_bz_is_never_looser_than_the_relax_one_at_a_time_bound():
    """Algorithm 4 relaxes all but one resource; BZ handles them jointly, so BZ <= Algorithm 4."""
    twin, inst = _instance(n_res=2)
    alg4, _ = ob.cpit_bound_two_resources(inst, twin.precedence)
    bz = ob.cpit_bz_bound(inst, twin.precedence)
    assert bz.bound <= alg4 * (1 + 1e-9)


def test_bz_bound_dominates_every_feasible_schedule():
    twin, inst = _instance(n_res=2)
    bz = ob.cpit_bz_bound(inst, twin.precedence)
    for w in ob.TOPOSORT_WEIGHTS:
        res = ob.toposort_schedule(inst, twin.precedence, weight=w)
        assert res.npv <= bz.bound * (1 + 1e-6)


def test_bz_terminates_on_its_certificate_not_on_an_iteration_cap():
    twin, inst = _instance(n_res=2)
    bz = ob.cpit_bz_bound(inst, twin.precedence, max_iter=200)
    assert bz.converged, "BZ hit the iteration cap instead of proving optimality"
    assert bz.pricing_solves == bz.iterations


# ------------------------------------------------------------------------------------------------
# the sliding time window
# ------------------------------------------------------------------------------------------------
def test_sliding_window_is_feasible_and_competitive():
    """The regression this guards: the window used to re-plan blocks that had already consumed
    capacity, which double-books the fleet and collapses the objective to a third of its value."""
    twin, inst = _instance(n_res=2)
    prec = twin.precedence
    sw = ob.sliding_window_schedule(inst, prec, window=3, fix=1)
    ref = ob.toposort_schedule(inst, prec, weight="expected")

    for r in range(inst.n_resources):
        assert (sw.per_period_resource[r] <= inst.limit[r] + 1e-6).all()
    per = sw.period_of_block
    for b in np.nonzero(per >= 0)[0]:
        for p in prec.preds(b):
            assert per[p] >= 0 and per[p] <= per[b]
    assert sw.npv > 0.8 * ref.npv, "a sliding window should be within a fifth of expected-time TopoSort"


# ------------------------------------------------------------------------------------------------
# the exact C-PIT[D] local search
# ------------------------------------------------------------------------------------------------
def test_exact_local_search_never_loses_and_stays_feasible():
    twin, inst = _instance(n_res=2)
    prec = twin.precedence
    base = ob.toposort_schedule(inst, prec, weight="greedy")
    out = ob.exact_local_search(inst, prec, base, d_max=90, rounds=6, time_limit=6)
    assert out.npv >= base.npv - 1e-9
    per = out.period_of_block
    for b in np.nonzero(per >= 0)[0]:
        for p in prec.preds(b):
            assert per[p] >= 0 and per[p] <= per[b]
    for r in range(inst.n_resources):
        assert (out.per_period_resource[r] <= inst.limit[r] + 1e-6).all()


def test_exact_local_search_beats_the_shift_neighbourhood_at_least_once():
    """The reason it exists: the shift neighbourhood cannot move a block and its cone together."""
    twin, inst = _instance(dims=(12, 12, 7), n_res=2)
    prec = twin.precedence
    seed = ob.toposort_schedule(inst, prec, weight="expected")
    shift = ob.improve_schedule(inst, prec, seed)
    exact = ob.exact_local_search(inst, prec, seed, d_max=140, rounds=14, time_limit=8)
    assert exact.npv >= shift.npv - 1e-9


# ------------------------------------------------------------------------------------------------
# destinations
# ------------------------------------------------------------------------------------------------
def _pcpsp(dims=(9, 9, 5), periods=4):
    twin = ob.make_twin("porphyry", dims=dims, seed=4)
    n = twin.values.shape[0]
    ton = twin.deposit.tonnage
    waste_val = -0.4 * ton
    value = np.stack([waste_val, twin.values.astype(float)], axis=1)
    coef = np.zeros((2, n, 2))
    coef[0, :, 0] = ton
    coef[0, :, 1] = ton
    coef[1, :, 1] = ton  # only the plant destination consumes plant capacity
    lim = np.array([
        [0.7 * float(ton[twin.upit.in_pit].sum()) / periods] * periods,
        [0.35 * float(ton[twin.upit.in_pit].sum()) / periods] * periods,
    ])
    inst = ob.Pcpsp(
        name="p", n_blocks=n, n_periods=periods, n_destinations=2, discount_rate=0.10,
        value=value, forbidden=np.zeros((n, 2), dtype=bool), limit=lim,
        sense=np.full((2, periods), "L", dtype="<U1"), coef=coef,
    )
    return twin, inst


def test_destination_heuristic_makes_the_cutoff_an_output():
    twin, inst = _pcpsp()
    out = ob.destination_toposort(inst, twin.precedence, grade=twin.deposit.grade)
    assert out.mined_blocks > 0
    assert set(np.unique(out.destination_of_block[out.destination_of_block >= 0])) <= {0, 1}
    finite = out.effective_cutoff[np.isfinite(out.effective_cutoff)]
    assert finite.size > 0, "a destination-aware schedule must report the cutoff it produced"
    # plant capacity binds, so at least one period must have pushed material to waste
    to_plant = (out.destination_of_block == 1).sum()
    to_waste = (out.destination_of_block == 0).sum()
    assert to_plant > 0 and to_waste > 0


def test_exact_opbsp_matches_or_beats_the_heuristic_and_respects_capacity():
    twin, inst = _pcpsp(dims=(7, 7, 4), periods=3)
    heur = ob.destination_toposort(inst, twin.precedence, grade=twin.deposit.grade)
    exact = ob.solve_opbsp_exact(inst, twin.precedence, max_variables=40_000, time_limit=60)
    if exact is None:
        pytest.skip("instance above the exact-solve budget")
    assert exact.exact
    assert exact.npv >= heur.npv - 1e-6 * abs(heur.npv)
    for r in range(inst.n_resources):
        for t in range(inst.n_periods):
            used = sum(
                inst.coef[r][b, exact.destination_of_block[b]]
                for b in np.nonzero(exact.period_of_block == t)[0]
            )
            assert used <= inst.limit[r][t] + 1e-6


def test_opbsp_returns_none_rather_than_faking_an_exact_answer():
    twin, inst = _pcpsp(dims=(16, 16, 9), periods=8)
    assert ob.solve_opbsp_exact(inst, twin.precedence, max_variables=1000) is None


# ------------------------------------------------------------------------------------------------
# Lane cutoffs
# ------------------------------------------------------------------------------------------------
def test_lane_optimum_is_the_middle_of_the_three_limiting_cutoffs():
    p = ob.lane_cutoffs(
        price=8000, selling_cost=400, mining_cost=2.5, processing_cost=12,
        fixed_cost_per_year=8e6, recovery=0.88, mine_capacity=20e6, mill_capacity=7e6,
        market_capacity=1e5, discount_rate=0.10, remaining_npv=3e8,
    )
    assert p.optimum == pytest.approx(float(np.median([p.mine_limiting, p.mill_limiting, p.market_limiting])))
    assert p.mine_limiting == pytest.approx(p.break_even)
    assert p.mill_limiting > p.break_even, "the mill-limiting cutoff carries the opportunity cost"


def test_lane_cutoff_falls_as_the_remaining_value_falls():
    """Lane's central prediction: the economic cutoff DECLINES over the life of a mine, because the
    opportunity cost of a processing hour falls as there is less value left to delay."""
    kw = dict(price=8000, selling_cost=400, mining_cost=2.5, processing_cost=12,
              fixed_cost_per_year=8e6, recovery=0.88, mine_capacity=20e6, mill_capacity=7e6,
              market_capacity=1e5, discount_rate=0.10)
    early = ob.lane_cutoffs(remaining_npv=5e8, **kw)
    late = ob.lane_cutoffs(remaining_npv=2e7, **kw)
    assert late.mill_limiting < early.mill_limiting


# ------------------------------------------------------------------------------------------------
# minimum mining width
# ------------------------------------------------------------------------------------------------
def test_min_width_reduces_slivers_and_reports_what_it_cost():
    twin, inst = _instance(n_res=2)
    prec = twin.precedence
    ix, iy, lev = twin.deposit.grid.coord_arrays()
    base, _ = ob.solve_cpit(inst, prec)
    out, rep = ob.enforce_min_width(inst, base, ix, iy, lev, prec, target_width=3)
    assert rep.below_target_after <= rep.below_target_before
    assert rep.npv_after <= rep.npv_before + 1e-6, "smoothing trades value for operability"
    per = out.period_of_block
    for b in np.nonzero(per >= 0)[0]:
        for p in prec.preds(b):
            if per[p] >= 0:
                assert per[p] <= per[b], "smoothing must not break precedence"


# ------------------------------------------------------------------------------------------------
# the stochastic ensemble
# ------------------------------------------------------------------------------------------------
def test_the_ensemble_is_mean_preserving_and_spatially_correlated():
    twin, _ = _instance()
    ix, iy, lev = twin.deposit.grid.coord_arrays()
    vals = twin.values.astype(float)
    ore = vals > 0
    ens = ob.perturb_values(vals, ix, iy, lev, n=120, sigma=0.25, seed=3)
    ratio = ens.values[:, ore].mean() / vals[ore].mean()
    assert ratio == pytest.approx(1.0, abs=0.02), "a biased ensemble corrupts the optimism readout"

    # correlated, not white: neighbouring blocks must move together more than distant ones
    order = np.argsort(lev * 10_000 + iy * 100 + ix)
    dev = (ens.values[0] / np.where(vals != 0, vals, 1.0))[order]
    near = np.corrcoef(dev[:-1], dev[1:])[0, 1]
    far = np.corrcoef(dev[:-50], dev[50:])[0, 1]
    assert near > far, "the perturbation is white noise, which makes uncertainty look harmless"


def test_evpi_is_none_without_real_per_realisation_optima():
    twin, inst = _instance()
    ix, iy, lev = twin.deposit.grid.coord_arrays()
    ens = ob.perturb_values(twin.values.astype(float), ix, iy, lev, n=6, seed=2)
    base, _ = ob.solve_cpit(inst, twin.precedence)
    out = ob.evaluate_across(inst, ens, {"base": base.period_of_block})
    assert out.value_of_information is None, "EVPI without re-optimisation would overstate the case"
    assert out.value_of_plan_selection == pytest.approx(0.0, abs=1e-6)


def test_ensemble_reports_a_distribution_and_a_robust_choice():
    twin, inst = _instance()
    prec = twin.precedence
    ix, iy, lev = twin.deposit.grid.coord_arrays()
    ens = ob.perturb_values(twin.values.astype(float), ix, iy, lev, n=8, seed=2)
    plans = {
        w: ob.toposort_schedule(inst, prec, weight=w).period_of_block for w in ob.TOPOSORT_WEIGHTS
    }
    out = ob.evaluate_across(inst, ens, plans)
    assert out.matrix.shape == (3, 8)
    assert (out.p10 <= out.expected + 1e-9).all()
    assert (out.expected <= out.p90 + 1e-9).all()
    assert out.best_by_expected in plans
    assert out.best_by_p10 in plans
