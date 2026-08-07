"""Tests for the CPIT scheduling lane: file formats, the certified bound, the heuristics, controls.

Everything here runs on synthetic instances generated in-process, so the suite needs no MineLib
download and redistributes nothing. The published-instance reproduction lives in the PhaseFlow
product repo, which caches MineLib locally under the academic-download grant.
"""

from __future__ import annotations

import numpy as np
import pytest

import oreblocks as ob


# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
def _twin(dims=(8, 8, 5), seed=7, archetype="porphyry"):
    twin = ob.make_twin(archetype, dims=dims, seed=seed)
    return twin


def _cpit_from_twin(twin, *, periods=4, rate=0.10, slack=1.2, n_resources=1):
    """Build a CPIT instance from a twin: capacity = slack * (pit tonnage / periods)."""
    values = twin.values
    tonnage = twin.deposit.tonnage
    in_pit = twin.upit.in_pit
    cap = slack * float(tonnage[in_pit].sum()) / periods
    limit = np.full((n_resources, periods), cap)
    coef = np.zeros((n_resources, values.shape[0]))
    coef[0] = tonnage
    if n_resources > 1:
        ore = values > 0
        coef[1] = np.where(ore, tonnage, 0.0)
        limit[1] = 0.7 * cap
    return ob.Cpit(
        name="twin",
        n_blocks=values.shape[0],
        n_periods=periods,
        discount_rate=rate,
        value=values.astype(float),
        limit=limit,
        sense=np.full((n_resources, periods), "L", dtype="<U1"),
        coef=coef,
    )


# ------------------------------------------------------------------------------------------------
# file formats
# ------------------------------------------------------------------------------------------------
def test_cpit_round_trip(tmp_path):
    twin = _twin()
    inst = _cpit_from_twin(twin, n_resources=2)
    p = ob.write_cpit(tmp_path / "twin.cpit", inst)
    back = ob.read_cpit(p)
    assert back.n_blocks == inst.n_blocks
    assert back.n_periods == inst.n_periods
    assert back.n_resources == inst.n_resources
    assert back.discount_rate == pytest.approx(inst.discount_rate)
    assert np.allclose(back.value, inst.value, rtol=1e-8)
    assert np.allclose(back.limit, inst.limit, rtol=1e-8)
    assert np.allclose(back.coef, inst.coef, rtol=1e-8)


def test_cpit_reader_accepts_sparse_coefficients(tmp_path):
    """The published newman1 file lists resource 1 for only the ore blocks. Absent means zero."""
    text = (
        "NAME: tiny\nTYPE: CPIT\nNBLOCKS: 3\nNPERIODS: 2\nNRESOURCE_SIDE_CONSTRAINTS: 2\n"
        "DISCOUNT_RATE: 0.1\nRESOURCE_CONSTRAINT_LIMITS:\n"
        "0 0 L 10\n0 1 L 10\n1 0 L 4\n1 1 L 4\n"
        "OBJECTIVE_FUNCTION:\n0 5\n1 -1\n2 3\n"
        "RESOURCE_CONSTRAINT_COEFFICIENTS:\n0 0 1\n1 0 1\n2 0 1\n0 1 2\nEOF\n"
    )
    p = tmp_path / "tiny.cpit"
    p.write_text(text, encoding="ascii")
    inst = ob.read_cpit(p)
    assert inst.coef.shape == (2, 3)
    assert inst.coef[1].tolist() == [2.0, 0.0, 0.0]


def test_pcpsp_forbidden_destination_sentinel(tmp_path):
    text = (
        "NAME: tiny\nTYPE: PCPSP\nNBLOCKS: 2\nNPERIODS: 1\nNDESTINATIONS: 2\n"
        "NRESOURCE_SIDE_CONSTRAINTS: 1\nNGENERAL_SIDE_CONSTRAINTS: 0\nDISCOUNT_RATE: 0.0\n"
        "RESOURCE_CONSTRAINT_LIMITS:\n0 0 L 100\n"
        "OBJECTIVE_FUNCTION:\n0 -1 7\n1 -2 -5.36024E+19\n"
        "RESOURCE_CONSTRAINT_COEFFICIENTS:\n0 0 0 1\n0 1 0 1\n1 0 0 1\nEOF\n"
    )
    p = tmp_path / "tiny.pcpsp"
    p.write_text(text, encoding="ascii")
    inst = ob.read_pcpsp(p)
    assert inst.forbidden.tolist() == [[False, False], [False, True]]
    assert np.isneginf(inst.value[1, 1])
    dest, val = inst.best_destination()
    assert dest.tolist() == [1, 0]
    assert val.tolist() == [7.0, -2.0]


def test_pcpsp_round_trip(tmp_path):
    twin = _twin(dims=(6, 6, 4))
    n = twin.values.shape[0]
    rng = np.random.default_rng(3)
    value = np.stack([twin.values, twin.values - rng.uniform(0, 50, n)], axis=1)
    value[0, 1] = -np.inf
    forbidden = ~np.isfinite(value)
    coef = np.zeros((1, n, 2))
    coef[0, :, 0] = twin.deposit.tonnage
    coef[0, :, 1] = twin.deposit.tonnage
    inst = ob.Pcpsp(
        name="twin", n_blocks=n, n_periods=3, n_destinations=2, discount_rate=0.1,
        value=value, forbidden=forbidden, limit=np.full((1, 3), 1e6),
        sense=np.full((1, 3), "L", dtype="<U1"), coef=coef,
    )
    back = ob.read_pcpsp(ob.write_pcpsp(tmp_path / "twin.pcpsp", inst))
    assert back.forbidden.tolist() == forbidden.tolist()
    fin = np.isfinite(value)
    assert np.allclose(back.value[fin], value[fin], rtol=1e-8)


def test_pcpsp_to_cpit_fixes_the_destination():
    twin = _twin(dims=(6, 6, 4))
    n = twin.values.shape[0]
    value = np.stack([np.full(n, -1.0), twin.values], axis=1)
    coef = np.zeros((1, n, 2))
    coef[0, :, 0] = 1.0
    coef[0, :, 1] = 2.0
    inst = ob.Pcpsp(
        name="t", n_blocks=n, n_periods=2, n_destinations=2, discount_rate=0.1,
        value=value, forbidden=np.zeros((n, 2), dtype=bool), limit=np.full((1, 2), 1e9),
        sense=np.full((1, 2), "L", dtype="<U1"), coef=coef,
    )
    cp = inst.to_cpit()
    # blocks worth more than -1 take destination 1 (coefficient 2), the rest take destination 0
    picks_one = twin.values > -1.0
    assert np.allclose(cp.coef[0][picks_one], 2.0)
    assert np.allclose(cp.coef[0][~picks_one], 1.0)


# ------------------------------------------------------------------------------------------------
# discounting
# ------------------------------------------------------------------------------------------------
def test_first_period_is_undiscounted_by_default():
    inst = _cpit_from_twin(_twin(), periods=3, rate=0.10)
    d = inst.discount_factors()
    assert d[0] == pytest.approx(1.0)
    assert d[1] == pytest.approx(1 / 1.1)
    assert d[2] == pytest.approx(1 / 1.21)


def test_gamma_is_the_abel_summation_of_the_discount_factors():
    """sum_t d_t (x_t - x_{t-1}).v must equal sum_t gamma_t (x_t.v) for any monotone x."""
    inst = _cpit_from_twin(_twin(), periods=5, rate=0.12)
    rng = np.random.default_rng(11)
    n = inst.n_blocks
    v = rng.normal(size=n)
    x = np.sort(rng.uniform(size=(inst.n_periods, n)), axis=0)
    d = inst.discount_factors()
    prev = np.zeros(n)
    lhs = 0.0
    for t in range(inst.n_periods):
        lhs += d[t] * float((x[t] - prev) @ v)
        prev = x[t]
    rhs = float((inst.gamma() * (x @ v)).sum())
    assert lhs == pytest.approx(rhs, rel=1e-12)


# ------------------------------------------------------------------------------------------------
# the certified bound
# ------------------------------------------------------------------------------------------------
def test_max_closure_within_matches_the_full_solve_when_everything_is_a_candidate():
    twin = _twin()
    full = ob.solve_upit(twin.values, twin.precedence)
    sub = ob.max_closure_within(twin.values, twin.precedence, np.ones(twin.values.shape[0], bool))
    assert np.array_equal(sub, full.in_pit)


def test_lp_relaxation_is_an_upper_bound_and_hits_upit_in_the_degenerate_case():
    twin = _twin()
    exact = ob.solve_upit(twin.values, twin.precedence)
    huge = float(twin.deposit.tonnage.sum()) + 1.0
    inst = ob.Cpit(
        name="degenerate", n_blocks=twin.values.shape[0], n_periods=1, discount_rate=0.0,
        value=twin.values.astype(float), limit=np.full((1, 1), huge),
        sense=np.full((1, 1), "L", dtype="<U1"),
        coef=twin.deposit.tonnage.reshape(1, -1).astype(float),
    )
    rel = ob.cpit_lp_relaxation(inst, twin.precedence)
    assert rel.bound == pytest.approx(exact.pit_value, rel=1e-9)
    assert np.array_equal(rel.x[-1] > 0.5, exact.in_pit)
    assert rel.integral.all()


def test_lp_relaxation_is_monotone_and_saturates_the_cumulative_capacity():
    twin = _twin()
    inst = _cpit_from_twin(twin, periods=5, slack=0.5)  # tight: capacity must bind
    rel = ob.cpit_lp_relaxation(inst, twin.precedence)
    for t in range(inst.n_periods - 1):
        assert (rel.x[t] <= rel.x[t + 1] + 1e-9).all()
    used = rel.x @ inst.coef[0]
    caps = inst.cumulative_limits(0)
    for t in range(inst.n_periods):
        assert used[t] <= caps[t] + 1e-6
        if not rel.integral[t]:
            assert used[t] == pytest.approx(caps[t], rel=1e-9)


def test_bound_dominates_every_feasible_schedule():
    twin = _twin()
    inst = _cpit_from_twin(twin, periods=5, slack=0.6, n_resources=2)
    bound, _ = ob.cpit_bound_two_resources(inst, twin.precedence)
    for w in ob.TOPOSORT_WEIGHTS:
        res = ob.toposort_schedule(inst, twin.precedence, weight=w)
        assert res.npv <= bound + 1e-6 * abs(bound)


def test_a_looser_capacity_can_only_raise_the_bound():
    twin = _twin()
    tight = ob.cpit_lp_relaxation(_cpit_from_twin(twin, slack=0.4), twin.precedence)
    loose = ob.cpit_lp_relaxation(_cpit_from_twin(twin, slack=0.9), twin.precedence)
    assert loose.bound >= tight.bound - 1e-9


# ------------------------------------------------------------------------------------------------
# schedules
# ------------------------------------------------------------------------------------------------
def test_schedules_respect_precedence_and_capacity():
    twin = _twin()
    inst = _cpit_from_twin(twin, periods=5, slack=0.7, n_resources=2)
    prec = twin.precedence
    for w in ob.TOPOSORT_WEIGHTS:
        res = ob.toposort_schedule(inst, prec, weight=w)
        per = res.period_of_block
        for b in np.nonzero(per >= 0)[0]:
            for p in prec.preds(b):
                assert per[p] >= 0, f"block {b} mined but predecessor {p} was not"
                assert per[p] <= per[b], f"block {b} mined before its predecessor {p}"
        for r in range(inst.n_resources):
            assert (res.per_period_resource[r] <= inst.limit[r] + 1e-6).all()


def test_expected_time_weighting_beats_the_greedy_baseline():
    """The point of computing the bound: the LP's expected times are a much better ordering."""
    twin = _twin(dims=(10, 10, 6))
    inst = _cpit_from_twin(twin, periods=6, slack=0.55, n_resources=2)
    greedy = ob.toposort_schedule(inst, twin.precedence, weight="greedy")
    expected = ob.toposort_schedule(inst, twin.precedence, weight="expected")
    assert expected.npv > greedy.npv


def test_never_mines_outside_the_ultimate_pit():
    twin = _twin()
    inst = _cpit_from_twin(twin, periods=4)
    res = ob.toposort_schedule(inst, twin.precedence, weight="greedy")
    outside = ~twin.upit.in_pit
    assert not (res.period_of_block[outside] >= 0).any()


def test_local_search_never_loses_value_and_stays_feasible():
    twin = _twin(dims=(10, 10, 6))
    inst = _cpit_from_twin(twin, periods=6, slack=0.55, n_resources=2)
    base = ob.toposort_schedule(inst, twin.precedence, weight="greedy")
    better = ob.improve_schedule(inst, twin.precedence, base)
    assert better.npv >= base.npv - 1e-9
    per = better.period_of_block
    for b in np.nonzero(per >= 0)[0]:
        for p in twin.precedence.preds(b):
            assert per[p] >= 0 and per[p] <= per[b]
    for r in range(inst.n_resources):
        assert (better.per_period_resource[r] <= inst.limit[r] + 1e-6).all()


def test_solve_cpit_attaches_a_bound_and_a_gap():
    twin = _twin()
    inst = _cpit_from_twin(twin, periods=5, slack=0.6, n_resources=2)
    res, rels = ob.solve_cpit(inst, twin.precedence)
    assert res.bound is not None
    assert len(rels) == 2
    assert 0.0 <= res.gap_pct < 100.0
    assert res.npv <= res.bound + 1e-6 * abs(res.bound)


# ------------------------------------------------------------------------------------------------
# the controls
# ------------------------------------------------------------------------------------------------
def test_controls_all_pass_on_a_healthy_instance():
    twin = _twin()
    inst = _cpit_from_twin(twin, periods=4, n_resources=2)
    res, _ = ob.solve_cpit(inst, twin.precedence)
    c = ob.run_controls(inst, twin.precedence, res)
    assert c.duality_set_matches
    assert c.duality_bound_error == pytest.approx(0.0, abs=1e-6)
    assert c.bound_geq_feasible
    assert c.order_invariant
    assert c.all_pass


def test_zero_rate_unlimited_capacity_makes_the_ordering_irrelevant():
    twin = _twin()
    huge = float(twin.deposit.tonnage.sum()) + 1.0
    inst = ob.Cpit(
        name="flat", n_blocks=twin.values.shape[0], n_periods=3, discount_rate=0.0,
        value=twin.values.astype(float), limit=np.full((1, 3), huge),
        sense=np.full((1, 3), "L", dtype="<U1"),
        coef=twin.deposit.tonnage.reshape(1, -1).astype(float),
    )
    npvs = [ob.toposort_schedule(inst, twin.precedence, weight=w).npv for w in ("greedy", "gershon")]
    assert npvs[0] == pytest.approx(npvs[1], rel=1e-9)
    assert npvs[0] == pytest.approx(twin.upit.pit_value, rel=1e-9)


# ------------------------------------------------------------------------------------------------
# coherence
# ------------------------------------------------------------------------------------------------
def test_coherence_counts_components_and_the_narrowest_run():
    per = np.array([0, 0, 0, 0, 1])
    x = np.array([0, 1, 2, 5, 0])
    y = np.zeros(5, dtype=int)
    lev = np.zeros(5, dtype=int)
    c = ob.period_coherence(per, x, y, lev, 0)
    assert c.blocks == 4
    assert c.components == 2  # the run 0,1,2 and the isolated block at x = 5
    assert c.largest_component == 3
    assert c.largest_share == pytest.approx(0.75)
    assert c.min_width_blocks == 1


def test_coherence_over_a_real_schedule_is_reported_per_period():
    twin = _twin(dims=(10, 10, 6))
    inst = _cpit_from_twin(twin, periods=5, slack=0.55, n_resources=2)
    res, _ = ob.solve_cpit(inst, twin.precedence)
    g = twin.deposit.grid
    ix, iy, level = g.coord_arrays()
    rows = ob.schedule_coherence(res.period_of_block, ix, iy, level, inst.n_periods)
    assert len(rows) == inst.n_periods
    assert sum(r.blocks for r in rows) == res.mined_blocks
    for r in rows:
        assert r.components >= (1 if r.blocks else 0)
        assert 0.0 <= r.largest_share <= 1.0
