"""The compiled maximum-closure path, and the integer arithmetic it rests on.

Two implementations of the same problem is two chances to be wrong, and the second one is the fast
one, so it is checked against the first rather than trusted.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

import oreblocks as ob
from oreblocks.fastcut import _TOTAL_CAPACITY_CEILING, has_fast_cut, max_closure_fast
from oreblocks.upit import max_closure_within


def _twin(dims=(20, 20, 12), seed=101):
    return ob.make_twin("porphyry", dims, seed=seed)


def test_fast_closure_matches_the_exact_one_set_for_set() -> None:
    twin = _twin()
    v = twin.values.astype(float)
    exact = max_closure_within(v, twin.precedence, np.ones(v.shape[0], dtype=bool))
    fast = max_closure_fast(v, twin.precedence)
    assert np.array_equal(exact, fast.mask)
    assert v[fast.mask].sum() == pytest.approx(v[exact].sum())


def test_the_reported_value_never_under_estimates() -> None:
    """The direction is the whole safety argument.

    The Bienstock-Zuckerberg certificate is that ``L(pi)`` bounds the optimum for every dual vector.
    A pricing solve that came in LOW would produce a number that is not a bound, and nothing
    downstream could tell. Rounding the node weights up makes the error one-sided by construction.
    """
    rng = np.random.default_rng(11)
    for seed in (3, 5, 7):
        twin = _twin(dims=(12, 12, 8), seed=seed)
        for _ in range(3):
            w = twin.values.astype(float) * rng.uniform(0.2, 2.0, twin.values.shape[0])
            exact = max_closure_within(w, twin.precedence, np.ones(w.shape[0], dtype=bool))
            fast = max_closure_fast(w, twin.precedence)
            true_value = float(w[exact].sum())
            assert fast.value >= true_value - 1e-9
            assert fast.value <= true_value + fast.slack + 1e-9


def test_the_integer_ceiling_is_where_it_was_measured() -> None:
    """`scipy.sparse.csgraph.maximum_flow` takes an int64 matrix and is wrong past 2**31.

    It does not raise and it does not warn: it returns a plausible number. This asserts the failure
    is still ABOVE the ceiling the module uses, so a scipy release that fixes it, or one that makes it
    worse, is caught here rather than in a bound that quietly moved.
    """
    twin = _twin(dims=(14, 14, 8), seed=17)
    v = twin.values.astype(float)
    prec = twin.precedence
    n = v.shape[0]
    truth = float(v[max_closure_within(v, prec, np.ones(n, dtype=bool))].sum())
    positive = float(v[v > 0].sum())

    def solve_at(total_capacity: float) -> float:
        scale = total_capacity / positive
        caps = np.ceil(v * scale).astype(np.int64)
        src = np.where(caps > 0, caps, 0)
        snk = np.where(caps < 0, -caps, 0)
        inf = int(src.sum()) + 1
        counts = (prec.pstart[1:] - prec.pstart[:-1]).astype(np.int64)
        tails = np.repeat(np.arange(n, dtype=np.int64), counts)
        rows = np.concatenate([np.full(int((src > 0).sum()), n), np.nonzero(snk > 0)[0], tails])
        cols = np.concatenate([np.nonzero(src > 0)[0], np.full(int((snk > 0).sum()), n + 1), prec.plist])
        data = np.concatenate([src[src > 0], snk[snk > 0], np.full(tails.shape[0], inf, dtype=np.int64)])
        g = csr_matrix((data, (rows, cols)), shape=(n + 2, n + 2), dtype=np.int64)
        return (int(src.sum()) - int(maximum_flow(g, n, n + 1).flow_value)) / scale

    at_ceiling = solve_at(float(_TOTAL_CAPACITY_CEILING))
    assert at_ceiling >= truth - 1e-6
    assert (at_ceiling - truth) / truth < 1e-4, "the ceiling no longer gives a usable answer"


def test_the_fast_path_is_faster_where_it_matters() -> None:
    """On a BLOCK MODEL the two are comparable; on a TIME-EXPANDED graph they are not.

    The time-expanded graph is the only reason this module exists, so that is what is timed.
    """
    twin = _twin(dims=(20, 20, 12), seed=101)
    prec = twin.precedence
    v = twin.values.astype(float)
    n = v.shape[0]
    periods = 8
    tonnes = np.asarray(twin.deposit.tonnage, dtype=float)
    cap = 1.15 * tonnes.sum() / periods
    inst = ob.Cpit(
        name="t", n_blocks=n, n_periods=periods, discount_rate=0.10, value=v,
        limit=np.array([[cap] * periods]), sense=np.array(["L"]), coef=np.array([tonnes]),
    )
    g = ob.cpit_to_gpcp(inst, prec)
    from oreblocks.bz import _build_csr

    _build_csr(g)
    w = np.array(g.c, dtype=np.float64)
    allb = np.ones(g.n, dtype=bool)

    t0 = time.perf_counter()
    fast = max_closure_fast(w, g._csr, allb)  # noqa: SLF001
    t_fast = time.perf_counter() - t0
    t0 = time.perf_counter()
    exact = max_closure_within(w, g._csr, allb)  # noqa: SLF001
    t_exact = time.perf_counter() - t0

    assert np.array_equal(fast.mask, exact)
    assert t_fast < t_exact, f"fast {t_fast:.3f}s is not faster than exact {t_exact:.3f}s"


@pytest.mark.skipif(not has_fast_cut(), reason="needs the milp extra")
def test_bz_on_a_single_resource_still_equals_the_critical_multiplier_bound() -> None:
    """The strongest check in the package, now on a graph the exact path could not finish.

    Two entirely different algorithms computing the same LP. If they disagree one is wrong, and no
    amount of plausible output says which. This runs at a scale that only works because the pricing
    is compiled AND the final bound is certified by one exact solve.
    """
    twin = _twin(dims=(20, 20, 12), seed=101)
    prec = twin.precedence
    v = twin.values.astype(float)
    n = v.shape[0]
    periods = 8
    tonnes = np.asarray(twin.deposit.tonnage, dtype=float)
    cap = 1.15 * tonnes.sum() / periods
    inst = ob.Cpit(
        name="t", n_blocks=n, n_periods=periods, discount_rate=0.10, value=v,
        limit=np.array([[cap] * periods]), sense=np.array(["L"]), coef=np.array([tonnes]),
    )
    bz = ob.cpit_bz_bound(inst, prec, max_iter=60, time_budget_s=600.0)
    alg4, _ = ob.cpit_bound_two_resources(inst, prec)

    assert bz.converged
    assert bz.pricing_slack == 0.0, "the reported bound must be certified, not rounded"
    assert bz.bound == pytest.approx(alg4, rel=1e-6)
