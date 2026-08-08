"""Multi-realisation scheduling: what geological uncertainty does to a plan, honestly framed.

Kriging is a smoother. It returns a conditional mean, so a schedule optimised on a single interpolated
block model is optimised for a deposit that does not exist. Meagher, Dimitrakopoulos and Avis
(Journal of Mining Science 50(3):508-526, 2014, doi:10.1134/S1062739114030132) summarise the
consequence: an open-pit gold example where "the consideration of geological uncertainty predicts a
NPV that is 50% less than that forecasted via conventional modeling".

**What this module does and does not claim.** It does NOT solve a two-stage stochastic integer
program. Ramazan and Dimitrakopoulos (Optimization and Engineering 14(2):361-380, 2013,
doi:10.1007/s11081-012-9186-2) do that, and report "approximately 10% higher NPV than the schedule
derived from the traditional approach" on an Australian gold mine. That is a different model and a
much larger one, and claiming it because a spread of NPVs is on screen would be exactly the
animating-an-assumption failure.

What it does is what Rio Tinto's planners actually do, in Blom, Pearce and Cote's words
(arXiv:2403.18213): "In practice, mining engineers using this platform solve a stochastic problem by
solving many instances of a deterministic one. Across these instances, parameters that capture aspects
such as price and grade are varied to reflect the uncertainty present in the original problem. Key
strategic decisions are made by analysing the resulting plans across these scenarios."

So: build an ensemble of equally probable realisations, solve the deterministic problem on each,
evaluate EVERY schedule against EVERY realisation, and report three things that a single number
cannot say:

- the **NPV distribution** of the plan built on the mean model, evaluated across realisations. This
  is the risk the deterministic plan actually carries.
- the **expected value of perfect information**: the average of per-realisation optima minus the
  expected value of the mean-model plan. It is what a perfect forecast would be worth, and therefore
  a ceiling on what any stochastic method could buy.
- the **best robust choice among the candidate plans**, by expected value and by P10, because a plan
  that is best on average can be third-best when things go badly.

The ensemble here is SYNTHETIC geological uncertainty from a seeded generator, not a conditionally
simulated orebody with drillhole data behind it. That is stated on every surface that shows it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .minelib_models import Cpit

__all__ = ["EnsembleResult", "RealisationSet", "evaluate_across", "perturb_values"]


@dataclass
class RealisationSet:
    """N equally probable realisations of the same deposit, as per-block value vectors."""

    values: np.ndarray  # (N, n)
    seed: int
    sigma: float
    note: str = (
        "SYNTHETIC geological uncertainty: a seeded lognormal multiplicative perturbation of the "
        "block values with a spatially correlated field, NOT a conditional simulation with drillhole "
        "data behind it."
    )

    @property
    def n_realisations(self) -> int:
        return int(self.values.shape[0])


def perturb_values(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    level: np.ndarray,
    *,
    n: int = 12,
    sigma: float = 0.25,
    correlation_blocks: float = 3.0,
    seed: int = 11,
) -> RealisationSet:
    """Build an ensemble by perturbing block values with a SPATIALLY CORRELATED field.

    Uncorrelated noise is the wrong model and it is the easy mistake: independent per-block error
    averages out over a pushback, so an uncorrelated ensemble makes uncertainty look harmless. Real
    grade error is correlated over tens of metres, which is why whole benches come in rich or poor
    together. Here a white field is smoothed with a separable box kernel of width
    ``correlation_blocks`` and applied multiplicatively in log space, so values keep their sign and
    the ore/waste classification can genuinely flip near the cutoff.
    """
    rng = np.random.default_rng(seed)
    nx, ny, nz = int(x.max()) + 1, int(y.max()) + 1, int(level.max()) + 1
    k = max(1, int(round(correlation_blocks)))
    out = np.empty((n, values.shape[0]), dtype=np.float64)

    for i in range(n):
        field3 = rng.normal(size=(nz, ny, nx))
        # separable box smoothing: cheap, deterministic, and correlated over ~k blocks
        for axis in (0, 1, 2):
            csum = np.cumsum(field3, axis=axis)
            csum = np.concatenate([np.zeros_like(np.take(csum, [0], axis=axis)), csum], axis=axis)
            lo = np.take(csum, range(0, csum.shape[axis] - k), axis=axis)
            hi = np.take(csum, range(k, csum.shape[axis]), axis=axis)
            smoothed = (hi - lo) / k
            pad = [(0, 0)] * 3
            pad[axis] = (0, field3.shape[axis] - smoothed.shape[axis])
            field3 = np.pad(smoothed, pad, mode="edge")
        # CENTRE before scaling. Box smoothing does not preserve the mean of a finite field, and a
        # multiplier built from an off-centre field is biased, which silently corrupts the one number
        # this module exists to report: whether the single-model forecast is optimistic.
        field3 -= float(field3.mean())
        field3 /= max(1e-9, float(field3.std()))
        mult = np.exp(sigma * field3[level, y, x] - 0.5 * sigma**2)
        # multiplicative on the REVENUE side only: costs do not become uncertain because the grade is
        out[i] = np.where(values > 0, values * mult, values * (2.0 - mult))

    return RealisationSet(values=out, seed=seed, sigma=sigma)


@dataclass
class EnsembleResult:
    """The risk readout: every candidate schedule scored on every realisation."""

    method_ids: list[str]
    #: (n_methods, n_realisations) discounted value of each plan under each realisation
    matrix: np.ndarray
    mean_model_npv: np.ndarray  # (n_methods,) value on the mean model, for reference
    per_realisation_best: np.ndarray  # (n_realisations,) best value any CANDIDATE achieves
    #: (n_realisations,) the value of a schedule RE-OPTIMISED on each realisation. Only this makes
    #: the expected value of perfect information meaningful; without it the "best per realisation"
    #: is merely the best of the plans you happened to bring, which is a much smaller quantity.
    per_realisation_optimum: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def expected(self) -> np.ndarray:
        return self.matrix.mean(axis=1)

    @property
    def p10(self) -> np.ndarray:
        return np.quantile(self.matrix, 0.10, axis=1)

    @property
    def p90(self) -> np.ndarray:
        return np.quantile(self.matrix, 0.90, axis=1)

    @property
    def best_by_expected(self) -> str:
        return self.method_ids[int(np.argmax(self.expected))]

    @property
    def best_by_p10(self) -> str:
        """The robust choice. It is often NOT the same plan as the best expected value."""
        return self.method_ids[int(np.argmax(self.p10))]

    @property
    def value_of_plan_selection(self) -> float:
        """What it would be worth to know WHICH of these candidate plans to use, per realisation.

        Small, and often exactly zero when one plan dominates. It is NOT the value of information,
        and calling it that would overstate by a wide margin.
        """
        return float(self.per_realisation_best.mean() - self.expected.max())

    @property
    def value_of_replanning(self) -> float | None:
        """What it is worth to RE-PLAN once the realisation is known, rather than commit in advance.

        A **lower bound on EVPI**, and it is important that it is only that. True EVPI needs the
        per-realisation OPTIMUM, and the per-realisation solve available here is a heuristic, so this
        quantity understates by however much that heuristic loses. Calling it EVPI would claim a
        ceiling this cannot certify.

        The caller must pass ``per_realisation_optimum`` as the best value it can actually achieve on
        each realisation, which means taking the maximum of the re-solve and every candidate plan
        already evaluated. Without that maximum the number can come out NEGATIVE, because a fixed plan
        can beat a heuristic re-solve on a lucky realisation, and a negative "value of information" is
        a naming error rather than a finding.
        """
        if self.per_realisation_optimum is None:
            return None
        return float(self.per_realisation_optimum.mean() - self.expected.max())

    @property
    def deterministic_optimism(self) -> np.ndarray:
        """Mean-model value minus expected value across realisations, per method.

        Positive means the single-model forecast is OPTIMISTIC, which is the effect the stochastic
        literature exists to name.
        """
        return self.mean_model_npv - self.expected


def evaluate_across(
    inst: Cpit,
    realisations: RealisationSet,
    schedules: dict[str, np.ndarray],
    per_realisation_optimum: np.ndarray | None = None,
) -> EnsembleResult:
    """Score every candidate schedule against every realisation.

    A schedule is a period assignment, so re-scoring it under a different value vector is exact: no
    re-solve, no approximation. Capacity feasibility is unaffected because the perturbation touches
    values and not tonnages.
    """
    d = inst.discount_factors()
    ids = list(schedules)
    n_m, n_r = len(ids), realisations.n_realisations
    mat = np.zeros((n_m, n_r))
    base = np.zeros(n_m)
    v0 = np.where(np.isfinite(inst.value), inst.value, 0.0)

    for i, key in enumerate(ids):
        per = np.asarray(schedules[key])
        mined = per >= 0
        base[i] = float((d[per[mined]] * v0[mined]).sum())
        for j in range(n_r):
            vj = realisations.values[j]
            mat[i, j] = float((d[per[mined]] * vj[mined]).sum())

    return EnsembleResult(
        method_ids=ids,
        matrix=mat,
        mean_model_npv=base,
        per_realisation_best=mat.max(axis=0),
        per_realisation_optimum=per_realisation_optimum,
        notes=[realisations.note],
    )
