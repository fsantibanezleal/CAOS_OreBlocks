# Changelog

## [0.5.1] - 2026-08-10

### Fixed
- **Lane's market-limiting cutoff was wrong by a factor of `1/recovery`.** Lane's is
  `g = h / (y (p - k - F/K))`, which in this notation is `h / (margin - recovery * F / K)`; the code
  divided by recovery where Lane multiplies, inflating the opportunity charge by `1/recovery^2`.
  Measured against the package's own test parameters: +1.6 percent at a recovery of 0.88, +5.8 at
  0.70, +18.8 at 0.50, +113.8 at 0.30. At those parameters that cutoff is also the median, so
  `CutoffPolicy.optimum` returned the wrong number as well.

### Changed
- **`balancing_mine_mill` and `balancing_mill_market` are now `midpoint_mine_mill` and
  `midpoint_mill_market`.** Lane's balancing cutoffs are the grades at which two capacities are
  exhausted at once, which is a property of the deposit's GRADE-TONNAGE CURVE. `lane_cutoffs` is given
  economics only and has no distribution to integrate, so it never computed them. An interpolation
  between two limiting cutoffs is a usable number; calling it Lane's balancing cutoff was the part
  that was not true.

## [0.5.0] - 2026-08-10

### Changed
- **`sliding_window_schedule` is now a sliding time window.** It was not one. The previous version
  scheduled each window with the same greedy TopoSort the SOTA rung uses, then undid every placement
  past the frozen prefix and returned its capacity; since a placement consumes only its own period's
  capacity, nothing inside the window could influence the prefix. Measured: `window` of 1, 2, 3, 5, 8
  and T gave BIT-IDENTICAL schedules on three instances, zero blocks moved. The rung carried
  Cullenbine, Wood and Newman (doi:10.1007/s11590-011-0306-2) for a look-ahead it did not perform.

  It now solves each window JOINTLY as an integer program, with everything past the window aggregated
  into one optimistic tail, so a block worth taking in period 1 only because of what it unlocks in
  period 3 is visible to the solver. Cumulative variables, so precedence is one row of two entries per
  arc per slot: written with per-slot assignment variables the matrix grew quadratically in the window
  and one slide took ninety seconds. Measured on a 1008-block twin, window 3 against window 1: 186
  blocks move and NPV rises 4.2 percent. It needs scipy (`oreblocks[milp]`).

  Two approximations, both named in the docstring and in the result's notes: the tail is optimistic
  (its discount factor is that of the first tail period), and the candidate set is sized by TONNAGE to
  cover `cover` times the window's capacity.

- **It REFUSES to be starved.** If the candidate set needed to fill a period's capacity exceeds
  `cand_max`, it raises with both numbers instead of returning a schedule. A flat cap of 150 blocks
  cut the objective from 39.7 M to 10.4 M on a 1008-block twin, and a silent cap on a 14,400-block one
  mined 550 blocks and reported an NPV for it. A starved answer still looks like a schedule.

### Added
- Tests: the window must CHANGE the schedule, any loss against no look-ahead must be inside the
  per-slide MIP gap (it loses 0.012 percent on one instance and wins 4.2 percent on another, which is
  the honest property of a heuristic with an optimistic tail), and the starvation guard must raise.

## [0.4.1] - 2026-08-10

### Added
- **`exact_local_search(..., time_limit=None)`**: stop on the relative MIP gap alone. A wall-clock
  budget makes the answer depend on the machine and its load, which is the right trade for an
  interactive call and the wrong one for a bake whose artifacts are committed as evidence. A
  downstream product found its headline gap was not reproducible from `(params, seed)` because this
  rung is the reported best on nine cases of thirteen and was stopping on eight seconds of CPU. The
  option is now omitted from the solver call rather than passed as `None`, and a test asserts two
  runs agree block for block.

## [0.4.0] - 2026-08-10

The joint bound now runs on a real deposit. It could not before, and the product said so rather than
hiding it; this is the fix rather than a bigger budget.

### Added
- **`fastcut`: a compiled maximum-closure path.** The pure-Python Dinic is fine on a block model and
  loses on a TIME-EXPANDED graph, which is where the Bienstock-Zuckerberg pricing problem lives: a
  10,976-block deposit over ten periods is 109,760 nodes and 972,904 arcs.
  `scipy.sparse.csgraph.maximum_flow` is compiled and takes INTEGER capacities, so node weights are
  scaled and rounded UP, which makes the error one-sided: the reported value can only OVER-estimate
  the true maximum closure, by at most `n / scale`. That direction is the safety argument, because
  `L(pi)` bounds the optimum for every dual vector and a pricing solve that came in low would produce
  a number that is not a bound.
- **`solve_gpcp_lp` certifies its answer.** The rounded pricing is fast enough to SEARCH but its
  slack on a time-expanded graph is about `1e-4` relative, the same order as the tightening the joint
  bound exists to measure, so a bound built from it cannot answer the question it was asked. Since
  `L(pi)` is valid for every `pi`, one EXACT solve at the best dual vector found turns the search
  into a certificate: `pricing_slack` comes back at exactly `0.0` and the number is the real thing.
  Measured on that twin: 15 iterations, 15 seconds, and BZ agrees with the critical multiplier
  algorithm to 1.3e-8 relative on a single resource, which is the check that says both are right.
- `solve_gpcp_lp(..., time_budget_s=...)`. BZ returns a valid upper bound at EVERY iteration, so a
  run that stops early is looser rather than wrong, and `converged` says which one you got.
- `BzResult` gains `pricing_slack`, `pricing_solver`, `certified` and `seconds`.

### Measured, and written into the source
- `scipy.sparse.csgraph.maximum_flow` accepts an int64 matrix and is WRONG above a total capacity of
  `2**31`: on a closure whose true value is 769,925,542 it returns +3,210 at `2**30` (inside the
  rounding slack, as designed) and +15,491,856 at `2**31`, which is the whole positive mass, meaning
  the flow came back as zero. It does not raise and it does not warn. `test_fastcut.py` re-measures
  the ceiling so a scipy release that moves it is caught here rather than in a bound that drifted.

## [0.3.1] - 2026-08-08

### Added
- **`solve_cpit(..., bound=False)`**: schedule without computing the certified bound. The bound is a
  parametric family of maximum closures per resource, hundreds of them; a schedule from a
  combinatorial weight is one closure and a topological pass. A caller that needs many schedules and
  no bound was paying a factor of a hundred for a number it discarded, which is what an uncertainty
  ensemble does once per realisation. Measured: a thirteen-case downstream bake spent four hours and
  finished one case inside that loop. `method='expected'` needs the relaxation by definition and is
  rejected with `bound=False` rather than silently downgraded, and `ScheduleResult.bound` is `None`
  so nothing downstream can mistake a missing bound for a computed one.

## [0.3.0] - 2026-08-08

The rest of the ladder: the joint bound, the exact local search, the destination decision, the
cutoff-grade theory, operability, and geological uncertainty. Everything that 0.2 named as not
implemented and every rung the mine-scheduling literature actually uses.

### Added
- **`bz`: the Bienstock-Zuckerberg decomposition**, as column generation over the LINEAR hull of the
  precedence polytope with orthogonal 0-1 generator matrices, per Munoz, Espinoza, Goycoolea, Moreno,
  Queyranne and Rivera Letelier (doi:10.1007/s10589-017-9946-1). The pricing problem is a maximum
  closure, so it rides on the same max-flow the package already ships, and it terminates on the
  duality certificate rather than an iteration cap. `cpit_bz_bound` gives the JOINT bound over all
  resources, which `cpit_bound_two_resources` cannot: that one relaxes all but one resource and keeps
  the smallest result, which is certified and loose. On the published `newman1.cpit` BZ converges in
  9 iterations and 1.6 s to 24,486,184 against Algorithm 4's 24,487,410, and sits correctly below the
  published PCPSP LP bound of 24,486,549.
- **`sliding_window_schedule`**: the sliding time window heuristic (Cullenbine, Wood and Newman,
  doi:10.1007/s11590-011-0306-2), the industrial baseline that Rio Tinto's platform uses to seed its
  large neighbourhood search.
- **`refine.exact_local_search`**: the exact `C-PIT[D]` neighbourhood (Chicoisne et al. 2012 section
  3.3), all three of their neighbourhood constructions, each re-solved as a restricted MILP. This is
  the rung `improve_schedule` is explicitly NOT.
- **`destinations`**: PCPSP and OPBSP. `destination_toposort` chooses each block's destination against
  the remaining capacity, so the cutoff grade becomes an OUTPUT of the schedule rather than an input;
  `solve_opbsp_exact` solves the fully binary formulation (Jelvez et al. 2018 equations 3-10) exactly
  with HiGHS, and returns `None` rather than passing a heuristic off as exact when the instance is
  too large.
- **`refine.lane_cutoffs`**: Lane's three limiting cutoffs and the balancing cutoffs between them,
  with the opportunity cost that makes the economic cutoff decline over the life of a mine.
- **`refine.enforce_min_width`**: adaptive opening toward a minimum mining width (Bai et al. 2018),
  reporting how many slivers it absorbed and what the operability cost in NPV.
- **`stochastic`**: a spatially correlated, mean-preserving ensemble, every candidate plan scored on
  every realisation, P10 and P90, the robust choice by P10, the optimism of the single-model forecast,
  and a true EVPI that requires the problem actually re-solved per realisation. The plan-selection
  value is reported separately and never labelled EVPI.
- Optional extra `oreblocks[milp]` (scipy) for the BZ master, the exact local search, and OPBSP.

### Fixed
- The sliding window re-planned blocks that had already consumed capacity, double-booking the fleet
  and collapsing the objective to a third of its value. The frozen prefix is now the only decision and
  everything after it is released before the next slide.
- `enforce_min_width` checked only predecessors, so moving a block later could be overtaken by a
  successor already scheduled ahead of it. Precedence cuts both ways and is now checked both ways.
- The ensemble perturbation was not mean-preserving (box smoothing does not centre a finite field),
  which biased the one number the module exists to report.

### Notes
- `Z_BZ = Z_LP` still holds and is still stated: BZ is a speed result and a JOINT-bound result, never
  a tighter-than-LP result. On a single resource it agrees with the critical multiplier algorithm to
  machine precision, which is the strongest correctness check in the suite.
- Still not implemented, on purpose: stockpiles with inventory and rehandle (bilinear), blending and
  other general side constraints, minimum-production (`sense = 'G'`) constraints, and true two-stage
  stochastic integer programming.

## [0.2.1] - 2026-08-07

### Fixed
- `read_blocks` called `float()` on every column after `id x y z` and therefore crashed on the
  published `newman1.blocks`, whose first free column is a rock-type code (`FRWS`, `FROR`, `OXOR`).
  Non-numeric tokens now become `NaN` in `free` and are kept verbatim in a new `labels` dict, so a
  real published file reads and nothing is guessed.
- The critical multiplier algorithm re-derived the same parametric pits once per period. The
  break-points of the family do not depend on the period, only the target capacity does, so they are
  now solved lazily into a shared cache and every new solve is restricted to the smallest known pit
  that must contain it. Measured on the published `newman1.cpit`: 91 and 89 closure solves before,
  34 and 45 after, and 5.4 s to 1.3 s. On a 6912-block twin with two resources: 347 and 337 solves
  before, 129 and 73 after, 36 s to 11.7 s. Identical bounds and schedules.
- The bisection now stops on the DUALITY CERTIFICATE rather than on an interval width. Strong
  duality gives `CP(U) = min_lambda [ UPL(v - lambda a) + lambda U ]`, so the search refines until
  the primal estimate and that dual expression agree; when they do, the two bracketing pits are
  provably consecutive break-points. The previous heuristic stop (a midpoint reproducing a bracket
  twice) was not a proof and could return a non-adjacent bracket, which the duality assertion then
  correctly rejected.

## [0.2.0] - 2026-08-07

The scheduling half of MineLib. The package could describe a deposit and solve its ultimate pit; it
can now read the published scheduling model files, produce a certified bound on the discounted NPV,
and produce feasible schedules against it.

### Added
- `minelib_models`: `.cpit` and `.pcpsp` read and write (`read_cpit`, `read_pcpsp`, `write_cpit`,
  `write_pcpsp`, `Cpit`, `Pcpsp`). Format measured on the published `newman1` files. Handles the
  sparse coefficient section, the large-negative destination sentinel (`FORBIDDEN_VALUE`), and
  `Pcpsp.to_cpit()` for the fix-the-destination reduction.
- `schedule.cpit_lp_relaxation`: the critical multiplier algorithm (Chicoisne et al. 2012,
  Operations Research 60(3):517-528, doi:10.1287/opre.1120.1050, Theorem 3.1). Exact optimum of the
  CPIT LP relaxation for one resource constraint per period, in `O(mn log n)`, with no LP solver:
  parametric nested pits plus a convex combination. Verified against strong duality at every period.
- `schedule.cpit_bound_two_resources`: Algorithm 4, a certified bound when several resources bind.
- `schedule.toposort_order` / `toposort_schedule`: the TopoSort family (Algorithms 2 and 3) with the
  three published weightings, `greedy` (GrTS), `gershon` (GeTS) and `expected` (ExTS, seeded by the
  LP expected extraction times).
- `schedule.improve_schedule`: shift local search, pull value forward and push cost back. Both moves
  strictly improving under discounting and feasible by construction.
- `schedule.solve_cpit`: the whole ladder in one call, with the bound attached and the gap reported.
- `schedule.run_controls`: duality (rate 0 with unlimited capacity reproduces the exact ultimate pit
  block for block and its value), bound (certified bound at least any feasible objective), and order
  invariance.
- `coherence`: per-period connected components, largest-component share and narrowest mined run.
  Chicoisne et al. predict block-level schedules scatter; this measures it rather than hoping.
- `upit.max_closure_within`: maximum closure restricted to a candidate set, which is what makes the
  parametric sequence cheap (each solve runs on the shrinking difference).
- `docs/scheduling.md`: the deep page, with the formulation, the theorem, the file formats and the
  explicit list of what is not implemented.

### Changed
- Licence moved from Apache-2.0 to **MIT**, aligning with the rest of the CAOS line.
- `__version__` 0.1.0 to 0.2.0.

### Notes
- On the published `newman1.cpit` (1060 blocks, 6 periods, rate 0.08, two capacities) the certified
  bound is **24 487 410** against the published LP bound of **24 486 549**, a relative difference of
  3.5e-5, with the residual in the expected direction because the single-resource relaxation is
  looser than the joint bound. That measurement is also what settled the discount convention: the
  first period is undiscounted, `1/(1+r)^(t-1)`. The alternative gives 22 673 528, off by 8 percent.
- `Z_BZ = Z_LP` (Munoz et al., doi:10.1007/s10589-017-9946-1): the Bienstock-Zuckerberg
  decomposition is a speed result, not a tighter bound. It is not implemented here.
- Not implemented, on purpose and stated in the docs: stockpiles, blending and other general side
  constraints, minimum-production (`sense = 'G'`) constraints, the exact `C-PIT[D]` local search,
  and stochastic scheduling.

## [0.1.0] — 2026-07-03

Initial release.

### Added
- `BlockGrid` (MineLib level convention: z up), seeded named RNG streams.
- Deposit archetypes `porphyry / vein / layered / core_halo` — trend + box-smoothed correlated
  noise, mass-fraction grades, deterministic per (archetype, dims, seed).
- UPIT economics (`Econ`, `block_values`): net value at the per-block optimal destination —
  the `.upit` semantics (max of waste/ore destinations, floating cutoff).
- Slope precedence (`build_precedence`): reduced one-level-up template (rx = round(dz/(dx·tanθ)),
  clamped ≥ 1), CSR layout matching `.prec` semantics.
- Exact UPIT solver (`solve_upit`): Picard max-closure → iterative Dinic min-cut with closure
  feasibility + value-identity self-checks.
- Extraction states (`extraction_state`): top-down bench extraction by tonnage progress with a
  deterministic centre-out partial-bench order; loading faces (`loading_faces`): seeded k-means
  clusters with mean grade, ore fraction, tonnes and bench elevation.
- MineLib IO: `write_minelib` (`.blocks/.prec/.upit` + `.meta.json` sidecar with the stamped
  exact optimum and a SYNTHETIC statement) and `read_blocks/read_prec/read_upit/read_meta`.
- `make_twin`: one-call instance generation + exact solve + write.

### Verified
- 14/14 tests: closed-form inverted-pyramid oracle (9 blocks, value 2), determinism, IO
  round-trips, twin re-read → stamped optimum, extraction/face invariants.
- Cross-engine validation: a 4,000-block twin solved by the independent CAOS PitForge TypeScript
  min-cut engine agrees with the stamped optimum to rel ~3e-11 (identical pit membership).
