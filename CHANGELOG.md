# Changelog

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
