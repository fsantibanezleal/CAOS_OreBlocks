# Changelog

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
