# Changelog

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
