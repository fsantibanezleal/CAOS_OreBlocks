# oreblocks

[![CI](https://img.shields.io/github/actions/workflow/status/fsantibanezleal/CAOS_OreBlocks/ci.yml?branch=main&label=CI)](https://github.com/fsantibanezleal/CAOS_OreBlocks/actions)
[![License](https://img.shields.io/github/license/fsantibanezleal/CAOS_OreBlocks)](LICENSE)
[![Version](https://img.shields.io/github/v/tag/fsantibanezleal/CAOS_OreBlocks?label=version&sort=semver)](https://github.com/fsantibanezleal/CAOS_OreBlocks/tags)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21512088-blue)](https://doi.org/10.5281/zenodo.21512088)

Software note (CC-BY-4.0): *"oreblocks: License-Free Synthetic Ore-Body Block Models with a Stamped Exact
Ultimate-Pit Optimum"*, concept DOI [10.5281/zenodo.21512088](https://doi.org/10.5281/zenodo.21512088) (source in
[`manuscripts/ore-body-twins/`](manuscripts/ore-body-twins/)). It gives the deposit archetypes, the exact
max-closure ultimate-pit solver, and an independent LP cross-check to machine precision (~1e-15).

**Synthetic 3-D ore-body block models of the MineLib nature, and their scheduling** — seeded deposit
archetypes with per-block grades, bench (level) structure, slope precedence, UPIT economics with
per-block optimal destination, an **exact max-closure solver**, extraction states with **loading
faces**, MineLib `.blocks/.prec/.upit/.cpit/.pcpsp` **read/write**, the **certified CPIT LP bound**
by the critical multiplier algorithm, the **TopoSort** rounding heuristics, and **spatial-coherence**
metrics. Deterministic given a seed; every generated instance is clearly labelled SYNTHETIC.

Why: per-block ground truth on real mines is licensed or proprietary (MineLib grants academic
download only, no redistribution). oreblocks generates instances of the same *nature* — 3-D
benches, grades, precedence, net values — with a **stamped exact optimum**, so solvers, dispatch
simulators and teaching apps get license-free realistic instances with known-by-construction
answers.

## Install

```bash
pip install oreblocks
```

## Quickstart — a MineLib-format twin with a stamped optimum

```python
from oreblocks import make_twin

twin = make_twin("porphyry", dims=(20, 20, 10), seed=42)
print(twin.upit.pit_value, twin.upit.n_in_pit)   # the EXACT optimum, stamped
twin.write("out/")   # -> twin-*.blocks / .prec / .upit / .meta.json (MineLib format)
```

The emitted triplet is byte-consumable by any MineLib UPIT reader. Cross-validated against an
independent TypeScript min-cut engine (CAOS PitForge, which reproduces the published newman1 /
zuck_small / kd optima): relative disagreement ~1e-11 on a 4,000-block twin.

## Scheduling: which blocks, and WHEN

The ultimate pit says which blocks are worth mining. The **constrained pit limit problem** (CPIT)
says when: assign every block a period so slope precedence holds in every period, per-period
capacities hold, and discounted value is maximised. It is NP-hard, so what ships is a **certified
upper bound** plus **feasible schedules**, with the gap between them reported rather than hidden.

```python
import oreblocks as ob

inst = ob.read_cpit("newman1.cpit")                    # periods, discount rate, capacities
prec = ob.read_prec("newman1.prec", inst.n_blocks)
result, relaxations = ob.solve_cpit(inst, prec)        # bound, schedule, local search
print(f"{result.npv:,.0f}  bound {result.bound:,.0f}  gap {result.gap_pct:.2f}%")

controls = ob.run_controls(inst, prec, result)         # duality, bound, order invariance
assert controls.all_pass
```

The bound needs **no LP solver**. Chicoisne et al. 2012 (Operations Research 60(3):517-528,
[doi:10.1287/opre.1120.1050](https://doi.org/10.1287/opre.1120.1050), Theorem 3.1) show the CPIT LP
relaxation is solved exactly in `O(mn log n)` for one resource per period, as a sequence of
parametric nested pits, which are maximum closures, which this package already computes exactly.

On the published `newman1.cpit` this implementation returns a certified bound of **24 487 410**
against the published LP bound of **24 486 549** (relative difference 3.5e-5, and the residual is in
the right direction because the single-resource relaxation is looser than the joint bound).

Full detail, including the three file-format traps and the explicit list of what is not implemented
(stockpiles, blending, minimum-production constraints, the exact `C-PIT[D]` local search, stochastic
scheduling): **[docs/scheduling.md](docs/scheduling.md)**.

## Pieces

| Module | What |
|---|---|
| `BlockGrid` | regular grid; LEVELS increase upward (the MineLib convention) |
| `make_deposit` | seeded archetypes: `porphyry`, `vein`, `layered`, `core_halo` (trend + correlated noise) |
| `Econ` / `block_values` | UPIT net value at the optimal destination (floating cutoff = the max) |
| `build_precedence` | slope-cone template one level up (45° cubic → the classic 9-point), CSR |
| `solve_upit` | exact Picard max-closure → Dinic min-cut; closure + value-identity self-checks |
| `extraction_state` / `loading_faces` | top-down bench extraction + seeded k-means shovel faces (grade at face, ore fraction, tonnes) — the bridge to haulage simulators |
| `write_minelib` / `read_*` | the `.blocks/.prec/.upit` triplet + a meta sidecar with the stamped optimum |

## Convention notes

- Levels (z) increase **upward**: level 0 is the deepest bench — exactly how published MineLib
  instances index (verified against newman1). Depth-down viewers flip with `z_down = nz-1-level`.
- `.blocks` free columns written by oreblocks are documented in the meta sidecar:
  `grade (mass fraction) · tonnage (t) · density (t/m³)`.
- Nothing here downloads or redistributes published MineLib data.

## Used by

- **minehaulsim** (haulage DES) — geology-grounded scenarios: loading faces with grade/bench.
- **CAOS PitForge** — license-free synthetic twins next to the published-instance lane.

## License

Apache-2.0.
