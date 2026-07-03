# oreblocks

[![CI](https://img.shields.io/github/actions/workflow/status/fsantibanezleal/CAOS_OreBlocks/ci.yml?branch=main&label=CI)](https://github.com/fsantibanezleal/CAOS_OreBlocks/actions)
[![License](https://img.shields.io/github/license/fsantibanezleal/CAOS_OreBlocks)](LICENSE)
[![Version](https://img.shields.io/github/v/tag/fsantibanezleal/CAOS_OreBlocks?label=version&sort=semver)](https://github.com/fsantibanezleal/CAOS_OreBlocks/tags)

**Synthetic 3-D ore-body block models of the MineLib nature** — seeded deposit archetypes with
per-block grades, bench (level) structure, slope precedence, UPIT economics with per-block optimal
destination, an **exact max-closure solver**, extraction states with **loading faces**, and MineLib
`.blocks/.prec/.upit` **read/write**. Deterministic given a seed; every generated instance is
clearly labelled SYNTHETIC.

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
