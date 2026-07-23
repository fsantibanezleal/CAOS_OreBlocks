#!/usr/bin/env python3
"""Reproducible validation + figure data for the oreblocks software note.

Three things, all from the real package:

  (1) INDEPENDENT cross-check of the exact max-closure UPIT solver against a totally-unimodular linear program
      solved by SciPy HiGHS (a completely different algorithm). Because the UPIT constraint matrix is a network
      matrix, the LP relaxation is integral, so the LP optimum equals the exact pit value; agreement validates the
      max-flow/min-cut solver end to end.
  (2) the stamped optima of the four deposit archetypes at a common size (pit value, blocks in pit, the solver's
      own closure + value-identity self-checks), and a small solve-time scaling table.
  (3) a vertical cross-section of a porphyry twin (grade field + the ultimate-pit mask) for the pit figure.

Writes ../data/upit_validation.json and ../data/porphyry_section.npz.

Run:  python upit_validation.py
Deps: oreblocks, numpy, scipy.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from oreblocks import ARCHETYPES, make_twin

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)


def lp_upit_value(values: np.ndarray, prec) -> float:
    """Independent UPIT optimum via the LP relaxation (integral by total unimodularity of the network matrix):
    max sum v_b x_b  s.t.  x_b - x_p <= 0 for each precedence arc (b needs p mined first),  0 <= x_b <= 1.
    Solved by SciPy HiGHS. Returns the optimal value."""
    v = np.asarray(values, dtype=np.float64)
    n = v.shape[0]
    rows, cols, dat = [], [], []
    r = 0
    for b in range(n):
        for k in range(prec.pstart[b], prec.pstart[b + 1]):
            p = int(prec.plist[k])
            rows += [r, r]; cols += [b, p]; dat += [1.0, -1.0]     # x_b - x_p <= 0
            r += 1
    A = csr_matrix((dat, (rows, cols)), shape=(r, n)) if r else None
    res = linprog(c=-v, A_ub=A, b_ub=np.zeros(r) if r else None,
                  bounds=[(0.0, 1.0)] * n, method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    return float(-res.fun)


def main():
    out = {"cross_check": [], "archetypes": [], "scaling": [], "note":
           "oreblocks exact max-closure UPIT vs SciPy-HiGHS LP; stamped archetype optima; solve-time scaling"}

    # (1) independent LP cross-check on modest instances of each archetype
    for arch in ARCHETYPES:
        tw = make_twin(arch, dims=(10, 10, 6), seed=7)
        mc = tw.upit.pit_value
        lp = lp_upit_value(tw.values, tw.precedence)
        rel = abs(mc - lp) / max(1.0, abs(lp))
        out["cross_check"].append({"archetype": arch, "n_blocks": int(tw.values.shape[0]),
                                   "max_closure_value": round(mc, 3), "lp_value": round(lp, 3),
                                   "rel_disagreement": rel})
        print(f"cross-check {arch}: max-closure={mc:.3f} lp={lp:.3f} rel={rel:.2e}")

    # (2) stamped optima of the four archetypes at a common size, with the solver self-checks
    for arch in ARCHETYPES:
        t0 = time.time()
        tw = make_twin(arch, dims=(24, 24, 12), seed=11)
        dt = time.time() - t0
        u = tw.upit
        gap = abs(u.pit_value - (u.sum_positive - u.maxflow))
        out["archetypes"].append({"archetype": arch, "n_blocks": int(tw.values.shape[0]),
                                  "pit_value": round(u.pit_value, 2), "n_in_pit": int(u.n_in_pit),
                                  "pit_fraction": round(u.n_in_pit / tw.values.shape[0], 3),
                                  "identity_gap": gap, "solve_seconds": round(dt, 3)})
        print(f"archetype {arch}: value={u.pit_value:.0f} in_pit={u.n_in_pit} gap={gap:.2e} {dt:.2f}s")

    # (3) solve-time scaling on porphyry
    for dims in [(16, 16, 8), (24, 24, 12), (32, 32, 16), (40, 40, 20)]:
        t0 = time.time()
        tw = make_twin("porphyry", dims=dims, seed=3)
        dt = time.time() - t0
        out["scaling"].append({"dims": list(dims), "n_blocks": int(np.prod(dims)),
                               "solve_seconds": round(dt, 3), "n_in_pit": int(tw.upit.n_in_pit)})
        print(f"scaling {dims}: {np.prod(dims)} blocks {dt:.2f}s")

    (DATA / "upit_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # (4) porphyry cross-section for the figure: grade + pit mask at mid-y
    tw = make_twin("porphyry", dims=(40, 40, 20), seed=3)
    g = tw.deposit.grid
    nx, ny, nz = g.nx, g.ny, g.nz
    iy = ny // 2
    grade = tw.deposit.grade.reshape(nz, ny, nx)      # index = (level*ny + iy)*nx + ix
    pit = tw.upit.in_pit.reshape(nz, ny, nx)
    sec_grade = grade[:, iy, :]                        # (nz, nx): level (up) x x
    sec_pit = pit[:, iy, :]
    np.savez(DATA / "porphyry_section.npz", grade=sec_grade, pit=sec_pit,
             nx=nx, nz=nz, block_m=getattr(g, "block_m", 10.0),
             pit_value=tw.upit.pit_value, n_in_pit=tw.upit.n_in_pit, n_blocks=tw.values.shape[0])
    print(f"wrote upit_validation.json + porphyry_section.npz (section {sec_grade.shape})")


if __name__ == "__main__":
    main()
