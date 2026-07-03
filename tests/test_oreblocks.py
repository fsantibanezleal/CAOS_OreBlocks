"""oreblocks test suite — closed-form oracle, determinism, IO round-trips, twin e2e."""

from __future__ import annotations

import numpy as np
import pytest

from oreblocks import (
    ARCHETYPES,
    BlockGrid,
    Econ,
    block_values,
    build_precedence,
    cutoff_grade,
    extraction_state,
    loading_faces,
    make_deposit,
    make_twin,
    read_blocks,
    read_meta,
    read_prec,
    read_upit,
    slope_offsets,
    solve_upit,
)

# ---------------------------------------------------------------- grid


def test_grid_indexing_roundtrip():
    g = BlockGrid(nx=4, ny=3, nz=2)
    seen = set()
    for lv in range(2):
        for iy in range(3):
            for ix in range(4):
                i = g.index(ix, iy, lv)
                assert g.coords(i) == (ix, iy, lv)
                seen.add(i)
    assert seen == set(range(g.n_blocks))
    x, y, level = g.coord_arrays()
    assert (x[g.index(3, 2, 1)], y[g.index(3, 2, 1)], level[g.index(3, 2, 1)]) == (3, 2, 1)


def test_grid_validation():
    with pytest.raises(ValueError):
        BlockGrid(nx=0, ny=1, nz=1)
    g = BlockGrid(nx=2, ny=2, nz=2)
    with pytest.raises(IndexError):
        g.index(2, 0, 0)


# ---------------------------------------------------------------- fields


def test_deposits_deterministic_and_physical():
    g = BlockGrid(nx=16, ny=16, nz=8)
    for arch in ARCHETYPES:
        a = make_deposit(g, arch, seed=11)
        b = make_deposit(g, arch, seed=11)
        assert np.array_equal(a.grade, b.grade), f"{arch} not deterministic"
        c = make_deposit(g, arch, seed=12)
        assert not np.array_equal(a.grade, c.grade), f"{arch} ignores the seed"
        assert a.grade.min() >= 0.0 and a.grade.max() <= 1.0
        assert a.grade.max() > 0.005, f"{arch} produced no mineralisation"
        assert (a.tonnage > 0).all() and (a.density > 0).all()
        assert a.meta["synthetic"] is True


# ---------------------------------------------------------------- economics


def test_values_are_optimal_destination():
    g = BlockGrid(nx=2, ny=1, nz=1)
    dep = make_deposit(g, "porphyry", seed=1)
    dep.grade[0] = 0.0  # pure waste
    dep.grade[1] = 0.05  # rich ore
    econ = Econ()
    v = block_values(dep, econ)
    t = dep.tonnage
    assert v[0] == pytest.approx(-econ.mining_cost * t[0])  # waste destination
    ore = (0.05 * econ.recovery * econ.price - econ.processing_cost) * t[1] - econ.mining_cost * t[1]
    assert v[1] == pytest.approx(ore)
    assert 0 < cutoff_grade(econ) < 0.01


# ---------------------------------------------------------------- precedence


def test_slope_template_45_is_nine_point():
    g = BlockGrid(nx=5, ny=5, nz=3)
    offs = slope_offsets(g, 45.0)
    assert len(offs) == 9
    prec = build_precedence(g, 45.0)
    # an interior block depends on exactly 9 blocks one level up
    b = g.index(2, 2, 0)
    preds = prec.preds(b)
    assert preds.shape[0] == 9
    for p in preds:
        _, _, lv = g.coords(int(p))
        assert lv == 1
    # the top level depends on nothing
    assert prec.preds(g.index(2, 2, 2)).shape[0] == 0


def test_flatter_slope_widens_the_cone():
    g = BlockGrid(nx=9, ny=9, nz=2)
    assert len(slope_offsets(g, 30.0)) > len(slope_offsets(g, 45.0))


# ---------------------------------------------------------------- upit (closed-form oracle)


def _pyramid_instance():
    """5x1x3, one buried +10 block at the bottom centre, every other block -1, 45 deg."""
    g = BlockGrid(nx=5, ny=1, nz=3)
    v = np.full(g.n_blocks, -1.0)
    v[g.index(2, 0, 0)] = 10.0
    return g, v, build_precedence(g, 45.0)


def test_upit_inverted_pyramid_oracle():
    g, v, prec = _pyramid_instance()
    r = solve_upit(v, prec)
    # the optimal pit is EXACTLY the 9-block inverted pyramid: value 10 - 8 = 2
    assert r.n_in_pit == 9
    assert r.pit_value == pytest.approx(2.0)
    assert r.in_pit[g.index(2, 0, 0)]
    assert r.in_pit[g.level_slice(2)].sum() == 5  # full top row of the cone
    assert r.pit_value == pytest.approx(r.sum_positive - r.maxflow)


def test_upit_unprofitable_stays_in_the_ground():
    g = BlockGrid(nx=5, ny=1, nz=3)
    v = np.full(g.n_blocks, -1.0)
    v[g.index(2, 0, 0)] = 7.0  # cone costs 8 -> mining nets -1
    r = solve_upit(v, build_precedence(g, 45.0))
    assert r.n_in_pit == 0
    assert r.pit_value == 0.0


def test_upit_rejects_mismatched_sizes():
    g, v, prec = _pyramid_instance()
    with pytest.raises(ValueError):
        solve_upit(v[:-1], prec)


# ---------------------------------------------------------------- minelib io


def test_minelib_roundtrip(tmp_path):
    twin = make_twin("vein", (10, 8, 6), seed=5)
    paths = twin.write(tmp_path)
    blocks = read_blocks(paths["blocks"])
    n = twin.deposit.grid.n_blocks
    assert blocks["x"].shape[0] == n
    x, y, level = twin.deposit.grid.coord_arrays()
    assert np.array_equal(blocks["x"], x) and np.array_equal(blocks["level"], level)
    assert np.allclose(blocks["free"][:, 0], twin.deposit.grade)  # grade column round-trips
    prec = read_prec(paths["prec"], n)
    assert np.array_equal(prec.pstart, twin.precedence.pstart)
    assert np.array_equal(prec.plist, twin.precedence.plist)
    values = read_upit(paths["upit"], n)
    assert np.allclose(values, twin.values, rtol=1e-8)
    meta = read_meta(paths["meta"])
    assert meta["synthetic"] is True and meta["schema"] == "oreblocks.minelib-twin/v1"


def test_prec_reader_rejects_malformed(tmp_path):
    p = tmp_path / "bad.prec"
    p.write_text("0 2 1\n", encoding="ascii")  # declares 2 preds, lists 1
    with pytest.raises(ValueError):
        read_prec(p, 2)


# ---------------------------------------------------------------- twin e2e


def test_twin_reread_solves_to_stamped_optimum(tmp_path):
    twin = make_twin("porphyry", (12, 12, 8), seed=9)
    assert twin.upit.n_in_pit > 0, "the default twin should have a profitable pit"
    paths = twin.write(tmp_path)
    n = twin.deposit.grid.n_blocks
    values = read_upit(paths["upit"], n)
    prec = read_prec(paths["prec"], n)
    r = solve_upit(values, prec)
    meta = read_meta(paths["meta"])
    assert r.pit_value == pytest.approx(meta["stamped_optimum"], rel=1e-9)
    assert r.n_in_pit == meta["stamped_n_in_pit"]


def test_twin_deterministic():
    a = make_twin("core_halo", (10, 10, 6), seed=3)
    b = make_twin("core_halo", (10, 10, 6), seed=3)
    assert a.upit.pit_value == b.upit.pit_value
    assert np.array_equal(a.upit.in_pit, b.upit.in_pit)


# ---------------------------------------------------------------- extraction + faces


def test_extraction_progress_and_faces():
    twin = make_twin("porphyry", (14, 14, 8), seed=7)
    dep, in_pit = twin.deposit, twin.upit.in_pit
    total = float(dep.tonnage[in_pit].sum())

    s0 = extraction_state(dep, in_pit, 0.0)
    assert s0.tonnes_extracted == 0.0 and s0.tonnes_remaining == pytest.approx(total)

    s_half = extraction_state(dep, in_pit, 0.5)
    assert s_half.tonnes_extracted == pytest.approx(0.5 * total, rel=0.05)
    assert not (s_half.extracted & ~in_pit).any(), "extraction must stay inside the pit"
    # top-down: every level above the active one is fully extracted (within the pit)
    per = dep.grid.nx * dep.grid.ny
    for lv in range(s_half.active_level + 1, dep.grid.nz):
        sl = slice(lv * per, (lv + 1) * per)
        assert not (in_pit[sl] & ~s_half.extracted[sl]).any()

    faces = loading_faces(dep, s_half, twin.econ, n_faces=3, seed=1)
    assert 1 <= len(faces) <= 3
    assert sum(f.n_blocks for f in faces) == int(s_half.remaining[dep.grid.level_slice(s_half.active_level)].sum())
    for f in faces:
        assert f.tonnes > 0 and 0.0 <= f.mean_grade <= 1.0 and 0.0 <= f.ore_fraction <= 1.0
        assert f.level == s_half.active_level

    # deterministic given the seed
    faces2 = loading_faces(dep, s_half, twin.econ, n_faces=3, seed=1)
    assert [(f.x, f.y, f.tonnes) for f in faces] == [(f.x, f.y, f.tonnes) for f in faces2]

    s_full = extraction_state(dep, in_pit, 1.0)
    assert s_full.tonnes_remaining == pytest.approx(0.0)
