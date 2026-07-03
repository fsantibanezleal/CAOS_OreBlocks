"""Seeded, named random streams — same discipline as minehaulsim.

Every stochastic component draws from its own named stream derived from (seed, name), so adding a
new consumer never perturbs existing draws and results are byte-identical given the seed.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["stream"]


def stream(seed: int, name: str) -> np.random.Generator:
    """A generator for the (seed, name) pair — stable across platforms and numpy versions."""
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    words = [int.from_bytes(digest[k : k + 4], "little") for k in range(0, 16, 4)]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(words)))
