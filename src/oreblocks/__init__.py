"""oreblocks — synthetic 3-D ore-body block models of the MineLib nature.

Seeded deposit archetypes with per-block grades, bench (level) structure, slope precedence, UPIT
economics with per-block optimal destination, an exact max-closure solver, extraction states with
loading faces, and MineLib ``.blocks/.prec/.upit`` read/write. Everything deterministic given a
seed; every generated instance is clearly labelled SYNTHETIC.
"""

__version__ = "0.1.0"

from .economics import Econ, block_values, cutoff_grade, is_ore
from .extraction import ExtractionState, Face, extraction_state, loading_faces
from .fields import ARCHETYPES, Deposit, make_deposit
from .grid import BlockGrid
from .minelib_io import read_blocks, read_meta, read_prec, read_upit, write_minelib
from .precedence import Precedence, build_precedence, slope_offsets
from .twins import Twin, make_twin
from .upit import UpitResult, solve_upit

__all__ = [
    "__version__",
    "ARCHETYPES",
    "BlockGrid",
    "Deposit",
    "Econ",
    "ExtractionState",
    "Face",
    "Precedence",
    "Twin",
    "UpitResult",
    "block_values",
    "build_precedence",
    "cutoff_grade",
    "extraction_state",
    "is_ore",
    "loading_faces",
    "make_deposit",
    "make_twin",
    "read_blocks",
    "read_meta",
    "read_prec",
    "read_upit",
    "slope_offsets",
    "solve_upit",
    "write_minelib",
]
