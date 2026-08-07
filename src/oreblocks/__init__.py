"""oreblocks — synthetic 3-D ore-body block models of the MineLib nature, and their scheduling.

Seeded deposit archetypes with per-block grades, bench (level) structure, slope precedence, UPIT
economics with per-block optimal destination, an exact max-closure solver, extraction states with
loading faces, and MineLib ``.blocks/.prec/.upit`` read/write. Everything deterministic given a
seed; every generated instance is clearly labelled SYNTHETIC.

Since 0.2.0 the package also covers the *scheduling* half of MineLib: the ``.cpit`` and ``.pcpsp``
model files (periods, discount rate, per-period capacities, destinations), the certified CPIT LP
bound by the critical multiplier algorithm, the TopoSort family of rounding heuristics, and the
spatial-coherence measurement that says whether a period's mined increment is one workable volume or
a scatter of fragments.
"""

__version__ = "0.2.0"

from .coherence import PeriodCoherence, period_coherence, schedule_coherence
from .economics import Econ, block_values, cutoff_grade, is_ore
from .extraction import ExtractionState, Face, extraction_state, loading_faces
from .fields import ARCHETYPES, Deposit, make_deposit
from .grid import BlockGrid
from .minelib_io import read_blocks, read_meta, read_prec, read_upit, write_minelib
from .minelib_models import (
    FORBIDDEN_VALUE,
    Cpit,
    Pcpsp,
    read_cpit,
    read_pcpsp,
    write_cpit,
    write_pcpsp,
)
from .precedence import Precedence, build_precedence, slope_offsets
from .schedule import (
    TOPOSORT_WEIGHTS,
    Controls,
    LpRelaxation,
    ScheduleResult,
    cpit_bound_two_resources,
    cpit_lp_relaxation,
    expected_extraction_times,
    improve_schedule,
    run_controls,
    schedule_value,
    solve_cpit,
    toposort_order,
    toposort_schedule,
)
from .twins import Twin, make_twin
from .upit import UpitResult, max_closure_within, solve_upit

__all__ = [
    "__version__",
    "ARCHETYPES",
    "FORBIDDEN_VALUE",
    "TOPOSORT_WEIGHTS",
    "BlockGrid",
    "Controls",
    "Cpit",
    "Deposit",
    "Econ",
    "ExtractionState",
    "Face",
    "LpRelaxation",
    "Pcpsp",
    "PeriodCoherence",
    "Precedence",
    "ScheduleResult",
    "Twin",
    "UpitResult",
    "block_values",
    "build_precedence",
    "cpit_bound_two_resources",
    "cpit_lp_relaxation",
    "cutoff_grade",
    "expected_extraction_times",
    "extraction_state",
    "improve_schedule",
    "is_ore",
    "loading_faces",
    "make_deposit",
    "make_twin",
    "max_closure_within",
    "period_coherence",
    "read_blocks",
    "read_cpit",
    "read_meta",
    "read_pcpsp",
    "read_prec",
    "read_upit",
    "run_controls",
    "schedule_coherence",
    "schedule_value",
    "slope_offsets",
    "solve_cpit",
    "solve_upit",
    "toposort_order",
    "toposort_schedule",
    "write_cpit",
    "write_minelib",
    "write_pcpsp",
]
