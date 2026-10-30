"""Proof and resource receipts for bounded neighborhood MILP searches."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NeighborhoodSolveReceipt:
    """One restricted MILP solve, separating a feasible incumbent from proof status."""

    round_index: int
    backend: str
    status: str
    seconds: float
    nodes: int
    mip_gap: float | None
    incumbent_npv: float | None
    upper_bound_npv: float | None
    used_start: bool
    improved: bool

    @property
    def certified_optimal(self) -> bool:
        """Whether the backend proved this neighborhood optimum."""
        status = self.status.casefold()
        return status == "optimal" or "highs status 7: optimal" in status


@dataclass(frozen=True)
class LocalSearchReceipt:
    """Configuration and per-neighborhood evidence for a local-search run."""

    neighborhoods: tuple[NeighborhoodSolveReceipt, ...]
    requested_rounds: int
    skipped_rounds: int
    mip_gap_target: float
    time_limit_seconds: float | None
    node_limit: int | None

    @property
    def certified_optimal(self) -> bool:
        """True only when every requested neighborhood was proven optimal."""
        return (
            self.skipped_rounds == 0
            and len(self.neighborhoods) == self.requested_rounds
            and all(item.certified_optimal for item in self.neighborhoods)
        )
