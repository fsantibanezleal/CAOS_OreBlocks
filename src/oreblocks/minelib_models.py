"""MineLib scheduling model files: ``.cpit`` (CPIT) and ``.pcpsp`` (PCPSP) read/write.

``.blocks``/``.prec``/``.upit`` describe a *deposit*. They carry no time, no capacity and no
discount rate, so they can only pose the ultimate-pit question. The scheduling question lives in two
further published files, and this module is the reader and writer for both.

Format, measured on the published ``newman1`` triple (1060 blocks, 3922 precedence arcs) rather than
inferred from prose::

    NAME: Newman1
    TYPE: CPIT
    NBLOCKS: 1060
    NPERIODS: 6
    NRESOURCE_SIDE_CONSTRAINTS: 2
    DISCOUNT_RATE: 0.08
    RESOURCE_CONSTRAINT_LIMITS:
    0 0 L 2000000            <- "<resource> <period> <sense> <limit>", R*T rows
    ...
    OBJECTIVE_FUNCTION:
    0 -2236.7886             <- "<block> <value>", NBLOCKS rows
    ...
    RESOURCE_CONSTRAINT_COEFFICIENTS:
    0 0 2192.93              <- "<block> <resource> <coefficient>", sparse: only non-zero rows
    ...
    EOF

``.pcpsp`` is the same with ``NDESTINATIONS``/``NGENERAL_SIDE_CONSTRAINTS`` added, one objective
value per destination on each block row, and a destination index inserted into the coefficient rows
(``<block> <destination> <resource> <coefficient>``).

Three things a naive parser gets wrong, all measured on the real file:

1. **The coefficient section is sparse.** On ``newman1`` resource 0 (extraction tonnage) has a row
   for all 1060 blocks but resource 1 (processing tonnage) has rows for only 572 of them: waste
   consumes no plant capacity. A parser that requires R*n rows fails on the published instance.
2. **Large negative values are destination sentinels, not numbers.** The last ``newman1`` block
   carries ``-5.36024E+19`` for destination 1, which means "this destination is forbidden", not
   "this destination costs 5e19". Summing it produces nonsense. Values at or below
   ``FORBIDDEN_VALUE`` are reported through ``forbidden`` masks and replaced by ``-inf``.
3. **The first period is UNDISCOUNTED.** MineLib weights period ``t`` by ``1/(1+r)^(t-1)`` with
   ``t`` counting from 1, so period 1 carries a factor of exactly 1. This was settled by
   measurement, not by reading: running the critical multiplier algorithm on the published
   ``newman1.cpit`` gives a certified bound of **24 487 410** under this convention against the
   published LP bound of **24 486 549** (a relative difference of 3.5e-5, and the residual is in the
   right direction because the single-resource relaxation is looser than the joint bound). The other
   convention gives 22 673 528, off by 8 percent. ``period_one_undiscounted`` defaults to ``True``
   for that reason; set it ``False`` for an instance that genuinely discounts its first period.

Nothing here downloads or redistributes MineLib data; it reads files the caller already has, and
writes the same formats for synthetic instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "FORBIDDEN_VALUE",
    "Cpit",
    "Pcpsp",
    "read_cpit",
    "read_pcpsp",
    "write_cpit",
    "write_pcpsp",
]

#: Objective entries at or below this are "destination forbidden" sentinels, not costs.
FORBIDDEN_VALUE = -1e18


@dataclass(frozen=True)
class Cpit:
    """A CPIT instance: values, per-period capacities, discount rate. Destinations are fixed."""

    name: str
    n_blocks: int
    n_periods: int
    discount_rate: float
    value: np.ndarray  # float64 (n,) net value at the pre-decided destination
    limit: np.ndarray  # float64 (R, T) resource availability per period
    sense: np.ndarray  # "<U1" (R, T) "L" (<=) or "G" (>=)
    coef: np.ndarray  # float64 (R, n) resource consumed by extracting a block
    resource_names: tuple[str, ...] = ()
    period_one_undiscounted: bool = True

    @property
    def n_resources(self) -> int:
        return int(self.limit.shape[0])

    def discount_factors(self) -> np.ndarray:
        """``d[t]`` for t = 0..T-1, applied to value realised in period ``t+1``."""
        r = float(self.discount_rate)
        expo = np.arange(self.n_periods, dtype=np.float64)
        if not self.period_one_undiscounted:
            expo = expo + 1.0
        return 1.0 / (1.0 + r) ** expo

    def gamma(self) -> np.ndarray:
        """Weights turning the by-period objective into a sum over CUMULATIVE pit values.

        ``sum_t d[t] (x_t - x_{t-1}) . v  ==  sum_t gamma[t] (x_t . v)`` by Abel summation, with
        ``gamma[t] = d[t] - d[t+1] > 0`` and ``gamma[T-1] = d[T-1]``. Every weight is positive, which
        is exactly why the cumulative-capacity relaxation decomposes into independent per-period
        maximum-closure problems (Chicoisne et al. 2012, proof of Theorem 3.1).
        """
        d = self.discount_factors()
        g = np.empty_like(d)
        g[:-1] = d[:-1] - d[1:]
        g[-1] = d[-1]
        return g

    def cumulative_limits(self, resource: int = 0) -> np.ndarray:
        """``U_t = sum_{s<=t} c_s`` for one resource, the relaxation's cumulative capacity."""
        return np.cumsum(self.limit[resource])


@dataclass(frozen=True)
class Pcpsp:
    """A PCPSP instance: the model chooses a destination per block, so value is (n, D)."""

    name: str
    n_blocks: int
    n_periods: int
    n_destinations: int
    discount_rate: float
    value: np.ndarray  # float64 (n, D), -inf where the destination is forbidden
    forbidden: np.ndarray  # bool (n, D)
    limit: np.ndarray  # float64 (R, T)
    sense: np.ndarray  # "<U1" (R, T)
    coef: np.ndarray  # float64 (R, n, D)
    n_general_side_constraints: int = 0
    period_one_undiscounted: bool = True

    @property
    def n_resources(self) -> int:
        return int(self.limit.shape[0])

    def best_destination(self) -> tuple[np.ndarray, np.ndarray]:
        """The a-priori optimal destination per block, and its value: the CPIT reduction."""
        d = np.argmax(np.where(self.forbidden, -np.inf, self.value), axis=1)
        v = self.value[np.arange(self.n_blocks), d]
        return d.astype(np.int64), v.astype(np.float64)

    def to_cpit(self, name: str | None = None) -> Cpit:
        """Collapse to CPIT by fixing each block's destination to its best one.

        This is the reduction the whole industry performs when it computes a cutoff grade before
        scheduling, and it is exactly what makes CPIT a *different problem*: the destination stops
        being a decision. Use it to compare a fixed-destination plan against a chosen-destination
        one on the same instance.
        """
        dest, val = self.best_destination()
        coef = np.stack(
            [self.coef[r][np.arange(self.n_blocks), dest] for r in range(self.n_resources)]
        )
        return Cpit(
            name=name or f"{self.name}-fixeddest",
            n_blocks=self.n_blocks,
            n_periods=self.n_periods,
            discount_rate=self.discount_rate,
            value=val,
            limit=self.limit.copy(),
            sense=self.sense.copy(),
            coef=coef,
            period_one_undiscounted=self.period_one_undiscounted,
        )


# ------------------------------------------------------------------------------------------------
# parsing
# ------------------------------------------------------------------------------------------------
def _sections(text: str) -> tuple[dict[str, str], dict[str, list[list[str]]]]:
    """Split a MineLib model file into scalar headers and the three keyword-terminated blocks."""
    headers: dict[str, str] = {}
    blocks: dict[str, list[list[str]]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line == "EOF":
            continue
        if line.endswith(":") and not line[0].isdigit():
            current = line[:-1].strip().upper()
            blocks[current] = []
            continue
        if ":" in line and not line[0].isdigit():
            key, _, val = line.partition(":")
            headers[key.strip().upper()] = val.strip()
            current = None
            continue
        if current is None:
            raise ValueError(f"data row outside any section: {line!r}")
        blocks[current].append(line.split())
    return headers, blocks


def _limits(rows: list[list[str]], n_resources: int, n_periods: int) -> tuple[np.ndarray, np.ndarray]:
    limit = np.full((n_resources, n_periods), np.nan)
    sense = np.full((n_resources, n_periods), "", dtype="<U1")
    for row in rows:
        if len(row) != 4:
            raise ValueError(f"RESOURCE_CONSTRAINT_LIMITS row must be '<r> <t> <sense> <limit>': {row}")
        r, t, s, lim = int(row[0]), int(row[1]), row[2].upper(), float(row[3])
        if not (0 <= r < n_resources and 0 <= t < n_periods):
            raise ValueError(f"limit row out of range: resource {r}, period {t}")
        if s not in ("L", "G"):
            raise ValueError(f"unknown constraint sense {s!r} (expected L or G)")
        limit[r, t] = lim
        sense[r, t] = s
    if np.isnan(limit).any():
        missing = int(np.isnan(limit).sum())
        raise ValueError(f"RESOURCE_CONSTRAINT_LIMITS: {missing} (resource, period) cells missing")
    return limit, sense


def read_cpit(path: str | Path) -> Cpit:
    """Read a published ``.cpit`` file. Sparse coefficient rows and sentinels handled."""
    headers, blocks = _sections(Path(path).read_text(encoding="ascii", errors="strict"))
    if headers.get("TYPE", "").upper() not in ("CPIT", ""):
        raise ValueError(f"not a CPIT file: TYPE = {headers.get('TYPE')!r}")
    n = int(headers["NBLOCKS"])
    t_max = int(headers["NPERIODS"])
    n_res = int(headers.get("NRESOURCE_SIDE_CONSTRAINTS", 1))
    rate = float(headers["DISCOUNT_RATE"])

    limit, sense = _limits(blocks.get("RESOURCE_CONSTRAINT_LIMITS", []), n_res, t_max)

    value = np.full(n, np.nan)
    for row in blocks.get("OBJECTIVE_FUNCTION", []):
        b = int(row[0])
        if not (0 <= b < n):
            raise ValueError(f"OBJECTIVE_FUNCTION: block {b} out of range (n={n})")
        value[b] = float(row[1])
    if np.isnan(value).any():
        raise ValueError(f"OBJECTIVE_FUNCTION: {int(np.isnan(value).sum())} blocks have no value")
    value[value <= FORBIDDEN_VALUE] = -np.inf

    coef = np.zeros((n_res, n))
    for row in blocks.get("RESOURCE_CONSTRAINT_COEFFICIENTS", []):
        b, r, q = int(row[0]), int(row[1]), float(row[2])
        if not (0 <= b < n and 0 <= r < n_res):
            raise ValueError(f"coefficient row out of range: block {b}, resource {r}")
        coef[r, b] = q
    if (coef < 0).any():
        raise ValueError("resource coefficients must be non-negative")

    return Cpit(
        name=headers.get("NAME", Path(path).stem),
        n_blocks=n,
        n_periods=t_max,
        discount_rate=rate,
        value=value,
        limit=limit,
        sense=sense,
        coef=coef,
    )


def read_pcpsp(path: str | Path) -> Pcpsp:
    """Read a published ``.pcpsp`` file (multiple destinations per block)."""
    headers, blocks = _sections(Path(path).read_text(encoding="ascii", errors="strict"))
    if headers.get("TYPE", "").upper() not in ("PCPSP", ""):
        raise ValueError(f"not a PCPSP file: TYPE = {headers.get('TYPE')!r}")
    n = int(headers["NBLOCKS"])
    t_max = int(headers["NPERIODS"])
    n_dest = int(headers["NDESTINATIONS"])
    n_res = int(headers.get("NRESOURCE_SIDE_CONSTRAINTS", 1))
    n_gen = int(headers.get("NGENERAL_SIDE_CONSTRAINTS", 0))
    rate = float(headers["DISCOUNT_RATE"])

    limit, sense = _limits(blocks.get("RESOURCE_CONSTRAINT_LIMITS", []), n_res, t_max)

    value = np.full((n, n_dest), np.nan)
    for row in blocks.get("OBJECTIVE_FUNCTION", []):
        b = int(row[0])
        if len(row) != 1 + n_dest:
            raise ValueError(f"OBJECTIVE_FUNCTION: block {b} has {len(row) - 1} values, expected {n_dest}")
        for d in range(n_dest):
            value[b, d] = float(row[1 + d])
    if np.isnan(value).any():
        raise ValueError("OBJECTIVE_FUNCTION: some (block, destination) cells have no value")
    forbidden = value <= FORBIDDEN_VALUE
    value = np.where(forbidden, -np.inf, value)
    if forbidden.all(axis=1).any():
        raise ValueError("some blocks have every destination forbidden")

    coef = np.zeros((n_res, n, n_dest))
    for row in blocks.get("RESOURCE_CONSTRAINT_COEFFICIENTS", []):
        if len(row) != 4:
            raise ValueError(f"PCPSP coefficient row must be '<b> <d> <r> <q>': {row}")
        b, d, r, q = int(row[0]), int(row[1]), int(row[2]), float(row[3])
        coef[r, b, d] = q

    return Pcpsp(
        name=headers.get("NAME", Path(path).stem),
        n_blocks=n,
        n_periods=t_max,
        n_destinations=n_dest,
        discount_rate=rate,
        value=value,
        forbidden=forbidden,
        limit=limit,
        sense=sense,
        coef=coef,
        n_general_side_constraints=n_gen,
    )


# ------------------------------------------------------------------------------------------------
# writing
# ------------------------------------------------------------------------------------------------
def _fmt(x: float) -> str:
    return f"{x:.9g}"


def write_cpit(path: str | Path, inst: Cpit) -> Path:
    """Write a ``.cpit`` file in the published format (sparse coefficients, ``EOF`` terminated)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = [
        f"NAME: {inst.name}",
        "TYPE: CPIT",
        f"NBLOCKS: {inst.n_blocks}",
        f"NPERIODS: {inst.n_periods}",
        f"NRESOURCE_SIDE_CONSTRAINTS: {inst.n_resources}",
        f"DISCOUNT_RATE: {_fmt(inst.discount_rate)}",
        "RESOURCE_CONSTRAINT_LIMITS:",
    ]
    for r in range(inst.n_resources):
        for t in range(inst.n_periods):
            out.append(f"{r} {t} {inst.sense[r, t] or 'L'} {_fmt(float(inst.limit[r, t]))}")
    out.append("OBJECTIVE_FUNCTION:")
    for b in range(inst.n_blocks):
        v = float(inst.value[b])
        out.append(f"{b} {_fmt(FORBIDDEN_VALUE if np.isneginf(v) else v)}")
    out.append("RESOURCE_CONSTRAINT_COEFFICIENTS:")
    for r in range(inst.n_resources):
        nz = np.nonzero(inst.coef[r])[0]
        out.extend(f"{int(b)} {r} {_fmt(float(inst.coef[r, b]))}" for b in nz)
    out.append("EOF")
    p.write_text("\n".join(out) + "\n", encoding="ascii", newline="\n")
    return p


def write_pcpsp(path: str | Path, inst: Pcpsp) -> Path:
    """Write a ``.pcpsp`` file in the published format."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = [
        f"NAME: {inst.name}",
        "TYPE: PCPSP",
        f"NBLOCKS: {inst.n_blocks}",
        f"NPERIODS: {inst.n_periods}",
        f"NDESTINATIONS: {inst.n_destinations}",
        f"NRESOURCE_SIDE_CONSTRAINTS: {inst.n_resources}",
        f"NGENERAL_SIDE_CONSTRAINTS: {inst.n_general_side_constraints}",
        f"DISCOUNT_RATE: {_fmt(inst.discount_rate)}",
        "RESOURCE_CONSTRAINT_LIMITS:",
    ]
    for r in range(inst.n_resources):
        for t in range(inst.n_periods):
            out.append(f"{r} {t} {inst.sense[r, t] or 'L'} {_fmt(float(inst.limit[r, t]))}")
    out.append("OBJECTIVE_FUNCTION:")
    for b in range(inst.n_blocks):
        cells = []
        for d in range(inst.n_destinations):
            v = float(inst.value[b, d])
            cells.append(_fmt(FORBIDDEN_VALUE if np.isneginf(v) else v))
        out.append(f"{b} " + " ".join(cells))
    out.append("RESOURCE_CONSTRAINT_COEFFICIENTS:")
    for r in range(inst.n_resources):
        bb, dd = np.nonzero(inst.coef[r])
        out.extend(
            f"{int(b)} {int(d)} {r} {_fmt(float(inst.coef[r, b, d]))}" for b, d in zip(bb, dd, strict=True)
        )
    out.append("EOF")
    p.write_text("\n".join(out) + "\n", encoding="ascii", newline="\n")
    return p
