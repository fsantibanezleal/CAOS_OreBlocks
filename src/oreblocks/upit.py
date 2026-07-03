"""Exact ultimate-pit (UPIT) solve — Picard's max-closure -> min-cut reduction on Dinic max-flow.

The maximum-value closed set of the precedence DAG (mine a block only if everything above it is
mined) is, by Picard (1976), the source side of a minimum s-t cut:

- source -> block  with capacity  v_b      for every block with v_b > 0
- block  -> sink   with capacity  -v_b     for every block with v_b < 0
- block  -> predecessor with capacity INF  for every precedence arc

After max-flow the blocks reachable from the source in the residual graph are the optimal pit and
``pit_value = sum(positive values) - maxflow`` (asserted). Pure Python + numpy storage — exact and
deterministic; fine for the 1e3..1e5-block instances oreblocks targets (newman1-size in well under
a second, kd-size in seconds). This mirrors the verified CAOS PitForge engine that reproduces the
published MineLib optima.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .precedence import Precedence

__all__ = ["UpitResult", "solve_upit"]

_EPS = 1e-7


@dataclass(frozen=True)
class UpitResult:
    in_pit: np.ndarray  # bool per block
    pit_value: float
    sum_positive: float
    maxflow: float
    n_in_pit: int


class _Dinic:
    """Adjacency-list Dinic on Python lists — exact float capacities, EPS-guarded residuals."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.to: list[int] = []
        self.cap: list[float] = []
        self.first = [-1] * n
        self.nxt: list[int] = []

    def add_edge(self, u: int, v: int, c: float) -> None:
        self.to.append(v)
        self.cap.append(c)
        self.nxt.append(self.first[u])
        self.first[u] = len(self.to) - 1
        self.to.append(u)
        self.cap.append(0.0)
        self.nxt.append(self.first[v])
        self.first[v] = len(self.to) - 1

    def _bfs(self, s: int, t: int) -> bool:
        self.level = [-1] * self.n
        self.level[s] = 0
        q = [s]
        head = 0
        to, cap, nxt, first, level = self.to, self.cap, self.nxt, self.first, self.level
        while head < len(q):
            u = q[head]
            head += 1
            e = first[u]
            while e != -1:
                v = to[e]
                if cap[e] > _EPS and level[v] < 0:
                    level[v] = level[u] + 1
                    q.append(v)
                e = nxt[e]
        return self.level[t] >= 0

    def _dfs(self, s: int, t: int) -> float:
        # iterative blocking flow (recursion depth would explode on deep pits)
        to, cap, nxt, level, it = self.to, self.cap, self.nxt, self.level, self.it
        total = 0.0
        stack: list[int] = []  # the edges of the current partial path
        u = s
        while True:
            if u == t:
                push = min(cap[e] for e in stack)
                for e in stack:
                    cap[e] -= push
                    cap[e ^ 1] += push
                total += push
                # retreat to just before the first saturated edge on the path
                for k, e in enumerate(stack):
                    if cap[e] <= _EPS:
                        del stack[k:]
                        break
                u = to[stack[-1]] if stack else s
                continue
            e = it[u]
            while e != -1 and not (cap[e] > _EPS and level[to[e]] == level[u] + 1):
                e = nxt[e]
            it[u] = e
            if e == -1:
                level[u] = -1  # dead end: prune this node for the rest of the phase
                if not stack:
                    return total
                stack.pop()
                u = to[stack[-1]] if stack else s
            else:
                stack.append(e)
                u = to[e]

    def maxflow(self, s: int, t: int) -> float:
        flow = 0.0
        while self._bfs(s, t):
            self.it = list(self.first)
            flow += self._dfs(s, t)
        return flow

    def reachable(self, s: int) -> np.ndarray:
        seen = np.zeros(self.n, dtype=bool)
        seen[s] = True
        q = [s]
        head = 0
        to, cap, nxt, first = self.to, self.cap, self.nxt, self.first
        while head < len(q):
            u = q[head]
            head += 1
            e = first[u]
            while e != -1:
                v = to[e]
                if cap[e] > _EPS and not seen[v]:
                    seen[v] = True
                    q.append(v)
                e = nxt[e]
        return seen


def solve_upit(values: np.ndarray, prec: Precedence) -> UpitResult:
    """Exact UPIT over explicit precedence. Self-checks closure feasibility + the value identity."""
    v = np.asarray(values, dtype=np.float64)
    n = v.shape[0]
    if prec.pstart.shape[0] != n + 1:
        raise ValueError(f"precedence is over {prec.pstart.shape[0] - 1} blocks, values over {n}")
    s, t = n, n + 1
    sum_positive = float(v[v > 0].sum())
    inf = sum_positive + 1.0

    g = _Dinic(n + 2)
    for b in range(n):
        vb = v[b]
        if vb > 0:
            g.add_edge(s, b, float(vb))
        elif vb < 0:
            g.add_edge(b, t, float(-vb))
    pstart, plist = prec.pstart, prec.plist
    for b in range(n):
        for k in range(pstart[b], pstart[b + 1]):
            g.add_edge(b, int(plist[k]), inf)

    maxflow = g.maxflow(s, t)
    seen = g.reachable(s)
    in_pit = seen[:n]

    pit_value = float(v[in_pit].sum())
    # closure feasibility: every predecessor of an in-pit block is in the pit
    for b in np.nonzero(in_pit)[0]:
        for k in range(pstart[b], pstart[b + 1]):
            if not in_pit[plist[k]]:
                raise AssertionError(f"closure violated: {b} in pit, predecessor {int(plist[k])} out")
    gap = abs(pit_value - (sum_positive - maxflow))
    if gap > max(1e-6 * max(1.0, sum_positive), 1e-3):
        raise AssertionError(f"value identity violated: gap {gap}")
    return UpitResult(
        in_pit=in_pit,
        pit_value=pit_value,
        sum_positive=sum_positive,
        maxflow=maxflow,
        n_in_pit=int(in_pit.sum()),
    )
