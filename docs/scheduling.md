# Production scheduling: CPIT, the certified bound, and the honest gap

The ultimate pit answers *which* blocks are worth mining. It says nothing about *when*. This page
covers the scheduling half of the package, added in 0.2.0.

## 1. The problem

The **constrained pit limit problem** (CPIT) assigns every block an extraction period so that slope
precedence holds in every period, per-period resource capacities hold, and discounted value is
maximised. In the cumulative variables that every published algorithm and every published bound is
stated in, with `x_bt = 1` meaning block `b` has been extracted by the end of period `t`:

```
max   sum_b sum_t  p_bt (x_bt - x_b,t-1)
s.t.  sum_b a_rb (x_bt - x_b,t-1) <= c_rt      for all resources r, periods t
      x_bt <= x_at                             for all precedence arcs (a, b), all t
      x_bt <= x_b,t+1                          monotone: once mined, stays mined
      x_bt in {0,1},   x_b0 = 0
```

Source: Chicoisne, R., Espinoza, D., Goycoolea, M., Moreno, E. and Rubio, E., *A New Algorithm for
the Open-Pit Mine Production Scheduling Problem*, Operations Research 60(3):517-528, 2012,
[doi:10.1287/opre.1120.1050](https://doi.org/10.1287/opre.1120.1050), equations (3a)-(3f).

Three things to keep straight:

- **Precedence is imposed in every period**, not once. That is what makes it a scheduling constraint
  rather than a set constraint.
- **Monotonicity is a real constraint.** Dropping it silently lets a block be un-mined.
- **The destination is fixed before the model runs**, folded into `p_b`. A model that *chooses* the
  destination is PCPSP, a different problem. See section 6.

CPIT is NP-hard: adding a knapsack constraint to a maximum-closure problem gives the
precedence-constrained knapsack problem. Caccetta, L. and Hill, S., *An Application of Branch and Cut
to Open Pit Mine Scheduling*, Journal of Global Optimization 27(2-3):349-365, 2003,
[doi:10.1023/A:1024835022186](https://doi.org/10.1023/A:1024835022186).

## 2. The certified bound, without an LP solver

`cpit_lp_relaxation` implements the **critical multiplier algorithm**, Chicoisne et al. 2012,
Theorem 3.1: for a single resource constraint per period, the LP relaxation of CPIT is solved
exactly in `O(mn log n)`.

The mechanism, in three steps:

1. **Abel summation.** With `d_t` the discount factor of period `t`, the objective rewrites as
   `sum_t gamma_t (x_t . p)` where `gamma_t = d_t - d_{t+1} > 0` and `gamma_T = d_T`. Every weight is
   positive, so maximising the whole is maximising each `x_t . p` independently.
2. **Cumulative capacity.** Relaxing the per-period capacities into their cumulative form
   `U_t = sum_{s<=t} c_s` decouples the periods into `T` problems
   `CP(U_t) = max p.x` over closures with `a.x <= U_t`.
3. **Parametric nested pits.** `CP(U)` is attained by a convex combination of two consecutive
   solutions of `UPL(p - lambda a)`, which are nested pits, hence maximum closures, hence min-cuts.
   With `b^u >= U >= b^l` the two bracketing capacities, `alpha = (b^u - U) / (b^u - b^l)` and
   `x = alpha x^l + (1 - alpha) x^u`.

The solution turns out to be feasible for CPIT itself (it saturates each period exactly), so the
relaxation is tight and the value is the LP optimum. In code the multiplier is found by bisection,
and each solve is restricted to the previous, larger pit, so the work collapses to a handful of
maximum closures rather than one per bisection.

```python
import oreblocks as ob

inst = ob.read_cpit("newman1.cpit")
prec = ob.read_prec("newman1.prec", inst.n_blocks)
rel = ob.cpit_lp_relaxation(inst, prec)
print(rel.bound, rel.closure_solves)
```

**Several resources.** `cpit_bound_two_resources` implements Algorithm 4: relax all but one resource,
solve, and keep the smallest bound over the choices. Each single-resource relaxation is a relaxation
of the full problem, so each bound is valid; the minimum is the tightest this construction gives. It
is looser than a joint LP bound.

**The joint bound needs Bienstock-Zuckerberg, and it is no tighter than the LP.** Munoz, G.,
Espinoza, D., Goycoolea, M., Moreno, E., Queyranne, M. and Rivera Letelier, O., *A study of the
Bienstock-Zuckerberg algorithm*, Computational Optimization and Applications, 2017,
[doi:10.1007/s10589-017-9946-1](https://doi.org/10.1007/s10589-017-9946-1)
([arXiv:1607.01104](https://arxiv.org/abs/1607.01104)), prove `Z_BZ = Z_LP` because the precedence
system is totally unimodular. BZ is a speed result on instances with millions of variables, not a
better bound. It is not implemented here.

## 3. Turning the bound into a plan

`toposort_schedule` implements the TopoSort heuristic (Chicoisne et al. 2012, section 3.2, Algorithms
2 and 3): take a topological ordering of the precedence DAG that puts high-weight blocks early, then
walk it, giving each block the earliest period that is at least its predecessors' periods and whose
remaining resources fit it. Feasibility is by construction.

| `weight=` | formula | origin |
|---|---|---|
| `"greedy"` | `w_b = p_b` | the obvious baseline (GrTS) |
| `"gershon"` | `w_b = p_b + sum of p over the whole successor cone` | Gershon 1987a (GeTS) |
| `"expected"` | `w_b = -E_b` from the LP relaxation | Chicoisne et al. 2012 (ExTS) |

with the expected extraction time

```
E_b = sum_{t=1..T} t (x*_bt - x*_b,t-1) + (T + 1)(1 - x*_bT)
```

read off the fractional LP solution. This is the bridge: the bound is not only the yardstick, it is
the seed of the plan. The published spread is large. On their AsiaMine instance with two resource
constraints, greedy reached 0.138 of the LP bound and expected-time reached 0.972, using the same
scheduling code.

`improve_schedule` then applies a **shift** local search: pull positive-value blocks forward when
precedence and capacity allow, push negative-value blocks back when their successors allow. Both
moves are strictly improving under discounting and preserve feasibility. This is the shift
neighbourhood family used across the mine-scheduling metaheuristic literature (Lamghari, A. and
Dimitrakopoulos, R., European Journal of Operational Research 222(3), 2012,
[doi:10.1016/j.ejor.2012.05.029](https://doi.org/10.1016/j.ejor.2012.05.029)). It is deliberately
**not** the exact `C-PIT[D]` neighbourhood of Chicoisne et al. section 3.3, which re-solves a
restricted integer program per neighbourhood and needs a MILP solver. That one is not implemented.

`solve_cpit` runs the whole ladder: bound, schedule from the tighter relaxation, improve, attach the
bound, and assert that the feasible objective never exceeds it.

## 4. The gap

```
gap = (bound - npv) / bound
```

which is the definition MineLib results are published under (Jelvez, E., Morales, N. and
Nancel-Penard, P., *Open-Pit Mine Production Scheduling: Improvements to MineLib Library Problems*,
MPES 2018, [doi:10.1007/978-3-319-99220-4_18](https://doi.org/10.1007/978-3-319-99220-4_18),
equation 12). `ScheduleResult.gap_pct` reports it. A schedule reported without its gap is a number
with no scale.

## 5. The controls

`run_controls` runs three checks that tie the scheduling lane to the proven ultimate pit. They are
not optional and they catch real bugs that no chart would show.

| control | statement |
|---|---|
| **duality** | at rate 0 with unlimited capacity, the LP's mined set equals the exact ultimate pit block for block and the bound equals the exact pit value |
| **bound** | the certified bound is at least any feasible objective produced |
| **order invariance** | still at rate 0 with unlimited capacity, every weighting returns the same objective |

## 6. The file formats

`.blocks`, `.prec` and `.upit` describe a deposit and carry no time. The scheduling question lives in
`.cpit` and `.pcpsp`, both read and written here. The format below was measured on the published
`newman1` files, not inferred:

```
NAME: Newman1
TYPE: CPIT
NBLOCKS: 1060
NPERIODS: 6
NRESOURCE_SIDE_CONSTRAINTS: 2
DISCOUNT_RATE: 0.08
RESOURCE_CONSTRAINT_LIMITS:
0 0 L 2000000            <- "<resource> <period> <sense> <limit>"
OBJECTIVE_FUNCTION:
0 -2236.7886             <- "<block> <value>"
RESOURCE_CONSTRAINT_COEFFICIENTS:
0 0 2192.93              <- "<block> <resource> <coefficient>", SPARSE
EOF
```

Three traps, all real:

1. **The coefficient section is sparse.** On `newman1`, resource 0 has a row for all 1060 blocks and
   resource 1 for only 572: waste consumes no plant capacity.
2. **A large negative objective value is a sentinel**, not a cost. The last `newman1` block carries
   `-5.36024E+19` for destination 1, meaning that destination is forbidden. `FORBIDDEN_VALUE` and the
   `forbidden` mask handle it; summing it produces nonsense.
3. **The first period is undiscounted**, that is `1/(1+r)^(t-1)`. This was settled by measurement:
   under this convention the critical multiplier algorithm gives 24 487 410 on the published
   `newman1.cpit` against the published LP bound of 24 486 549, a relative difference of 3.5e-5, and
   the residual is in the right direction because the single-resource relaxation is looser than the
   joint bound. The other convention gives 22 673 528, off by 8 percent.

`Pcpsp.to_cpit()` performs the reduction the industry performs when it computes a cutoff grade before
scheduling: fix each block's destination to its best one. Use it to compare a fixed-destination plan
against a chosen-destination one on the same instance, and to see what the fixing costs.

## 7. Spatial coherence

A block-level schedule is free to pick blocks anywhere in the model, and it does. From the Final
Remarks of Chicoisne et al. 2012:

> "It is likely that the C-PIT solutions are such that blocks scheduled in a same time period are
> scattered throughout the mine. This might lead to schedules that require manual intervention by
> mining engineers to consider additional operational constraints [...] exacerbated by the fact that
> our minimal planning units are blocks rather than bench-phases."

A schedule with a high NPV and forty disconnected fragments per year is not a mine plan, and no NPV
chart shows the difference. `schedule_coherence` reports, per period: the number of 6-connected
components, the share of the period held by the largest one, and the narrowest run of consecutive
mined blocks along a bench (a crude proxy for minimum mining width, which Bai et al. 2018,
[doi:10.17159/2411-9717/2018/v118n5a8](https://doi.org/10.17159/2411-9717/2018/v118n5a8), target at
about 100 m for equipment reasons).

## 8. What is deliberately not here

- **Stockpiles.** An inventory whose reclaimed grade is the blend of what is inside makes the model
  bilinear. Published linear models fix the stockpile grade as a parameter and search over it
  (Rezakhah, M., Moreno, E. and Newman, A., Computers and Operations Research, 2020,
  [doi:10.1016/j.cor.2019.02.001](https://doi.org/10.1016/j.cor.2019.02.001)), and the value is
  fragile: at 5 and 10 percent annual degradation it falls by 37 and 69 percent respectively
  (Rezakhah, M. and Newman, A., [doi:10.1016/j.cor.2018.11.009](https://doi.org/10.1016/j.cor.2018.11.009)).
- **Blending and other general side constraints.** The `.pcpsp` reader carries
  `NGENERAL_SIDE_CONSTRAINTS`, and the solver ignores them; an instance that declares them is not
  being solved as posed.
- **Minimum-production constraints** (`sense = 'G'`). The reader accepts them; the critical multiplier
  algorithm raises `NotImplementedError` rather than quietly solving a different problem.
- **Stochastic scheduling.** Multi-realisation SIP is a different model. Cite it, do not claim it.
