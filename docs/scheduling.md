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


## 9. The rest of the ladder (0.3.0)

### 9.1 The joint bound: Bienstock-Zuckerberg

`cpit_bound_two_resources` relaxes all but one resource and keeps the smallest of the resulting
bounds. Every one of them is valid, so the minimum is certified, and on an instance where two
capacities both bind it is **loose**. That matters because a reported gap then mixes two different
things: how much the heuristic loses, and how much the bound loses.

`cpit_bz_bound` computes the JOINT bound over all resources at once. It is column generation whose
restricted master runs over the LINEAR hull of the precedence polytope, with generator matrices whose
columns are orthogonal 0-1 vectors, so restricting to their span EQUATES the variables inside each
support and contracts the problem. The pricing problem is `max (c - pi'H)'v` over closures, which is a
maximum closure, which is a minimum cut.

| instance | Algorithm 4 | BZ joint | published |
|---|---|---|---|
| `newman1.cpit` | 24,487,410 | **24,486,184** | 24,486,549 (PCPSP LP) |

The ordering is the check: the CPIT LP bound must sit below the PCPSP LP bound, because PCPSP is the
richer problem, and it does. On a **single**-resource instance BZ and the critical multiplier
algorithm agree to machine precision, which is two entirely different algorithms computing the same
LP and the strongest correctness check in the suite.

`Z_BZ = Z_LP` still holds (Munoz et al.,
[doi:10.1007/s10589-017-9946-1](https://doi.org/10.1007/s10589-017-9946-1)). BZ is a speed result and
a joint-bound result, never a tighter-than-LP one.

### 9.2 The exact C-PIT[D] local search

`refine.exact_local_search` implements Chicoisne et al. section 3.3: fix every block outside a small
set `D` at its incumbent period and re-solve the restricted problem EXACTLY as a MILP. All three of
their neighbourhood constructions, chosen with equal probability: a connected subset of a random
block's predecessors, the same with successors, and the blocks scheduled in `t-1, t, t+1`.

This is the rung `improve_schedule` is explicitly not. The shift neighbourhood cannot move a block and
its cone together; this can, and the difference is the last few percent of the gap.

### 9.3 The sliding time window

`sliding_window_schedule` (Cullenbine, Wood and Newman,
[doi:10.1007/s11590-011-0306-2](https://doi.org/10.1007/s11590-011-0306-2)): enforce every constraint
inside a window, fix the first period or two, slide. It is what industry runs, and it is what Rio
Tinto's platform uses to seed its large neighbourhood search.

**The bug this shipped with, worth recording**: the first version re-planned blocks that had already
consumed capacity in an earlier window, so the plan quietly double-booked the fleet and the objective
collapsed to a third of its value while every feasibility check still passed. Only the frozen prefix
is a decision; everything after it must be released before the next slide.

### 9.4 Destinations: when the cutoff grade becomes an output

`destinations.destination_toposort` chooses each block's destination against the capacity remaining in
the period it is scheduled into. When the plant is full the same block goes to waste, so the effective
cutoff RISES exactly in the periods where processing binds. That is the qualitative difference between
CPIT and PCPSP made visible: in CPIT the cutoff is a number decided before the model ran.

`destinations.solve_opbsp_exact` solves the fully binary formulation exactly with HiGHS, in the
variables Jelvez et al. use, and returns `None` above a size budget rather than passing a heuristic
answer off as an exact one.

### 9.5 Lane's cutoff-grade policy

`refine.lane_cutoffs` computes the three limiting cutoffs and the MIDPOINTS between them. The
break-even cutoff makes a tonne pay for its own processing; Lane's point is that this is the wrong
cutoff whenever a capacity binds, because a marginal tonne consumes a scarce hour and pushes every
profitable tonne behind it further into the discount.

Two results the implementation reproduces and the tests assert:

- the **mine-limiting cutoff equals break-even**, which is not an oversight: when the shovels are the
  bottleneck the scarce hour is a mining hour, and ore and waste consume it identically, so the
  opportunity cost cancels out of the ore-versus-waste comparison.
- the economic cutoff **declines over the life of a mine**, because the opportunity cost falls as
  there is less value left to delay.

### 9.6 Minimum mining width

`refine.enforce_min_width` absorbs slivers into the period of their majority bench neighbour, toward a
target width. It reports how many blocks sat in a run narrower than the target before and after, and
what the change cost in NPV, because an operable plan is worth less on paper than an inoperable one
and hiding that is how a schedule looks better than it is.

Precedence cuts both ways here and the first version only checked one: moving a block later can be
overtaken by a successor already scheduled ahead of it.

### 9.7 Geological uncertainty, framed honestly

`stochastic` does NOT solve a two-stage stochastic integer program. It does what practitioners
actually do, in Blom, Pearce and Cote's words: solve many instances of a deterministic problem with
varied parameters and read the spread.

It reports the NPV distribution of each candidate plan across a spatially correlated, mean-preserving
ensemble; P10 and P90; the **robust choice by P10**, which is often not the plan with the best
expected value; the **optimism of the single-model forecast**; and the **value of re-planning** once the realisation is known, which requires the problem actually re-solved on each realisation and is a LOWER bound on EVPI rather than EVPI itself. The much smaller value of merely knowing which
candidate plan to pick is reported separately and never labelled EVPI.

Two modelling choices that are easy to get wrong and are asserted in the tests: the perturbation is
**spatially correlated** (uncorrelated noise averages out over a pushback and makes uncertainty look
harmless) and **mean-preserving** (box smoothing does not centre a finite field, and a biased
multiplier corrupts the optimism number the module exists to report).

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

## 10. Making the joint bound affordable (0.4.0)

The Bienstock-Zuckerberg pricing problem lives on the TIME-EXPANDED graph: a 10,976-block deposit
over ten periods is 109,760 nodes and 972,904 arcs. The package's pure-Python Dinic takes about two
seconds per closure there, and BZ needs one per iteration, so the joint bound was simply not computed
on any real deposit and the caller fell back to Algorithm 4's looser certified bound.

### The compiled path, and why its arithmetic is safe

`scipy.sparse.csgraph.maximum_flow` is compiled and takes INTEGER capacities. Scaling a float
objective to integers is where a bound usually stops quietly being a bound, so the rounding here is
DIRECTIONAL:

$$w'_b \;=\; \frac{\lceil s\, w_b \rceil}{s} \;\ge\; w_b
\qquad\Longrightarrow\qquad
\max_{C\ \mathrm{closed}} \sum_{b \in C} w'_b \;\ge\; \max_{C\ \mathrm{closed}} \sum_{b \in C} w_b$$
\max_{C 	ext{ closed}} \sum_{b \in C} w'_b \;\ge\; \max_{C} \sum_{b \in C} w_b$$

so the computed value can only OVER-estimate, by at most `n / s`. That direction is the entire
safety argument: the termination certificate is that `L(pi)` bounds the optimum for every dual
vector `pi`, and a pricing solve that came in LOW would produce a number that is not a bound and
nothing downstream could tell.

### The measured ceiling

`maximum_flow` accepts an int64 matrix and is WRONG above a total capacity of `2**31`. It does not
raise and it does not warn. On a closure whose true value is 769,925,542 it returns +3,210 at
`2**30`, inside the rounding slack as designed, and +15,491,856 at `2**31`, which is the entire
positive mass, meaning the flow came back as essentially zero. The module uses one power of two
below the observed failure and a test re-measures it, so a scipy release that moves the boundary is
caught in a test that names it rather than in a bound that drifted.

### Search fast, certify exactly

At that ceiling the slack is about `1e-4` relative on a time-expanded graph, which is the same order
as the tightening the joint bound exists to measure. A bound built from it therefore cannot answer
the question it was asked, and the first measurement showed exactly that: BZ came out ABOVE Algorithm
4 on a single-resource instance, a negative tightening, entirely inside the slack.

The fix is not more precision, it is the right division of labour. The rounded solves SEARCH for a
good dual vector; since `L(pi)` is a valid upper bound for every `pi`, ONE exact solve at the best
`pi` found turns the search into a certificate. `pricing_slack` comes back at exactly `0.0`.

Measured on that twin: 15 iterations, 15 seconds, and BZ agrees with the critical multiplier
algorithm to 1.3e-8 relative on a single resource. Two entirely different algorithms computing the
same LP is the check that says both are right.

### Two corrections to this section, both found by measurement (0.5.0)

**The market-limiting cutoff was wrong by a factor of `1/recovery`.** Lane's is
`g = h / (y (p - k - F/K))`, which in the notation here is `h / (margin - recovery * F / K)`. The code
divided by recovery where Lane multiplies, inflating the opportunity charge by `1/recovery^2`:
+1.6 percent at a recovery of 0.88 and +113.8 percent at 0.30. At the parameters of the package's own
test that cutoff is also the median, so `CutoffPolicy.optimum` returned the wrong number too.

**The balancing cutoffs were midpoints, and are now named as such.** Lane's balancing cutoffs are the
grades at which two capacities are exhausted at the same time, which is a property of the deposit's
GRADE-TONNAGE CURVE. `lane_cutoffs` is given economics only and has no distribution to integrate, so
it could never have computed them. The fields are `midpoint_mine_mill` and `midpoint_mill_market`
now: an interpolation between two limiting cutoffs is a usable number, and calling it Lane's
balancing cutoff was the part that was not true.

