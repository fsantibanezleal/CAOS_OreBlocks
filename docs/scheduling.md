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
| `"gershon"` | `w_b = sum of p_a over the successor SET B+(b)`, each successor once | Gershon 1987a (GeTS) |
| `"expected"` | `w_b = -E_b` from the LP relaxation | Chicoisne et al. 2012 (ExTS) |

with the expected extraction time

```
E_b = sum_{t=1..T} t (x*_bt - x*_b,t-1) + (T + 1)(1 - x*_bT)
```

read off the fractional LP solution. This is the bridge: the bound is not only the yardstick, it is
the seed of the plan. The published spread is large. On their AsiaMine instance with two resource
constraints, greedy reached 0.138 of the LP bound, Gershon 0.840 and expected-time 0.972, using the
same scheduling code.

**The Gershon weight counted paths before 0.6.0.** It summed each successor's already-accumulated
weight over a reverse topological sweep, so a block reachable from `b` along `k` precedence paths was
counted `k` times. With five or nine arcs per block the path count grows geometrically with depth and
the weight was dominated by the deepest blocks times their multiplicity; on a downstream product's
twelve non-trivial cases GeTS lost to greedy on seven. The cones are now bitsets (Python integers)
built in reverse topological order as the union of each successor's cone plus that successor, and a
cone is released once every predecessor has read it. A test compares the result with the set
definition computed the slow way.

With the definition exact, GeTS is still not uniformly better than greedy, and the reason is the
weight itself: it rewards what a block unlocks and ignores what it costs to reach. On a narrow vein
every block along the strike has the vein in its cone, so the order opens the whole strike length at
once and pays for its waste early. Measured on four seeded 10-period twins with two resources, as a
fraction of the Algorithm 4 bound: porphyry 0.810 against greedy 0.556; core-halo 0.779 against
0.136; layered 0.830 against 0.857; vein 0.174 against 0.762. ExTS reached 0.715 to 0.969 on the same
four.

`improve_schedule` then applies a **shift** local search: pull positive-value blocks forward when
precedence and capacity allow, push negative-value blocks back when their successors allow. Both
moves are strictly improving under discounting and preserve feasibility. This is the shift
neighbourhood family used across the mine-scheduling metaheuristic literature (Lamghari, A. and
Dimitrakopoulos, R., European Journal of Operational Research 222(3), 2012,
[doi:10.1016/j.ejor.2012.05.029](https://doi.org/10.1016/j.ejor.2012.05.029)). It is deliberately
**not** the exact `C-PIT[D]` neighbourhood of Chicoisne et al. section 3.3, which re-solves a
restricted integer program per neighbourhood and needs a MILP solver; that one is
`refine.exact_local_search`, section 9.2.

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

**The candidate set (0.6.0).** The published method solves the full model per window. Without a
commercial MILP solver the window works on a candidate set: the undecided blocks ordered by the LP
expected extraction time `E_b` of the tightest relaxation, taken from the front until their extraction
tonnage covers `cover` times the WINDOW's capacity, never more than `cand_max` (the method raises
rather than starving the window). The relaxation is closed in every period, so `E_a <= E_b` on every
arc (`b` requires `a`) and the prefix is predecessor-closed; a test asserts that inequality, and the
closure is still completed explicitly so a tie cannot block a candidate. Before 0.6.0 the set had to
cover the window plus the whole remaining horizon, ordered by value density, and the `relaxation`
argument was never read: on a downstream product's thirteen cases the method refused on twelve.
Measured after the change on that product's 6,912-block porphyry twin (8 periods, two resources):
1.34 percent below the bound, against 4.62 percent for ExTS and 4.22 percent for the exact C-PIT[D]
local search, in 17 minutes on one core.

**Warm starts (0.6.2), and what they do not fix.** Every window MILP, and every re-solve of the two exact
local searches, now goes through `oreblocks._milp.solve_binary_program`, which hands HiGHS a feasible
start through `highspy` (`setSolution`) and never answers below it. A window's start is the better of two
plans built by one repair pass over its candidates: the LP-guided greedy (each block, in `E_b` order, to
the first slot its predecessors and the remaining capacities allow) and the previous slide's answer for
the blocks still open. Measured on a downstream product's 14,400-block layered twin (ten periods, two
resources, windows of 19,356 binaries and about 150,000 rows, `mip_gap` 3 percent):

| window | start | answer | time | nodes |
|---|---:|---:|---:|---:|
| 1 | 753.80 M | 755.56 M | 9.9 s | 1 |
| 2 | 548.15 M | 548.23 M | 12.0 s | 1 |
| 3 | 438.68 M | 438.68 M | 30.6 s | 1 |
| 4 | 337.64 M | 370.41 M | 165.8 s | 1 |
| 5 | 275.65 M | 323.76 M after 13.6 min, still open | - | 0 |

Windows 1 to 4 close at the root. Window 5 is the hard slide of the 0.6.1 measurement: its root LP takes
143 s and sits at 350.6 M, 27 percent above the start, and root cuts barely move it (350.58 M after
380 s). HiGHS's sub-MIP heuristic found 323.76 M after 13.6 minutes from the warm start, where the cold
0.6.1 solve had 315.3 M after 18 minutes. What remains is the PROOF: a 3 percent gap needs the bound
near 334 M, and on a window LP this loose that is the branch and bound, at about 16 seconds a node. A
`node_limit` (a count, so it is reproducible where a time limit is not) does not help this window: with
the sub-MIP heuristics off and 200 nodes it ran 3,569 s and never improved its start. The option stays,
documented and tested, for windows where the tree is cheap; closing the hard slides needs a smaller
window model (for example blocks aggregated into bench-phase units), which is open work.

**The warm start cost quality, so the window keeps its plan as a FLOOR (0.6.3).** The first case of a
downstream re-bake on 0.6.2 came out worse: on the 14,400-block vein twin the whole method fell from 0.17
percent below the bound to 1.315. At a 3 percent window gap HiGHS proves the gap from a handed-in start
almost at once and stops near it, so every window fixes a period close to the greedy plan, and the slides
compound it. Measured on the same twin:

| window solve | gap of the whole method | time |
|---|---:|---:|
| warm start, `mip_gap` 3% | 1.315% | 1.3 min |
| warm start, `mip_gap` 1% | 1.088% | 2.1 min |
| warm start, `mip_gap` 0.5% | 0.209% | 14.2 min |
| cold, plan kept as a floor, `mip_gap` 3% (the default since 0.6.3) | 0.097% | 26.8 min |

The floor was the answer on 2 of the 10 windows. On a 6,912-block porphyry twin the same comparison gives
1.274 percent warm (47.9 min) and 1.113 percent cold with the floor (52.7 min), against 1.34 on 0.6.1. `warm_start=True` keeps the 0.6.2 behaviour for a caller
who wants the speed. The exact local searches keep their warm start: they stop at a 1e-4 gap and accept
only proven improvements, so a start cannot anchor them.

### 9.4 Destinations: when the cutoff grade becomes an output

`destinations.destination_toposort` walks a weighted topological order (pass `weight=-E_b` for the
ExTS order) and, for every destination of a block, finds the EARLIEST period not before any predecessor
whose remaining resources take it there; it keeps the destination whose discounted value at that
period is the largest. Ore therefore waits for the plant when the plant a period later is worth more
than the dump now, and falls to the dump when it is not, so the effective cutoff is an output of the
schedule. Before 0.6.0 it walked greedy weights by default and took the first period where ANY
destination fitted, which dumped ore the moment the plant was full; on a downstream product's cases it
ended below the fixed-destination plan everywhere, although choosing the destination is the richer
problem. With the plant never binding it now reduces to CPIT TopoSort block for block, which a test
asserts.

`destinations.lift_to_pcpsp` reads a fixed-destination plan as a PCPSP plan (every block at its
a-priori best destination, which is exactly how `Pcpsp.to_cpit` builds the CPIT instance), so its value
is the CPIT value. `destinations.exact_destination_local_search` is the C-PIT[D] re-solve with
destinations: the same three neighbourhoods, a restricted model whose free blocks carry cumulative
extraction variables `y_it` and binary destination variables `z_idt` linked by
`sum_d z_idt = y_it - y_i,t-1`, solved with HiGHS. Started from a lifted plan it can never end below
the CPIT plan, which is the property the richer problem has to show.

`destinations.pcpsp_lp_bound` solves the PCPSP LP relaxation (cumulative extraction monotone and closed
in every period, destination fractions linked to it, resource rows over the fractions) with HiGHS over
every block of the instance, and returns `None` above a row budget. On the published `newman1.pcpsp`
it gives 24,486,549.02 in about two seconds; the published PCPSP LP upper bound is 24,486,549
(Jelvez, Morales and Nancel-Penard 2018, Table 3, from two Bienstock-Zuckerberg implementations). On an
instance where only mining binds and every destination consumes it alike, the PCPSP LP must equal the
CPIT LP, and a test asserts it against the critical multiplier algorithm: two unrelated solvers on one
quantity.

`destinations.solve_opbsp_exact` solves the fully binary formulation exactly with HiGHS, in the
variables Jelvez et al. use, and returns `None` above a size budget rather than passing a heuristic
answer off as an exact one.

**The re-cut (0.6.1): let the relaxation choose the cutoff.** The destination TopoSort compares a
block's discounted values at its destinations and nothing else. When the plant binds that comparison
is myopic: a marginal ore block has a positive plant value and a negative dump value, so it always
goes to the plant, and the plant tonnage it takes is gone for the richer ore below it. That is Lane's
point about a mill-limited operation (the cutoff rises above break-even by the opportunity cost of the
mill), and the PCPSP relaxation prices it. `pcpsp_lp_bound(..., solution=True)` returns, with the
bound, each block's expected extraction time and the share of it the LP sends to each destination;
`PcpspBound.preferred_destination()` reads the dominant one, and `restrict_destinations` fixes it. The
restricted instance's `to_cpit()` is a CPIT whose plans are plans of the original PCPSP with the same
value, so the whole CPIT machinery schedules it: its own critical-multiplier relaxations, ExTS, the
sliding window, and then `exact_destination_local_search` on the original instance, where every
destination is free again.

Measured on a porphyry twin whose plant takes half of the pit's ore tonnage (`twin-porphyry-s`
economics and scenario, scaled down so the integer problem can be solved): on 320 blocks the exact
OPBSP incumbent after 300 s is 10.37 M (MIP bound 10.50 M, PCPSP LP 11.01 M), the best fixed-cutoff
plan 6.51 M, the destination TopoSort of 0.6.0 -0.79 M, and the re-cut scheduled by the sliding window
10.15 M. On 1,008 blocks the re-cut reaches 34.31 M with the sliding window and 34.81 M after the
local search, against 26.78 M for the best fixed-cutoff plan and a PCPSP LP of 36.03 M. A test asserts
on a smaller deposit that the re-cut ExTS plan is feasible, under the PCPSP LP, and above 1.15 times
the fixed-cutoff ExTS plan (measured 1.25 to 1.35 on three sizes).

A first attempt walked the PCPSP LP's own expected times and sent each block to its preferred
destination inside the destination TopoSort. It matched the re-cut on 320 blocks and fell to
26.41 M on 1,008: the PCPSP relaxation mines deep cones fractionally from period one, so its expected
times are a poor order for an integer plan. The CPIT relaxation of the re-cut instance does not have
that freedom on the destination side, and its order is the one that works. The attempt is not in the
package.

**The PCPSP bound at scale: the Lagrangian dual (0.6.1).** The direct LP has about `n T` precedence rows
times the arc density; on a 14,400-block, ten-period deposit that is 1.44 million rows, and HiGHS did not
finish two such instances in six and a half hours (its interior-point method was slower than its simplex
on the 6,912-block instance). `destinations.pcpsp_lagrangian_bound` dualises the `R T` capacity rows. For
multipliers `mu >= 0`, each block and period takes the destination worth most at those prices,
`g_bt = max_d (disc_t v_bd - sum_r mu_rt q_rbd)`, and what remains is a maximum closure on the
time-expanded graph (node `(b, t)` requires `(a, t)` for each predecessor and `(b, t + 1)`), with weights
`g_bt - g_b,t+1`. `L(mu) = sum mu c + closure value` bounds every destination schedule for every `mu`, and its
minimum is the LP value, because the inner problem is a closure. The closure is the compiled one, which
rounds weights up, so every `L(mu)` over-estimates and the reported bound is valid at any iteration; once
converged it exceeds the LP by at most the returned rounding slack. A cutting-plane method with a box trust
region drives `mu`. Measured against the exact LP: 0.2 to 1.5 parts per million above it on three small
instances and 4.8 parts per million on a 6,912-block twin (316,476,932 against 316,475,407, in 166 seconds
against about 18 minutes). The relaxed schedule at the best `mu` is returned as one-hot destinations, which
is what the re-cut fixes; on a 1,008-block twin the re-cut on those destinations is within 0.3 percent of
the re-cut on the LP's own.

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

Since 0.6.0 a move is made only if every resource of the receiving period has room, the input must be
capacity-feasible, and the output is then feasible by construction; the report counts the moves
refused for capacity. Before, capacity was not re-imposed: a downstream product's vein twin reported a
smoothed NPV of 288.9 M against a certified upper bound of 285.4 M, which no feasible plan can do, and
the "cost of operability" mixed the price of operability with the value of overrunning a mill.

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

