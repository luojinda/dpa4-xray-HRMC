# Phase 2 incremental X-ray PDF derivation

The Phase 1 histogram convention for pair `p=(a,b)`, `a<=b`, is

```text
g[p,j] = H[p,j] / (N_a * (N_b / V) * shell_volume[j]).
```

For unlike species, `H` stores the ordered lower-element central atom to
higher-element neighbour count.  For equal species, a distinct atomic pair is
therefore present in both central/neighbour directions; a move contributes
twice to the equal-species histogram.  Non-zero self images are part of the
full Phase 1 histogram but their distances are translation invariant and have
zero single-atom delta.

With the cell, composition, grids, Biso, and window fixed, substituting the
histogram normalization into the Phase 1 equations gives

```text
G[k] = G_background[k] + sum_pj K[k,p,j] * H[p,j].
```

`K` is built in float64 from all of the following unchanged Phase 1 factors:

- shell-volume histogram normalization;
- `4*pi*rho0*r_j^2*dr` radial quadrature;
- `sinc(Q*r_j)` partial structure-factor kernel;
- Q-dependent Faber--Ziman X-ray pair weights;
- the pair Debye--Waller factor from `Biso=0.3 A^2`;
- `Q[S(Q)-1]`, trapezoidal Q weights, finite Qmin/Qmax, and the sine transform.

The fixed `-1` part of every partial `g-1` term is retained in
`G_background`.  It cancels from a move delta but is required for the initial
affine reconstruction test.

For a moved atom, old bins are decremented and new bins incremented, duplicate
indices are coalesced, and

```text
delta_G = delta_H @ K.
```

Proposals do not mutate positions, counts, or G.  Commit applies all three
updates once; reject is a no-op.  The fast O(N) distance path uses minimum-image
vectors only when `r_cut < 0.5 * minimum_cell_height`.  The implementation also
contains a slower explicit multi-image path using the same conservative image
bounds as Phase 1, so using a larger cutoff cannot silently change the physics.

For the PZS 896-atom cell, `r_cut=12 A` is below half the 24.742 A minimum
height.  The correctness run compares every trial against an independent
Phase 1 full recomputation and records accumulated drift without periodic
rebuilding.
