# Theory and conventions

## Partial RDF convention

For each stored pair `alpha <= beta`, the implementation counts an ordered
central-alpha to neighbour-beta histogram over explicit lattice images. Only
the zero-translation self pair is removed. The normalization is

`g_ab(r_i) = H_ab(r_i) / [N_a * rho0 * c_b * DeltaV_i]`,

where `DeltaV_i` is the exact spherical-shell volume. Thus every partial tends
to one for a homogeneous configuration. Cross pairs are not doubled in the RDF;
the factor of two instead appears in the Faber-Ziman weight.

The image range is derived from the reciprocal cell and is valid even when the
cutoff exceeds half of the shortest cell height. This is not a minimum-image
calculation. The current CPU implementation builds image trees per neighbour
species and is correctness-oriented, not an incremental Monte Carlo engine.

## Reciprocal-space weighting and transform

The implemented equations are

`S_ab(Q)-1 = 4*pi*rho0 integral r^2 [g_ab(r)-1] sinc(Qr) dr`,

`w_ab(Q) = (2-delta_ab)c_a c_b f_a(Q)f_b(Q) / <f(Q)>^2`,

`S(Q)-1 = sum_ab w_ab(Q)[S_ab(Q)-1]`, `F(Q)=Q[S(Q)-1]`, and

`G(r) = (2/pi) integral_(Qmin)^(Qmax) F(Q)sin(Qr)dQ`.

RDF integration uses bin centres and the configured RDF bin width. Q and output
r integrations use explicit finite grids. The reduced PDF has units Å⁻² under
this convention. `G(0)` is defined as zero. The default Q window is rectangular;
the optional Lorch branch is off in the supplied configuration.

`xraydb.f0` is called with `Q/(4*pi)` because its argument is
`sin(theta)/lambda`, not scattering-vector magnitude Q. Neutral elastic `f0`
is used. The calculation does not include energy-dependent anomalous terms.

## Isotropic displacement

`Uiso = Biso/(8*pi^2)`. Independent isotropic displacement of a pair multiplies
its distinct-pair correlation by

`exp[-Q^2(U_a+U_b)/2]`.

For the common `Biso=0.3 Å²`, this becomes `exp[-Q² Uiso]`. It is applied exactly
once to each partial `S_ab(Q)-1`. Form factors and normalized Faber-Ziman weights
are not separately damped, and no real-space Gaussian is added. No undocumented
`qdamp` or `qbroad` is used.

## Finite-cutoff limitation

An ideal crystal retains long-range correlations, so `g(r)-1` does not rapidly
vanish. Truncating the RDF at 20, 30, or 40 Å therefore produces termination
effects and need not converge monotonically. The generated validation report
quantifies the differences; the 40 Å result is a tested approximation, not an
exact infinite-crystal claim.
