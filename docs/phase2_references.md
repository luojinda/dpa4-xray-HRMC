# Phase 2 references and implementation boundary

1. McGreevy and Pusztai, *Molecular Simulation* **1** (1988) 359–367,
   <https://doi.org/10.1080/08927028808080958>. Used for the conventional
   single-particle RMC/Metropolis framework.
2. Wang et al., *A practical route to amorphous solid-state electrolyte
   structures by hybrid reverse Monte Carlo with foundation potentials*,
   <https://arxiv.org/abs/2610.19846>. Used as scientific motivation for the
   Pure-RMC control and for post-run local-structure checks. Its reported
   neutron-PDF treatment is not treated as an X-ray implementation.
3. AM3GroupHub/amsrmc, <https://github.com/AM3GroupHub/amsrmc> (MIT). Its
   histogram organization was reviewed as a design reference; no source was
   copied. This project derives its own Q-dependent X-ray delta kernel from
   the existing Phase 1 equations.
4. Neder and Proffen, *J. Appl. Cryst.* **53** (2020) 710–721,
   <https://doi.org/10.1107/S1600576720004616>. Supports retaining the full
   Q dependence of X-ray form factors in fast PDF calculations.
5. Sławiński et al., *J. Appl. Cryst.* **57** (2024) 1251–1262,
   <https://doi.org/10.1107/S1600576724004175>. Provides big-box RMC and
   scattering-data context; multiphase fitting and extra instrument
   refinement are outside this phase.
6. fullrmc documentation,
   <https://bachiraoun.github.io/fullrmc/fullrmc.Core.html>. Consulted for
   scale-factor practice only. Phase 2 instead uses the explicitly specified
   per-valid-proposal bounded analytic optimum.

Phase 2 contains no DPA4/MLIP calculation, potential-energy term,
coordination constraint, bond-angle constraint, or rigid-polyhedron term.
Coordination and partial-RDF quantities are generated only after sampling as
diagnostics.
