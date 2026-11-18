# References and their use in Phase 1

1. Neder & Proffen, *J. Appl. Cryst.* **53** (2020) 710–721,
   <https://doi.org/10.1107/S1600576720004616>. Motivates retaining the true
   Q dependence of X-ray form factors instead of using fixed WKM weights.
2. Sławiński et al., *J. Appl. Cryst.* **57** (2024) 1251–1262,
   <https://doi.org/10.1107/S1600576724004175>. Provides a published precedent
   for the route partial correlations → reciprocal-space Faber-Ziman weighting
   → Fourier-transformed G(r).
3. Ehrhardt et al., *J. Comput. Chem.* **47** (2026) e70306,
   <https://doi.org/10.1002/jcc.70306>. Methodological context for partial-RDF
   based total-scattering calculations; no code was copied.
4. Singh et al., *JACS* **146** (2024) 17158–17169,
   <https://doi.org/10.1021/jacs.4c03142>. Qualitative material context only.
   No experimental XY data or processing parameters were assumed.
5. XrayDB Python API, <https://xraypy.github.io/XrayDB/python.html>. Defines the
   `f0(element, sin(theta)/lambda)` interface used here.

The Wang/amsrmc, DPA4, RMC, HRMC, and incremental-update topics in the project
specification are deliberately outside this phase and have not been implemented.
