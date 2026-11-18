# Validation strategy

`python -m pytest -q` covers the following independent properties:

- the supplied explicit P1 CIF expands to exactly 672 ASE atoms with the expected
  Li/Nb/Cl/O counts, volume, PBC, and unit occupancies;
- a one-atom simple-cubic crystal has six nearest periodic images and gives the
  same normalized RDF after a 2×2×2 supercell transformation;
- a seeded uniform random periodic configuration has a shell-volume-weighted
  RDF close to one, and exact raw counts survive 0.01→0.02 Å rebinning;
- neutral form factors use the explicit `Q/(4*pi)` conversion, approach atomic
  number at Q=0, decrease with Q for Li and Nb, and give non-negative weights
  summing to one;
- a unit RDF gives `S=1`, `F=0`, and `G=0`; the numerical sine transform agrees
  with an analytic transform of `F(Q)=Q`;
- the `Biso→Uiso` pair factor is exact, and Biso=0 differs from Biso=0.3 without
  modifying the input structure or RDF;
- a real-CIF short-cutoff end-to-end calculation produces finite nonzero G(r).

The production run additionally compares 20/30/40 Å RDF cutoffs and 0.01/0.02 Å
bin widths on both requested r ranges. Exact measured differences, tail RDF
normalization, peak positions, memory, timing, and finite-cutoff limitations are
written to `results/LNOC/validation_report.md` every time the CLI runs.

For an ideal crystal, truncated real-space integration produces ringing around
sharp reciprocal-space Bragg features. The default result follows the requested
rectangular treatment and does not hide that behavior with an unrequested
real-space taper. Consequently, individual truncated S(Q) samples can become
negative even though a physical scattering intensity cannot; the generated
report counts and locates those samples explicitly.
