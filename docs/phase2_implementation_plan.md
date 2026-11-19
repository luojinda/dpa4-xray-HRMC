# Phase 2 implementation plan

Phase 2 extends, rather than replaces, the Phase 1 partial-RDF X-ray forward
model.  The reference oracle remains `compute_partial_rdfs()` followed by
`forward_transform()` with Q-dependent `xraydb.f0` weights.

## Frozen scientific scope

- Initial structure: `PDF/PZS_quench_222.cif` (896 atoms, P1, fixed cell).
- Experimental target: `PDF_ZXS_1_100keV_350mm_10s_eiger16m_slice_0_.gr`.
- Fit window: 1.5--10.0 A, uniform point weights.
- X-ray model: Q=0.9--16.0 A^-1 on the Phase 1 0.05 A^-1 grid,
  Biso=0.3 A^2, no Lorch modification.
- Only single-atom Cartesian moves, pair-specific hard-sphere rejection, and
  the selected PDF residual enter the Monte Carlo acceptance rule.
- No DPA4, potential energy, coordination, bond-angle, or rigid-unit term is
  installed or called.

## Correctness-first sequence

1. Derive a float64 count-histogram-to-G(r) kernel from the existing Phase 1
   equations, including the fixed `g-1` background.
2. Implement transactional single-atom histogram deltas.  Use the O(N)
   minimum-image path only when the cutoff is below half the minimum cell
   height; retain an exact multi-image reference path otherwise.
3. Require incremental trial/commit/reject results to reproduce independent
   Phase 1 full recomputation before any RMC run.
4. Implement and test the three scale objectives and pair-specific
   hard-sphere feasibility check.
5. Measure 12/20/30/40 A cutoff sensitivity and cost, then freeze one common
   cutoff for all modes.
6. Run 10k pilots, freeze displacement and sigma_RMC, then run 100k and the
   requested 1M x 9 benchmark with checkpoints and exact attempted-step
   accounting.

Large kernels, experimental inputs, checkpoints, and chain artifacts live on
the AutoDL data disk.  Reproducible summaries and selected final artifacts are
copied into `results/PZS_phase2/` for version control after audit.
