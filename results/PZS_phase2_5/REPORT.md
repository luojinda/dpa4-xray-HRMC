# PZS Phase 2.5 CN-constrained Pure RMC report

## Scope and frozen model

Phase 2.5 adds only fixed-cutoff coordination penalties to the unchanged Phase 2 X-ray PDF, scale strategies, move distribution, and hard-sphere model. No DPA4/energy term, CrystalNN/ChemEnv hot-loop call, or extra coordination rule was used.

Frozen cutoffs (A): P-O 1.80, Si-O 1.85, Si-Cl 2.30, Zr-O 2.50, Zr-Cl 3.00. Frozen weights shared by all modes: P 800, Si 2500, Zr-total 1.0, Zr-O 0.2.

## Initial audit and gates

The 896-atom MQ structure has P-O=4 for 32/32 P and Si-(O+Cl)=4 for 64/64 Si. Initial Zr total CN is 63x6 + 1x5; Zr-O counts are 19x0 + 23x1 + 18x2 + 4x3. The selected cutoff distributions are unchanged under +/-0.10 A cutoff sensitivity. Offline CrystalNN and fixed-cutoff classifications agree site-by-site for all 160 P/Si/Zr centers.

Incremental validation: 500 trial moves and 10,000 mixed commit/reject moves gave integer error 0 and component error 0; incremental CN proposals were 59.2x faster than full CN recomputation. With all CN weights exactly zero, a 2,000-step Phase 2/2.5 regression was bitwise identical in accept decisions, coordinates, histograms, G(r), and score. Checkpoint/restart was bitwise identical to an uninterrupted chain.

## Formal calculations

All 10k pilot, 100k x 9 gate, and 1M x 9 formal chains completed. The 1M chains ran at 274.6-362.3 attempted moves/s and accepted 332,715-373,561 moves. Full PDF/CN validation was performed every 50k steps; maximum CN integer drift was zero. Final hard-sphere violations were zero in all nine chains.

| mode | mean final Rw (Phase 2) | mean final Rw (Phase 2.5) | P fourfold | Si fourfold |
|---|---:|---:|---:|---:|
| no_scale | 0.12331 | 0.12351 | 100% | 100% |
| fixed_initial | 0.10864 | 0.10492 | 100% | 100% |
| bounded_dynamic | 0.10482 | 0.09812 | 100% | 100% |

Final Phase 2.5 scales: no_scale=1; fixed_initial=0.63577698; bounded_dynamic is clipped at its upper bound 0.76293238, while its raw optimum is 0.76996-0.77017. Zr distributions remain close to the MQ target: most chains finish with 62xCN6 + 2xCN5; the largest observed deviation is one CN4 site or three CN5 sites. Zr-O histograms vary only by one or two sites per bin.

Offline CrystalNN was rerun after RMC for all nine final structures: all 160 target centers per chain were processed with zero errors. CrystalNN was not imported by the RMC runner.

## Interpretation

The unconstrained Phase 2 result severely damaged tetrahedral topology (mean Si fourfold fractions 5.2%-31.3%, depending on scale mode). Phase 2.5 preserves all P/Si tetrahedra while matching or improving PDF Rw for fixed_initial and bounded_dynamic; no_scale Rw is statistically unchanged. The bounded-dynamic scale remains at the allowed upper bound, so scale-model uncertainty is still active and should not be mistaken for a structural degree of freedom.

Files include every final, best-total, and best-PDF CIF/extxyz structure, G(r) tables, traces, timing/summary JSON, coordination tables, audit data, plots, and gate evidence. Large checkpoints/snapshots are retained on AutoDL but intentionally excluded from Git.
