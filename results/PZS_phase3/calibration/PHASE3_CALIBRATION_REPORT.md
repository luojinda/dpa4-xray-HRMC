# Phase 3 parameter calibration and method comparison

Date: 2026-11-20  
Initial structure: original `PDF/PZS_quench_222.cif` for every chain  
PDF scale: `no_scale`  
Initial proposal displacement: `d=0.1 A`

## Completed calculation matrix

The seed-101 screen completed five Alternating settings (`100/5`, `250/5`,
`500/5`, `500/10`, `500/20`) and three Inline temperatures (100, 175 and
300 K), all for 10k attempted moves.  The selected settings were then repeated
at 10k with seeds 101/202/303 and validated at 100k with all three seeds.

## Seed-101 10k screening

| Method | Parameter | Rw | Delta E from MQ (eV) | Force RMS (eV/A) | Wall (s) |
|---|---:|---:|---:|---:|---:|
| Alternating | 100 accepted / 5 FIRE | 0.3715 | +27.3 | 1.105 | 106.6 |
| Alternating | 250 / 5 | 0.2142 | +84.1 | 1.783 | 52.6 |
| Alternating | 500 / 5 | 0.1812 | +121.0 | 2.444 | 36.8 |
| Alternating | 500 / 10 | 0.2128 | +88.0 | 2.021 | 41.1 |
| Alternating | 500 / 20 | 0.3337 | +40.7 | 1.420 | 45.6 |
| Inline | 100 K | 0.2441 | +45.4 | 1.287 | 305.4 |
| Inline | 175 K | 0.2085 | +63.5 | 1.638 | 303.4 |
| Inline | 300 K | 0.1884 | +80.9 | 1.904 | 291.2 |

Every screened endpoint retained 100% P and Si fourfold coordination and had no
final frozen-MQ hard-sphere violation.  The screen shows a continuous PDF versus
energy/geometry tradeoff rather than a single optimum.

`500/10` was selected as the balanced Alternating setting: it has essentially
the same Rw as `250/5`, comparable energy, and lower wall time.  `500/5` was not
selected despite its lowest Rw because its energy and force RMS were much worse.
100 K was selected for Inline because increasing temperature improved Rw while
systematically increasing energy, force and tetrahedral distortion.

At the common wall time of 36.78 s, selected Alternating `500/10` had reached
approximately 8215 attempts (`Rw=0.2508`, Delta E about +62.5 eV), while Inline
100 K had reached about 1100 attempts (`Rw=0.5421`, Delta E about +19.6 eV).
Thus Alternating reduces the PDF residual much faster per wall time, whereas
Inline pays the full DPA4 cost to control energy on every legal move.

## Cross-seed 10k reproducibility

Alternating `500/10` produced Rw values 0.2128, 0.2010 and 0.2104 for seeds
101/202/303.  Inline 100 K produced 0.2441, 0.2399 and 0.2427, with energy
increments +45.37, +45.67 and +45.05 eV.  These results justified the 100k
validation runs.

## Three-seed 100k validation

| Method | Seed | Rw | Delta E (eV) | Force RMS | P angle RMS | Si angle RMS | P CSM | Si CSM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Alternating 500/10 | 101 | 0.12816 | +211.37 | 3.059 | 10.65 | 11.13 | 2.55 | 3.52 |
| Alternating 500/10 | 202 | 0.13007 | +212.36 | 2.899 | 10.99 | 11.33 | 2.75 | 3.65 |
| Alternating 500/10 | 303 | 0.14142 | +180.08 | 2.417 | 9.86 | 10.73 | 2.26 | 3.08 |
| Inline 100 K | 101 | 0.20302 | +30.03 | 0.886 | 3.47 | 4.42 | 0.31 | 0.80 |
| Inline 100 K | 202 | 0.20310 | +29.42 | 0.913 | 3.75 | 4.13 | 0.36 | 0.75 |
| Inline 100 K | 303 | 0.20285 | +29.16 | 0.863 | 3.82 | 4.02 | 0.38 | 0.74 |

All six 100k endpoints retained 100% P/Si fourfold coordination and zero final
hard-sphere violations.

Mean +/- sample standard deviation:

| Method | Rw | Delta E (eV) | Force RMS (eV/A) | Wall per chain |
|---|---:|---:|---:|---:|
| Alternating 500/10 | 0.13322 +/- 0.00717 | +201.27 +/- 18.36 | 2.791 +/- 0.334 | 319.5 +/- 3.4 s |
| Inline 100 K | 0.20299 +/- 0.00013 | +29.54 +/- 0.45 | 0.887 +/- 0.025 | 3028 +/- 36 s |

Alternating completed 73--74 FIRE blocks per chain, but 53--56 blocks were
rolled back because FIRE crossed a frozen MQ hard-sphere cutoff.  The high
rollback fraction means the nominal `500/10` schedule no longer supplies
effective energy relaxation through most of the late chain.  This explains why
Rw continues to improve while DPA4 energy, force RMS and tetrahedral distortion
become poor.

Inline 100 K is slower by roughly 9.5x at 100k, but it is exceptionally
reproducible across seeds and preserves substantially lower energy, forces and
tetrahedral distortion.  Its 100k energy is also lower than its 10k energy
increment, while Rw improves, so the result is not simply a frozen MQ structure.

## Decision

For physical HRMC refinement, **Inline 100 K is the preferred current method**.
Alternating `500/10` is useful as a fast PDF-optimization reference but its 100k
endpoints are not recommended as final structures.  The current evidence does
not justify a million-step Alternating run.

A million-step Inline run should also not start automatically.  First evaluate
the requested `d=0.2/0.3 A` short pilots against the present `d=0.1 A` baseline,
including acceptance, effective displacement per wall time, energy, Rw and
geometry.  If `d=0.1 A` remains preferred, the three 100k Inline structures are
already the reproducible starting evidence for deciding whether the roughly
8--9 hour per-chain million-step cost is worthwhile.

## Reproducibility outputs

- `calibration_evaluations.json`: unified full energy/PDF/CN/geometry audit.
- `calibration_summary.csv`: flattened machine-readable table.
- `wall_matched_summary.csv`: nearest samples at the common wall time.
- `plots/energy_pdf_pareto.png`: energy-PDF Pareto comparison.
- `plots/selected_10k_100k_comparison.png`: selected-method comparison.
- Each run directory retains final CIF/extxyz, summary, trace and real
  accepted/rejected Metropolis examples for Inline.

No 1M chain was launched.
