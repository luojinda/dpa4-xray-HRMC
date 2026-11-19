# PZS Phase 2: incremental X-ray PDF and Pure-RMC report

## Completion and scope

Phase 2 was completed on the AutoDL CPU container on 2026-11-19. The executed
work comprised one 10,000-attempt pilot for each scale mode, nine 100,000-attempt
chains, and the requested nine 1,000,000-attempt formal chains. All formal
chains reached exactly 1,000,000 generated/attempted single-atom moves, for
9,000,000 formal attempts total. The AutoDL test suite passed (`21 passed`).

Only the PDF residual, pair-specific hard-sphere rejection, and single-atom
coordinate moves entered the acceptance rule. No DPA4/MLIP calculation,
potential-energy term, coordination constraint, angle constraint, or rigid
polyhedron constraint was installed or called. Coordination and partial RDFs
below are post-analysis only.

The formal chains used source commit `ea654a2`; later commits only extended
documentation, benchmarking, and result post-processing. All chains were run
with `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`, nine chains in
parallel on the 32-vCPU container.

## Inputs and frozen forward model

- Structure: `PZS_quench_222.cif`, 896 atoms,
  `Cl512Li96O128P32Si64Zr64` = 32 x `Li3PO4Zr2Si2Cl16` exactly.
- Cell: 29.379999 x 24.742001 x 28.000000 A, PBC in all directions; number
  density 0.0440213525 A^-3. The minimum cell height is 24.742001 A.
- Experiment: ZXS_1 `.gr`; header Qmin=0.9 A^-1, Qmax=16.0056 A^-1,
  rstep=0.01 A. The simulation grid is 0.9--16.0 A^-1 in 0.05 A^-1 steps,
  i.e. 0.0056 A^-1 below the header endpoint rather than reverting to 24 A^-1.
- Fit window: 1.5--10.0 A, uniform point weights, no modification of the
  experimental values and no extra smoothing or baseline shift.
- Phase 1 physics retained: Q-dependent `xraydb.f0(Q/4pi)`, Biso=0.3 A^2,
  finite-Q transform, no Lorch modification.
- RDF histogram: dr=0.01 A and rcut=12 A. Since 12 < 24.742001/2, the
  triclinic-aware minimum-image fast path is valid for this orthorhombic cell.
  The same cutoff was used for all modes.
- Moves: independent Cartesian Uniform(-0.10,+0.10) A components;
  `sigma_RMC=0.030 A^-2`; pair-specific hard-sphere cutoffs are 0.95 times
  the initial pair minima. These values were frozen after the 10k pilot.

The common first-frame scale is
`s0=0.6357769831752518`, with dynamic bounds
`[0.5086215865402014, 0.7629323798103022]`.

## Incremental PDF correctness and cost

The implementation stores the Phase 1 pair-count histogram and a float64
affine count-to-G(r) kernel. A move visits the moved atom's correlations,
coalesces the sparse bin changes, and evaluates delta-G from the affected
kernel rows. Thus the distance/histogram part is O(N) for the safe
minimum-image path; the complete update also has a cost proportional to the
number of changed histogram bins times the fitted-r grid. It is not claimed
that the entire algorithm is strictly O(N) independent of the Q/r grids.

Correctness results on the real 896-atom PZS structure:

| Check | Maximum G error (A^-2) | Histogram error |
|---|---:|---:|
| Initial incremental vs independent Phase 1 full | 9.77e-15 | 0 |
| 100 reproducible trial moves | 1.02e-14 | 0 |
| After 10,000 mixed commit/reject operations | 1.49e-14 | 0 |
| Maximum over 180 full checks in the nine 1M chains | 2.07e-13 | 0 |

The maximum formal-chain RMS drift was 1.95e-14 A^-2. Reject operations were
byte-identical for positions, histogram, and G(r). No periodic rebuilding was
used to hide drift. A real checkpoint resume from 100,000 to 101,000 attempts
was also executed successfully with restored histogram, PDF coefficients, and
RNG states.

The float64 kernel occupies 171,568,408 bytes (163.62 MiB). In an idle
single-task AutoDL benchmark its initialization took 0.638 s. For 1,000
incremental and 100 full reference proposals:

| PDF operation | Median | p95 |
|---|---:|---:|
| Incremental proposal | 1.945 ms | 2.035 ms |
| Phase 1 full recompute | 234.564 ms | 277.234 ms |

The PDF-only median speedup was 120.62x, with a maximum incremental/full
difference of 1.04e-14 A^-2. Under the concurrent nine-chain load the measured
speedup was 111.82x (2.840 vs 317.579 ms); that number is retained separately
and is not mixed with the idle benchmark. No full-recompute end-to-end 100k
chain was run, so an end-to-end full-chain speedup is deliberately not quoted.

On the same cached delta-G sequence, score evaluation cost was 8.559 us for
`no_scale`, 8.539 us for `fixed_initial`, and 13.358 us for
`bounded_dynamic`. The analytic clipping therefore adds little compared with
the roughly 1.9 ms PDF update.

The rcut sensitivity relative to a 40 A periodically replicated reference was:

| rcut | max abs difference (A^-2) | RMS (A^-2) | relative RMS |
|---:|---:|---:|---:|
| 12 A | 0.014947 | 0.005213 | 1.056% |
| 20 A | 0.003484 | 0.001710 | 0.347% |
| 30 A | 0.000876 | 0.000459 | 0.093% |

Twelve angstrom was frozen because it remains inside the half-cell limit,
avoids fitting repeated-box medium-range correlations, limits the kernel to
163.62 MiB per chain, and changes the curve much less than the initial
experiment residual. This is a modeling choice, not a claim that 12 A is the
infinite-cutoff result.

## Formal 1M x 9 PDF results

`Rw=sqrt(sum residual^2 / sum Gexp^2)` over 1.5--10.0 A. Values below are
mean +/- sample standard deviation over three seeds. Each row must be read
under the named common diagnostic; the three sampling losses are not directly
interchangeable.

| Sampling mode | Final Rw(no scale) | Final Rw(fixed s0) | Final Rw(bounded) | Final Rw(shape-only) | Best mode-relevant Rw | Acceptance | attempts/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_scale | 0.12331 +/- 0.00175 | 0.37005 +/- 0.00108 | 0.25218 +/- 0.00116 | 0.12182 +/- 0.00154 | 0.12057 +/- 0.00076 | 0.33471 +/- 0.00193 | 293.5 +/- 30.1 |
| fixed_initial | 0.54680 +/- 0.00643 | 0.10864 +/- 0.00236 | 0.10651 +/- 0.00177 | 0.10651 +/- 0.00177 | 0.10499 +/- 0.00100 | 0.38566 +/- 0.00114 | 312.7 +/- 15.6 |
| bounded_dynamic | 0.32399 +/- 0.00125 | 0.19743 +/- 0.00119 | 0.10482 +/- 0.00189 | 0.10480 +/- 0.00189 | 0.10062 +/- 0.00196 | 0.38607 +/- 0.00096 | 303.9 +/- 58.7 |

For reference, the initial MQ values were Rw(no scale)=0.71309 and
Rw(fixed/bounded/shape-only)=0.51806. All modes therefore changed the PDF
substantially. `no_scale` is best under the common s=1 diagnostic, whereas
`bounded_dynamic` is best under bounded and shape-only diagnostics. This is
why comparing only each mode's own logged loss would be misleading.

For the final mode-scaled curves, short-range (1.5--4.5 A) RMS values were
0.07252, 0.06367, and 0.06097 A^-2 for no-scale, fixed, and dynamic;
medium-range (4.5--10 A) values were 0.01704, 0.01552, and 0.01597 A^-2.
Dynamic is best in the short range, while fixed is marginally lower in the
medium range. Final first-peak positions are 2.49 A for all three modes; the
second peak is about 3.47--3.48 A. Experiment gives 2.50 and 3.47 A. The
experimental second/first peak-height ratio is 0.4297, versus 0.4558,
0.4716, and 0.4583 for no-scale, fixed, and dynamic final curves. The initial
MQ ratio was 0.9955.

Per-chain wall times were 2,700--3,766 s (45.0--62.8 min) while nine chains
ran concurrently. Mean cumulative per-chain times were approximately
1,284--1,411 s in hard-sphere checks, 1,863--1,961 s in histogram/delta-PDF,
19.8--29.4 s in scoring, 13.3--14.4 s in commit, and 5.9--6.5 s in full drift
validation. Checkpoint I/O was below 0.17 s in the measured runner timer.

## Scale behavior

No-scale and fixed-scale traces remain exactly at 1 and s0. In dynamic mode,
all three final scales equal the upper bound 0.7629323798103019. The mean trace
scales were 0.76210, 0.76236, and 0.76240, and the fractions of recorded trace
points exactly at the upper bound were 74.3%, 82.9%, and 87.3% for seeds 101,
202, and 303. The experiment therefore prefers a scale beyond the allowed
range for much of sampling. Dynamic's small advantage partly uses the allowed
amplitude freedom and must not be interpreted as structural improvement alone.

## Post-run structure diagnostics

All final and best structures retain 896 atoms, the exact species order, cell,
PBC, density, and finite coordinates. Every formal final/best structure has
zero hard-sphere violations. Fixed cutoffs were derived once from the initial
MQ partial RDF and then reused for every structure.

| State (three-seed mean) | P-O mean CN | P fourfold fraction | Si-O mean CN | Zr-O mean CN | Zr-Cl mean CN | displacement RMS (A) |
|---|---:|---:|---:|---:|---:|---:|
| initial MQ | 4.000 | 1.000 | 0.719 | 1.109 | 4.875 | 0.000 |
| no_scale final | 3.885 | 0.885 | 0.703 | 0.901 | 4.885 | 0.814 |
| no_scale best | 3.917 | 0.917 | 0.693 | 0.953 | 4.885 | 0.786 |
| fixed_initial final | 2.521 | 0.135 | 0.266 | 0.724 | 4.880 | 0.815 |
| fixed_initial best | 2.573 | 0.135 | 0.323 | 0.714 | 4.880 | 0.774 |
| bounded_dynamic final | 3.219 | 0.406 | 0.432 | 0.870 | 4.875 | 0.854 |
| bounded_dynamic best | 3.375 | 0.500 | 0.500 | 0.839 | 4.875 | 0.797 |

The initial MQ file already has zero fourfold Si sites under the automatically
selected Si-O first-shell cutoff, so a "SiO4 preservation fraction" cannot be
claimed for this input. Nevertheless, its mean Si-O CN drops strongly for
fixed and dynamic modes. P-O is the clearest damage indicator: fixed-scale
sampling leaves only 13.5% fourfold P, and dynamic leaves 40.6% in final states
(50.0% in best states), despite their lower PDF residuals. Zr-Cl is nearly
unchanged, while Zr-O and Li local coordination also decrease. Maximum atomic
displacements reach roughly 3.2--4.5 A depending on mode.

Therefore the current Pure RMC residual can be used as a numerical PDF target,
but it is not a sufficient physical structure objective. For the later HRMC
stage, the DPA4 energy term should regularize the severe local-chemistry damage;
the present result is the required energy-free control. The scale should remain
explicitly monitored because dynamic scale saturation can mask amplitude and
coordination changes.

## Reproducible artifacts

The version-controlled result bundle contains correctness JSON/data/plots,
idle and concurrent benchmarks, 10k and 100k summaries, the nine formal
traces/configurations/manifests/timings, final and best CIF/extxyz structures,
calculated curves, scale evolution, partial-RDF data, and comparison plots.
The main tables are:

- `formal_1M/comparison_summary.csv`
- `formal_1M/comparison_statistics.json`
- `formal_1M/scale_evolution.csv`
- `formal_1M/structural_diagnostics.csv`
- `formal_1M/initial_baseline.json`
- `correctness/incremental_validation.json`
- `benchmarks/incremental_benchmark_idle.json`

Full 50k checkpoints and snapshots remain on the AutoDL data disk at
`/root/autodl-tmp/PZS_phase2_formal/<mode>/seed_<id>/checkpoints` and
`snapshots`; they were intentionally excluded from Git. The complete formal
output remains at `/root/autodl-tmp/PZS_phase2_formal`.

Example reproduction commands:

```bash
python scripts/validate_incremental.py --config configs/pzs_phase2_common.yaml
python scripts/benchmark_incremental_pdf.py --config configs/pzs_phase2_common.yaml
python scripts/run_phase2_pzs.py --config configs/pzs_phase2_common.yaml \
  --mode bounded_dynamic --seed 101 --steps 1000000
python scripts/compare_phase2_results.py \
  --root results/PZS_phase2/formal_1M \
  --config configs/pzs_phase2_common.yaml
```

The original user CIF and experimental files remain excluded from Git by the
repository privacy rules; their SHA-256 hashes and resolved paths are recorded
in each chain's `input_manifest.json`.
