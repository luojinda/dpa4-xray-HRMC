# PZS new-MQ local audit and four Inline HRMC 100k comparisons

Date: 2026-11-21

## Scope and reproducibility

This report compares `inputs/PZS_new_20261121/PZS_new_20261121.cif` with the original MQ structure and then evaluates four Inline HRMC runs from the new MQ structure. All four runs used seed 101, 100 K, maximum displacement 0.1 Å, 100,000 attempts, the experimental PDF over 1.4–10 Å, and the existing hard-sphere and coordination-number settings. The only changed variables were DPA4 model and PDF scale strategy.

Input SHA-256 values:

- CIF: `3284f7645aff9bf49a308cb15d6e5b96adecea11337fe69aa2af95bac9775c39`
- LoRA model: `64b0fede16c72af54adf45fca8b79fb76e0ed70fd5d594d14debef97eeac75ab`
- Frozen model: `d5e70cba5ba6f228c6818ef70f47594de46538d8750e0113ebe723392e7fb862`

Both models passed Gate 0: complete type map, 6.0 Å cutoff, finite energies/forces, and repeatability at the few-micro-eV/few-micro-eV-per-Å numerical level. The detailed record is in `gate0/model_validation.json`.

## New MQ versus original MQ

Both structures contain 896 atoms with identical composition (Li96 Zr64 Si64 P32 Cl512 O128). Coordination was audited with the established fixed cutoffs: P–O 1.80 Å, Si–O 1.85 Å, Si–Cl 2.30 Å, Zr–O 2.50 Å, and Zr–Cl 3.00 Å.

| Metric | Original MQ | New MQ |
|---|---:|---:|
| Cell volume (Å³) | 20353.759 | 22002.213 |
| Density (g cm⁻³) | 2.4060 | 2.2258 |
| P fourfold | 32/32 (100%) | 29/32 (90.625%) |
| Si fourfold | 64/64 (100%) | 64/64 (100%) |
| Zr total CN5/CN6 | 1/63 | 8/56 |
| Zr–O CN0/CN1/CN2/CN3 | 19/23/18/4 | 18/26/19/1 |
| P angle RMS (deg) | 3.672 | 4.560 |
| Si angle RMS (deg) | 2.949 | 4.088 |
| P fixed-center CSM (%) | 0.359 | 0.538 |
| Si fixed-center CSM (%) | 0.601 | 0.855 |
| P–O mean ± std (Å) | 1.5441 ± 0.0329 | 1.5495 ± 0.0441 |
| Si–O mean ± std (Å) | 1.6512 ± 0.0236 | 1.6512 ± 0.0315 |
| Si–Cl mean ± std (Å) | 2.0247 ± 0.0221 | 2.0328 ± 0.0379 |

The new cell is 8.10% larger and 7.49% less dense. Its first-shell peaks are broader, P/Si tetrahedra are more distorted, three P sites are three-coordinate under the fixed cutoff, and the Zr total-CN distribution contains more CN5 sites. The complete RDF/CN comparison is `mq_audit/partial_rdf_and_CN_comparison.png` and the machine-readable audit is `mq_audit/mq_local_environment_audit.json`.

Before HRMC, the new MQ PDF has no-scale Rw = 0.564273. Its positive least-squares scale is 0.770342, giving Rw = 0.501912. The bounded-dynamic interval is [0.616274, 0.924411].

## Four 100k Inline HRMC results

| Model | Scale strategy | Active Rw | Common no-scale Rw | Final scale | Acceptance | ΔE vs new MQ under same model (meV/atom) |
|---|---|---:|---:|---:|---:|---:|
| LoRA | no_scale | 0.187061 | 0.187061 | 1.000000 | 22.623% | +1.508 |
| LoRA | bounded_dynamic | 0.180705 | 0.355872 | 0.762363 | 21.830% | -5.750 |
| Frozen | no_scale | 0.186998 | 0.186998 | 1.000000 | 22.417% | +0.154 |
| Frozen | bounded_dynamic | **0.178338** | 0.351709 | 0.764480 | 22.047% | **-6.796** |

Each process took about 9,750–9,760 s (2.71 h), or about 10.25 attempts/s per chain. The four processes shared one RTX 5090 concurrently, so these wall times are not single-process model-speed benchmarks. Incremental PDF and coordination drift checks were exactly zero at recorded checkpoints.

The two dynamic runs give the lowest active-strategy Rw and also lower energy than the new MQ under both model cross-checks. Frozen + bounded_dynamic is the best of these four by active Rw and by its own-model energy reduction, although the differences relative to LoRA + bounded_dynamic are small. The no-scale runs preserve absolute amplitude directly and converge to nearly identical Rw values for the two models.

Important scale interpretation: the dynamic structures have common no-scale Rw around 0.35 because their unscaled calculated G(r) amplitude is larger. After applying the optimized scale near 0.763–0.764, their active Rw is 0.178–0.181. Therefore active Rw answers the scale-permitted fitting question; common no-scale Rw answers the absolute-amplitude question. They must not be conflated.

## Coordination and tetrahedral geometry after HRMC

All four final structures have zero hard-sphere violations, retain Si fourfold = 100%, and retain P fourfold = 90.625%. Thus the constraint prevented further P/Si coordination loss but did not repair the three initially three-coordinate P sites within 100k attempts. Zr–O histograms remain exactly 18/26/19/1 for CN0/CN1/CN2/CN3. Zr total CN5/CN6 changes only slightly: 6/58 (LoRA no-scale) and 7/57 for the other three runs.

| Structure | P angle RMS (deg) | Si angle RMS (deg) | P CSM (%) | Si CSM (%) |
|---|---:|---:|---:|---:|
| New MQ | 4.560 | 4.088 | 0.538 | 0.855 |
| LoRA no_scale | **3.590** | 3.692 | **0.332** | 0.762 |
| LoRA bounded_dynamic | 3.732 | **3.379** | 0.352 | 0.754 |
| Frozen no_scale | 3.765 | 3.511 | 0.361 | **0.748** |
| Frozen bounded_dynamic | 3.695 | 3.519 | 0.361 | 0.774 |

All four HRMC structures improve P/Si angular RMS and CSM relative to the new MQ rather than sacrificing tetrahedral geometry for the PDF fit.

## Artifacts

- New/original audit: `mq_audit/`
- Gate 0 validation: `gate0/model_validation.json`
- Unified metrics: `four_inline_100k_comparison/metrics.json`
- PDF curves and figure: `four_inline_100k_comparison/pdf_curves.csv`, `pdf_four_runs_1p4_10A.png`
- Trajectories and figure: `four_inline_100k_comparison/trajectories.csv`, `trajectories_Rw_energy_scale.png`
- Geometry data and figure: `four_inline_100k_comparison/tetrahedral_geometry.csv`, `tetrahedral_geometry.png`
- Final CIF/extxyz, summaries, traces, and accepted/rejected examples: `../PZS_phase3/phase3C/PZS_new_20261121/validation_100k_seed101_*`

One recording caveat: if an exact checkpoint step is rejected by the hard-sphere early-return branch, the current code skips that intermediate trace/checkpoint record. This caused an absent LoRA no-scale 80k trace point, but the process continued normally and the final 100k structure and summary were written and verified.
