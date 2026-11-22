# PZS_quench_new: three-MQ audit and four Inline-HRMC 100k comparisons

## Scope and common settings

`PZS_quench_new.cif` was compared with the original MQ and the expanded new MQ. Four independent 100k Inline HRMC runs then used `PZS_quench_new.cif` as the initial structure:

1. LoRA potential + no scale
2. LoRA potential + bounded dynamic scale
3. frozen Neo-OMat potential + no scale
4. frozen Neo-OMat potential + bounded dynamic scale

All four runs used seed 101, 100 K, maximum displacement 0.1 Å, PDF range 1.4–10.0 Å, and the unchanged Phase 2.5 hard-sphere and coordination-number constraints.

## Initial-structure audit

| metric | original MQ | expanded new MQ | quench new MQ |
|---|---:|---:|---:|
| cell volume (Å³) | 20353.759 | 22002.213 | 20353.759 |
| density (g/cm³) | 2.40603 | 2.22576 | 2.40603 |
| P CN3 / CN4 | 0 / 32 | 3 / 29 | 3 / 29 |
| Si CN4 | 64 | 64 | 64 |
| Zr total CN5 / CN6 | 1 / 63 | 8 / 56 | 7 / 57 |
| Zr–O CN0 / CN1 / CN2 / CN3 | 19 / 23 / 18 / 4 | 18 / 26 / 19 / 1 | 18 / 25 / 20 / 1 |
| P angle RMS (deg) | 3.6721 | 4.5595 | 3.8806 |
| Si angle RMS (deg) | 2.9490 | 4.0875 | 3.0693 |
| P CSM (%) | 0.35886 | 0.53779 | 0.38449 |
| Si CSM (%) | 0.60128 | 0.85538 | 0.64062 |

The quench-new structure restores the original cell volume and density and has tetrahedral geometry much closer to the original MQ than the expanded new MQ. It nevertheless retains the same three under-coordinated P sites as the expanded structure and has seven Zr CN5 sites rather than one in the original MQ.

For quench-new, the first-shell statistics are P–O `1.54703 ± 0.03297 Å`, Si–O `1.65989 ± 0.02760 Å`, and Si–Cl `2.02112 ± 0.02610 Å`.

## Four 100k Inline-HRMC results

| potential / scale | active Rw | common no-scale Rw | final w | acceptance | energy (eV/atom) | wall (s) |
|---|---:|---:|---:|---:|---:|---:|
| LoRA / no scale | 0.22468987 | 0.22468987 | 1.000000 | 0.21538 | -5.00161473 | 9773.90 |
| LoRA / bounded dynamic | 0.21793504 | 0.42871458 | 0.725541 | 0.20668 | -5.01312840 | 9869.87 |
| frozen / no scale | 0.22453792 | 0.22453792 | 1.000000 | 0.21455 | -4.86713131 | 9765.47 |
| frozen / bounded dynamic | 0.21247387 | 0.42041151 | 0.729264 | 0.20713 | -4.87658492 | 9864.43 |

The initial quench-new MQ has active Rw 0.503307 at its initial optimum scale `w=0.632400` and common no-scale Rw 0.711062. The bounded interval used by both dynamic runs is `[0.505920, 0.758880]`.

The active-strategy values show that dynamic scaling gives the lowest fitted residual, with the frozen/dynamic run lowest at 0.21247. However, on the common `w=1` diagnostic the dynamic structures have Rw 0.420–0.429, whereas the no-scale structures have Rw ≈0.2246. Thus the active Rw values across scale strategies are not interchangeable: the dynamic result mainly exploits amplitude adjustment, and no-scale remains the stronger result if the absolute experimental normalization is trusted.

## Energy interpretation

Independent evaluations relative to the initial quench-new MQ are:

| final structure | ΔE with LoRA (meV/atom) | ΔE with frozen model (meV/atom) |
|---|---:|---:|
| LoRA / no scale | +31.5306 | +29.6614 |
| LoRA / dynamic | +20.0169 | +19.0124 |
| frozen / no scale | +31.5813 | +29.3757 |
| frozen / dynamic | +21.4969 | +19.9220 |

All four HRMC endpoints are higher in potential energy than the quench-new initial MQ. Dynamic scaling reduces this penalty by roughly 10–12 meV/atom relative to no-scale, but does not make ΔE negative. This differs from the expanded-new MQ case, for which LoRA/dynamic HRMC lowered the energy.

## Tetrahedral geometry and coordination retention

| structure | P angle RMS (deg) | Si angle RMS (deg) | P CSM (%) | Si CSM (%) |
|---|---:|---:|---:|---:|
| quench new MQ | 3.8806 | 3.0693 | 0.38449 | 0.64062 |
| LoRA / no scale | 3.9451 | 4.0931 | 0.40513 | 0.84059 |
| LoRA / dynamic | 3.7352 | 3.6221 | 0.38291 | 0.78699 |
| frozen / no scale | 3.9562 | 3.7813 | 0.40550 | 0.78807 |
| frozen / dynamic | 3.8419 | 3.6424 | 0.37125 | 0.79301 |

All four endpoints preserve P fourfold fraction at 90.625% and Si fourfold fraction at 100%. Their final Zr total distribution is CN5/CN6 = 4/60, while Zr–O CN0/CN1/CN2/CN3 remains 18/25/20/1. No final hard-sphere violation was found. Dynamic scaling better controls P tetrahedral distortion than no-scale, but all four endpoints increase Si distortion relative to the initial quench-new MQ.

## Reproducible outputs

- Three-MQ audit: `results/PZS_quench_new/three_mq_audit/`
- Four-run consolidated metrics/figures: `results/PZS_quench_new/four_inline_100k_comparison/`
- Raw run directories and final CIF/extxyz files: `results/PZS_phase3/phase3C/PZS_quench_new/`
- Audit script: `scripts/audit_pzs_three_mq.py`
- Four-run analysis script: `scripts/analyze_pzs_quench_new_four_inline_100k.py`
