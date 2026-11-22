# PZS expanded-new MQ: LoRA dynamic-scale Inline HRMC, 1M attempts

## Run definition

- Initial structure: `inputs/PZS_new_20261121/PZS_new_20261121.cif` (896 atoms)
- Potential: `models/DPA4_PZS_new_LoRA.pt2`
- Algorithm: Inline HRMC
- Attempts / seed / temperature: 1,000,000 / 101 / 100 K
- Maximum displacement: 0.1 Å
- PDF fitting interval: 1.4–10.0 Å
- PDF scale: bounded dynamic, using the same ±20% bounds around the initial optimum as the preceding 100k validation
- Hard-sphere and coordination-number settings: unchanged from Phase 2.5/Phase 3

## Final result

| metric | 100k validation | 1M production |
|---|---:|---:|
| active bounded-dynamic Rw | 0.18070485 | 0.17112163 |
| common no-scale Rw | — | 0.34497951 |
| scale w | 0.7623633 | 0.7668528 |
| accepted / attempted | — | 208,692 / 1,000,000 |
| acceptance ratio | — | 0.208692 |
| hard-sphere rejections | — | 4,852 |
| DPA4 energy (eV/atom) | — | -5.02529160 |
| ΔE from initial MQ (independent cross evaluation, meV/atom) | -5.75025 | -8.85793 |
| wall time | — | 30,400.5 s (8.44 h) |

The production run improved the active Rw by 0.00958 relative to 100k and lowered the independently evaluated LoRA energy by another 3.11 meV/atom. The incremental-energy drift check was `-1.51e-7 eV`, so no material accumulated drift was detected.

## Local geometry

| structure | P angle RMS (deg) | Si angle RMS (deg) | P CSM (%) | Si CSM (%) | P fourfold | Si fourfold |
|---|---:|---:|---:|---:|---:|---:|
| expanded new MQ | 4.5595 | 4.0875 | 0.53779 | 0.85538 | 90.625% | 100% |
| LoRA dynamic 100k | 3.7324 | 3.3791 | 0.35203 | 0.75407 | 90.625% | 100% |
| LoRA dynamic 1M | 3.7578 | 3.2066 | 0.35447 | 0.70531 | 90.625% | 100% |

The run preserved all initially fourfold Si sites and did not recover the three initially under-coordinated P sites. Relative to the expanded new MQ, both tetrahedral angle RMS and CSM improved substantially; the 1M continuation further improved Si geometry while P geometry stayed close to the 100k state.

## Reproducible outputs

- Production structures and raw trace: `results/PZS_phase3/phase3C/PZS_new_20261121/production_1M_seed101_T100_d0p1_r1p4_10p0A_lora_bounded_dynamic/`
- Consolidated metrics and plots: `results/PZS_new_20261121/lora_dynamic_1M/`
- Main figures: `pdf_newMQ_100k_1M.png`, `trajectory_1M.png`
- Analysis script: `scripts/analyze_pzs_new_lora_dynamic_1m.py`
