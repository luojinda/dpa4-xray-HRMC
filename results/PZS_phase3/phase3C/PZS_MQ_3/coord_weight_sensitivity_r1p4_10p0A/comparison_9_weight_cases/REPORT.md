# MQ3 coordination-weight sensitivity: 9 dynamic-scale Inline 100k cases

All runs use MQ3, seed 101, T=100 K, d=0.1 A, 1.4-10 A, bounded dynamic scale, and the LoRA DPA4 potential.
The baseline weights are P=800, Si=2500, Zr-total=1.0, and Zr-O=0.2. Each sensitivity run changes one weight only.

## PDF, energy, and sampling

| Rank | Case | Active Rw | Rw 1.4-1.8 | Rw 1.8-2.2 | Scale | Delta E (meV/atom) | Acceptance |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Si x10 | 0.15598 | 0.40001 | 0.55092 | 0.76451 | -10.197 | 0.2182 |
| 2 | Si x0.1 | 0.15807 | 0.39408 | 0.55428 | 0.76479 | -10.310 | 0.2169 |
| 3 | Baseline | 0.15832 | 0.37684 | 0.55281 | 0.75553 | -10.301 | 0.2194 |
| 4 | P x10 | 0.15856 | 0.37832 | 0.54983 | 0.75473 | -10.659 | 0.2166 |
| 5 | Zr-O x0.1 | 0.15925 | 0.39694 | 0.56121 | 0.76519 | -10.328 | 0.2187 |
| 6 | Zr-total x0.1 | 0.15959 | 0.38967 | 0.55469 | 0.76273 | -10.231 | 0.2185 |
| 7 | Zr-total x10 | 0.16018 | 0.39241 | 0.55283 | 0.76407 | -11.013 | 0.2187 |
| 8 | Zr-O x10 | 0.16076 | 0.38521 | 0.55458 | 0.75832 | -10.527 | 0.2186 |
| 9 | P x0.1 | 0.16431 | 0.39144 | 0.57999 | 0.76356 | -11.965 | 0.2183 |

## Interpretation

- The lowest final active Rw is Si x10 (0.15598), a 1.47% relative improvement over the baseline. Its improvement comes mainly from 2.2-10 A; it does not improve the 1.4-1.8 A first-neighbor segment.
- The baseline retains the lowest 1.4-1.8 A segment Rw. P x10 gives the lowest 1.8-2.2 A segment Rw, but its total Rw is essentially tied with the baseline.
- P x0.1 is the only clearly adverse perturbation: both the total Rw and the 1.8-2.2 A segment deteriorate. P x10 remains close to baseline while slightly improving P geometry.
- Zr-total and Zr-O decade changes have small effects on final Rw, energy, acceptance, and coordination statistics. Their present baseline weights are not the dominant fitting bottleneck.
- All nine runs preserve 100% P/Si total fourfold coordination, have zero hard-sphere violations, and finish below the initial MQ3 DPA4 energy. Weight changes therefore alter the accepted pathway more than the final coordination counts.
- This is a one-seed sensitivity screen. Differences below roughly 0.002 in Rw should be treated as provisional until repeated with multiple seeds.

## Local structure

| Case | P 4-fold | Si 4-fold | P angle RMS | Si angle RMS | P CSM | Si CSM | Zr CN | Zr-O CN | HS violations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 100.00% | 100.00% | 4.0813 | 3.4652 | 0.4649 | 0.6881 | 5.9844 | 1.0625 | 0 |
| P x0.1 | 100.00% | 100.00% | 4.0805 | 3.3433 | 0.4583 | 0.6653 | 5.9844 | 1.0625 | 0 |
| P x10 | 100.00% | 100.00% | 3.8533 | 3.4981 | 0.4085 | 0.6755 | 5.9844 | 1.0625 | 0 |
| Si x0.1 | 100.00% | 100.00% | 3.8485 | 3.4308 | 0.4202 | 0.6656 | 5.9688 | 1.0469 | 0 |
| Si x10 | 100.00% | 100.00% | 3.9566 | 3.4594 | 0.4326 | 0.6512 | 5.9844 | 1.0625 | 0 |
| Zr-total x0.1 | 100.00% | 100.00% | 4.1497 | 3.2778 | 0.4696 | 0.6551 | 5.9844 | 1.0625 | 0 |
| Zr-total x10 | 100.00% | 100.00% | 3.7398 | 3.3867 | 0.4064 | 0.6631 | 5.9688 | 1.0469 | 0 |
| Zr-O x0.1 | 100.00% | 100.00% | 3.9542 | 3.2162 | 0.4460 | 0.6419 | 5.9844 | 1.0625 | 0 |
| Zr-O x10 | 100.00% | 100.00% | 3.9484 | 3.3138 | 0.4424 | 0.6545 | 5.9844 | 1.0625 | 0 |

Initial MQ3 reference and complete ligand-pattern, bond-length, coordination, and hard-sphere data are stored in metrics.json.
All results are one-seed (101) sensitivity comparisons, not uncertainty estimates across seeds.
