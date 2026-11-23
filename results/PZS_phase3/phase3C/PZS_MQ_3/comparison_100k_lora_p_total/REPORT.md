# PZS MQ3: two Inline 100k HRMC runs

Both runs use the LoRA DPA4 model, seed 101, T=100 K, d=0.1 A, PDF range 1.4-10 A, and the P-total coordination constraint.

## PDF and energy

| Structure | Active Rw | No-scale Rw | Scale | Delta E (meV/atom) | Acceptance |
|---|---:|---:|---:|---:|---:|
| No scale 100k | 0.17590 | 0.17590 | 1.00000 | -3.721 | 0.2245 |
| Dynamic 100k | 0.15832 | 0.35657 | 0.75553 | -10.301 | 0.2194 |

The bounded-dynamic active Rw includes its fitted scale. Its common no-scale Rw is reported separately and must be used for an unscaled amplitude comparison.

## Local structure

P tetrahedra include both P-O (<1.80 A) and P-Cl (<2.30 A), so the initial P(O3Cl) unit is counted as four-coordinate.

| Structure | P 4-fold | Si 4-fold | P angle RMS | Si angle RMS | P CSM | Si CSM | Zr CN | Zr-O CN | HS violations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Initial MQ3 | 100.00% | 100.00% | 4.9210 | 4.0459 | 0.6905 | 0.7451 | 5.9531 | 1.0469 | 0 |
| No scale 100k | 100.00% | 100.00% | 4.0950 | 3.6065 | 0.4664 | 0.6853 | 5.9688 | 1.0469 | 0 |
| Dynamic 100k | 100.00% | 100.00% | 4.0813 | 3.4652 | 0.4649 | 0.6881 | 5.9844 | 1.0625 | 0 |

Detailed ligand patterns, bond statistics, coordination histograms, and hard-sphere margins are stored in metrics.json and the CSV files.

All comparisons are for one seed (101); they are not three-seed averages.
