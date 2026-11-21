# Inline HRMC scale strategy comparison (1.4--10 A)

Both runs used the original MQ structure, seed 101, 100 K, maximum Cartesian displacement 0.1 A, 100,000 attempts, the same DPA4 model, hard-sphere model, and coordination penalties. Only the PDF scale strategy changed.

The bounded-dynamic definition is the existing project definition: `0.8*w0 <= w <= 1.2*w0` (a total range width of 40%, or +/-20% around the initial scale).

| Metric | Fixed initial | Bounded dynamic |
|---|---:|---:|
| Active Rw (1.4--10 A) | 0.231137 | **0.217952** |
| Rw with no scale (common diagnostic) | 0.559103 | **0.446798** |
| Final scale w | 0.633865 | 0.714470 |
| Observed scale range | 0.633865--0.633865 | 0.633865--0.720899 |
| Acceptance rate | 19.922% | 20.855% |
| Delta E (meV/atom from MQ) | **17.316** | 19.535 |
| Wall time (s) | 3110.43 | 3092.87 |
| Hard-sphere rejected | 2,987 | 3,656 |
| P fourfold / Si fourfold | 100% / 100% | 100% / 100% |
| Hard-sphere violations in final structure | 0 | 0 |

Initial scale `w0 = 0.6338649`; allowed dynamic bounds were `[0.5070919, 0.7606379]`. The dynamic scale never touched either bound, so its result was not controlled by clipping.

## Tetrahedral geometry

| Structure | P angle RMS (deg) | Si angle RMS (deg) | P CSM (%) | Si CSM (%) |
|---|---:|---:|---:|---:|
| MQ | 3.6721 | **2.9490** | 0.3589 | **0.6013** |
| Fixed initial | **3.4191** | 3.7659 | **0.3078** | 0.7540 |
| Bounded dynamic | 3.5400 | 3.9384 | 0.3287 | 0.7692 |

Both final structures retain the exact target P/Si coordination and the original Zr total/Zr-O coordination histograms. Dynamic scaling improves the PDF most, at a cost of 2.22 meV/atom relative to the fixed-scale final structure and a modest increase in Si tetrahedral distortion.

## Reproducible outputs

- `metrics.json`: cross-evaluated residuals, scale values, energies, and geometry.
- `pdf_curves.csv` and `pdf_comparison_1p4_10A.png`: strategy-specific scaled PDF curves.
- `trajectories.csv` and `trajectory_Rw_energy_scale.png`: active Rw, energy, and scale evolution.
- `tetrahedral_geometry.csv` and `tetrahedral_geometry.png`: MQ/final local geometry.

One output-cadence issue was observed: when an exact trace/checkpoint step is rejected by hard sphere, the current loop's early `continue` skips that trace/checkpoint record. This affected the dynamic run's 80k record but did not alter the Markov chain, final 100k state, or final checkpoint.
