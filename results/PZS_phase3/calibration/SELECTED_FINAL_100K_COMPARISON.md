# Selected final 100k structure comparison

The 100k runners did not retain historical best-coordinate snapshots, so this
comparison uses final structures, as requested.

- Alternating: seed 303 final structure.  It was selected over seed 101 because
  its DPA4 energy, force RMS and local geometry were substantially better,
  despite a slightly higher final Rw.
- Inline: seed 303 final structure.  It had the lowest final Rw, energy increase
  and force RMS among the three Inline 100 K chains.

The dual-axis trajectory figure uses the original MQ energy as zero and converts
total energy differences as `1000 * (E - E_MQ) / 896` meV/atom.  Alternating
points are recorded after FIRE blocks plus the final 100k structure; Inline
points are recorded every 100 attempts.  The different point densities reflect
the real logging schedules and are not interpolated into equivalent samples.

| Structure | P angle RMS (deg) | Si angle RMS (deg) | P CSM (%) | Si CSM (%) |
|---|---:|---:|---:|---:|
| Original MQ | 3.672 | 2.949 | 0.359 | 0.601 |
| Alternating seed303 final | 9.863 | 10.732 | 2.264 | 3.078 |
| Inline 100 K seed303 final | 3.822 | 4.021 | 0.375 | 0.737 |

The Inline final structure remains close to MQ tetrahedral quality, while the
Alternating final structure is substantially more distorted despite its lower
PDF residual.

Outputs:

- `plots/selected_final_100k_Rw_deltaE_vs_attempts.png`
- `plots/selected_final_100k_tetrahedral_geometry.png`
- `selected_final_100k_trajectories.csv`
- `selected_final_100k_geometry.csv`
