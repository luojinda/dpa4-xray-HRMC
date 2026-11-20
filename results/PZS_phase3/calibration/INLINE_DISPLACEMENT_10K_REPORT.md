# Inline HRMC displacement scan

Common wall time: 210.687 s.

| d (A) | final Rw | best Rw (attempt) | wall-matched Rw | dE (meV/atom) | acceptance | HS reject | P angle RMS | Si angle RMS | P CSM | Si CSM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.24412 | 0.24385 (9900) | 0.26748 | 50.63 | 31.71% | 5.69% | 3.963 | 6.111 | 0.415 | 1.214 |
| 0.2 | 0.22714 | 0.22714 (10000) | 0.23498 | 58.91 | 16.39% | 20.28% | 4.176 | 6.191 | 0.464 | 1.251 |
| 0.3 | 0.24134 | 0.24030 (9900) | 0.24134 | 63.78 | 10.51% | 35.41% | 3.864 | 6.714 | 0.396 | 1.430 |

## Recommendation

d=0.2 A is preferred: it improves final and wall-matched Rw relative to d=0.1 A, with modest energy and geometry penalties. d=0.3 A has lower acceptance, more hard-sphere rejection, higher energy, and worse Si distortion without matching the d=0.2 A PDF improvement.
