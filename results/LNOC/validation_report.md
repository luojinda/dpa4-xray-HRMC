# LNOC Phase 1 validation report

## Structure

- ASE formula: `Cl384Li96Nb96O96`; atoms: 672; counts: `{'Cl': 384, 'Li': 96, 'Nb': 96, 'O': 96}`.
- Cell volume: 15659.087192 Å³; number density: 0.04291438 Å⁻³.
- Cell lengths: [23.669561, 26.224598, 25.235737] Å; angles: [90.0, 88.501282, 90.0]°.
- PBC: [True, True, True]; minimum periodic non-self separation: 1.776202 Å.
- Explicit occupancies checked: 672 sites, all equal to 1.0.

## Numerical model

- RDF convention: ordered central-α/neighbour-β counts normalized by `Nα ρ0 cβ ΔV`; 10 α≤β partials.
- Primary RDF: `dr=0.01 Å`, bin centres, `r_cut=40 Å`; explicit image bounds (3, 3, 3) (343 candidate translations).
- Q grid: 1–24 Å⁻¹ in 0.05 Å⁻¹ steps; rectangular Q window (Lorch=False).
- `Biso=0.3 Å²`, hence `Uiso=0.0037995444 Å²`. Distinct-pair partials are multiplied once by `exp[-Q²(Uα+Uβ)/2]`; no qdamp/qbroad.
- Neutral elastic `f0(Q)` uses `xraydb.f0(element, Q/(4π))`; anomalous terms are not included.
- `G(0)` is explicitly set to zero; no smoothing, scale adjustment, background subtraction, or r shift is applied.

## Diagnostics

- Maximum `|sum(w_ab)-1|` over all Q points: 6.661e-16; all stored weights are finite and non-negative.
- Primary S(Q) range: -1.76893 to 7.19524; 24 of 461 samples are negative. The minimum occurs at Q=1.85 Å⁻¹.
- Negative S(Q) samples are finite-r-cut termination ringing around sharp ideal-crystal Bragg features, not a physical claim of negative scattering intensity. A smooth real-space window was deliberately not applied to the default result.
- Tail shell-volume-weighted g(r) means over the last 5 Å: `{'Cl-Cl': 1.0096997961917462, 'Cl-Li': 0.991266207945434, 'Li-Li': 0.9991004829501166, 'Cl-Nb': 0.9806668947038045, 'Li-Nb': 1.0083172770732727, 'Nb-Nb': 0.9843536123530668, 'Cl-O': 0.998178803537801, 'Li-O': 1.0083172770732727, 'Nb-O': 0.9917270476515918, 'O-O': 0.9843536123530668}`.
- Prominent peaks in 1.5–4.5 Å (r, G): `[(1.77, 0.065), (2.39, 3.3184), (3.1, -0.3684), (3.38, 0.3251), (3.94, 2.9316), (4.41, 0.3602)]`.
- Prominent peaks in 1.5–10 Å (r, G): `[(2.39, 3.3184), (3.94, 2.9316), (4.41, 0.3602), (4.85, 1.7123), (5.29, 0.8284), (6.83, 6.1867), (7.93, 3.0632), (8.86, 6.4092)]`.

### Convergence differences

Each entry is measured against the primary 40 Å / 0.01 Å result. Finite-cutoff crystal PDFs need not converge monotonically because an ideal periodic crystal has long-range order.

| variant/range | RMS (Å⁻²) | relative RMS | max abs (Å⁻²) |
|---|---:|---:|---:|
| rcut_20, 1.5-4.5 A | 0.00555499 | 0.00461451 | 0.00954286 |
| rcut_20, 1.5-10 A | 0.00778311 | 0.00427809 | 0.0167074 |
| rcut_30, 1.5-4.5 A | 0.00483062 | 0.00401277 | 0.00819245 |
| rcut_30, 1.5-10 A | 0.00509103 | 0.00279835 | 0.00981703 |
| dr_0.02, 1.5-4.5 A | 0.0573906 | 0.0476741 | 0.170514 |
| dr_0.02, 1.5-10 A | 0.0512094 | 0.0281479 | 0.170514 |

## Performance and interpretation

- Wall time: 1.687 s; sampled peak process RSS: 155.8 MiB; platform: `Windows-10-10.0.22631-SP0`.
- This is a finite-real-space-cutoff approximation to an ideal periodic crystal, not a claim of an exact infinite-crystal PDF.
- The ideal optimized CIF lacks nanocrystal size effects, static/local disorder, and experimental qdamp/qbroad. It should not be expected to reproduce every experimental peak width or medium-range amplitude.
- No experimental curve was available, so no Rwp/R² or agreement claim is reported.
