# PZS Phase 1 validation report

## Structure

- ASE formula: `Cl512Li96O128P32Si64Zr64`; atoms: 896; counts: `{'Cl': 512, 'Li': 96, 'O': 128, 'P': 32, 'Si': 64, 'Zr': 64}`.
- Cell volume: 20353.759010 Å³; number density: 0.04402135 Å⁻³.
- Cell lengths: [29.379999, 24.742001, 28.0] Å; angles: [90.0, 90.0, 90.0]°.
- PBC: [True, True, True]; minimum periodic non-self separation: 1.469529 Å.
- Explicit occupancies checked: 896 sites, all equal to 1.0.

## Numerical model

- RDF convention: ordered central-α/neighbour-β counts normalized by `Nα ρ0 cβ ΔV`; 21 α≤β partials.
- Primary RDF: `dr=0.01 Å`, bin centres, `r_cut=40 Å`; explicit image bounds (3, 3, 3) (343 candidate translations).
- Q grid: 0.9–16 Å⁻¹ in 0.05 Å⁻¹ steps; rectangular Q window (Lorch=False).
- `Biso=0.3 Å²`, hence `Uiso=0.0037995444 Å²`. Distinct-pair partials are multiplied once by `exp[-Q²(Uα+Uβ)/2]`; no qdamp/qbroad.
- Neutral elastic `f0(Q)` uses `xraydb.f0(element, Q/(4π))`; anomalous terms are not included.
- `G(0)` is explicitly set to zero; no smoothing, scale adjustment, background subtraction, or r shift is applied.

## Diagnostics

- Maximum `|sum(w_ab)-1|` over all Q points: 8.882e-16; all stored weights are finite and non-negative.
- Primary S(Q) range: -0.00111421 to 1.52584; 1 of 303 samples are negative. The minimum occurs at Q=0.95 Å⁻¹.
- Negative S(Q) samples are finite-r-cut termination ringing around sharp ideal-crystal Bragg features, not a physical claim of negative scattering intensity. A smooth real-space window was deliberately not applied to the default result.
- Tail shell-volume-weighted g(r) means over the last 5 Å: `{'Cl-Cl': 0.9999789870444057, 'Cl-Li': 0.9979228434099794, 'Li-Li': 1.0004327230555303, 'Cl-O': 0.9961480667992113, 'Li-O': 0.9975999440397936, 'O-O': 1.028364298526779, 'Cl-P': 0.9963411044661703, 'Li-P': 0.9922463660761319, 'O-P': 1.0338395487168872, 'P-P': 1.031930230701875, 'Cl-Si': 1.0006651482060507, 'Li-Si': 0.9987230722839183, 'O-Si': 0.9953162499434064, 'P-Si': 0.9938561832260441, 'Si-Si': 0.9947546858213441, 'Cl-Zr': 0.9997034696470188, 'Li-Zr': 0.9989851355408808, 'O-Zr': 1.01019769917806, 'P-Zr': 1.015083307040003, 'Si-Zr': 0.9984048526147496, 'Zr-Zr': 0.9944177473481064}`.
- Prominent peaks in 1.5–4.5 Å (r, G): `[(2.01, 0.1836), (2.47, 1.7284), (3.43, 1.7301)]`.
- Prominent peaks in 1.5–10 Å (r, G): `[(2.01, 0.1836), (2.47, 1.7284), (3.43, 1.7301), (4.42, -0.1196), (4.92, -0.1376), (5.81, 0.3548), (6.89, 0.2162), (9.05, 0.1966)]`.

### Convergence differences

Each entry is measured against the primary 40 Å / 0.01 Å result. Finite-cutoff crystal PDFs need not converge monotonically because an ideal periodic crystal has long-range order.

| variant/range | RMS (Å⁻²) | relative RMS | max abs (Å⁻²) |
|---|---:|---:|---:|
| rcut_20, 1-5 A | 0.00159781 | 0.00226241 | 0.00315007 |
| rcut_20, 1-10 A | 0.00170764 | 0.00347569 | 0.00348369 |
| rcut_30, 1-5 A | 0.000451361 | 0.0006391 | 0.000814709 |
| rcut_30, 1-10 A | 0.000461548 | 0.000939423 | 0.000875977 |
| dr_0.02, 1-5 A | 0.00336494 | 0.00476455 | 0.00841389 |
| dr_0.02, 1-10 A | 0.00292434 | 0.00595213 | 0.00841389 |

## Performance and interpretation

- Wall time: 2.493 s; sampled peak process RSS: 175.7 MiB; platform: `Windows-10-10.0.22631-SP0`.
- This is a finite-real-space-cutoff approximation to an ideal periodic crystal, not a claim of an exact infinite-crystal PDF.
- The ideal optimized CIF lacks nanocrystal size effects, static/local disorder, and experimental qdamp/qbroad. It should not be expected to reproduce every experimental peak width or medium-range amplitude.
- Experimental agreement metrics are outside this forward-calculation report; use the dedicated comparison output when experimental data are supplied.
