# PZS calculated/experimental X-ray PDF comparison

## Inputs and matched settings

- Structure: `PZS_quench_222.cif`, P1, 896 atoms, formula `Cl512Li96O128P32Si64Zr64`.
- The structure composition reduces exactly to the experimental header composition `Li3PO4Zr2Si2Cl16`.
- Experiment: wavelength 0.1243 Å, `Qmin=0.9 Å^-1`, `Qmax=16.0056 Å^-1`, PDFGetX/xPDFsuite `G(r)` output.
- Calculation: `Q=0.9–16.0 Å^-1`, `Biso=0.3 Å^2`, no Lorch window, no qdamp/qbroad, partial-RDF cutoff 40 Å.
- The 0.0056 Å^-1 difference in Qmax is negligible compared with the 0.05 Å^-1 model grid.

## Intensity and shape comparison

| Range | Experimental RMS | Raw calculated RMS | Raw RMS ratio | Pearson r | Raw RMSE | Aligned RMSE |
|---|---:|---:|---:|---:|---:|---:|
| 1–5 Å | 0.44497 | 0.70624 | 1.5872 | 0.8903 | 0.37039 | 0.20269 |
| 1–10 Å | 0.31052 | 0.49131 | 1.5822 | 0.8749 | 0.26618 | 0.15037 |

The diagnostic 1–10 Å least-squares alignment is

`G_aligned(r) = 0.5530200 G_calculated(r) + 0.0012367 Å^-2`.

This alignment is only a diagnostic and was not applied to the raw forward result. The nearly zero offset indicates no major constant-baseline mismatch, while the scale below one confirms that the calculated oscillations are too strong overall.

The difference is not purely a single normalization factor. The calculated 2.47 Å peak (~1.73 Å^-2) is close to the experimental 2.49 Å peak (~1.59 Å^-2), but the calculated 3.43 Å peak (~1.73 Å^-2) is much stronger than the experimental ~3.47 Å peak (~0.73 Å^-2). A global scale therefore makes the first peak too small while improving the second.

## Interpretation

- The peak positions and overall curve shape agree reasonably well (`r=0.875–0.890`), supporting the broad local topology of the model.
- The intensity discrepancy is substantial, especially for the second major shell, and should not be corrected by silently renormalizing the forward model before HRMC.
- Likely contributors are real structural population/disorder differences, the single idealized configuration, the assumed uniform `Biso`, and omitted experimental resolution effects (`qdamp`/`qbroad`). These effects can change peak widths and heights, but the unequal first/second-peak error shows that a structural mismatch remains.
- For later HRMC, keep a global scale factor (and at most a small baseline term) as an explicit nuisance parameter, while using the structural moves to correct relative peak intensities. Instrumental damping/broadening should be calibrated independently rather than absorbed into atom moves.
