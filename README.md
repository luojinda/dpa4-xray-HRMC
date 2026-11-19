# DPA4 X-ray HRMC — Phase 1 and Phase 2

The repository contains the Phase 1 CPU X-ray PDF forward model and the Phase 2
Pure-RMC benchmark with transactional single-atom partial-histogram/PDF
updates. Phase 2 contains no DPA4, ML potential, energy, coordination, or
bond-angle term; those remain explicitly outside the current scope.

## Phase 2 Pure RMC

Place the ignored PZS CIF and ZXS_1 `.gr` input in `PDF/`, then run:

```bash
python scripts/validate_incremental.py --config configs/pzs_phase2_common.yaml
python scripts/benchmark_incremental_pdf.py --config configs/pzs_phase2_common.yaml
python scripts/run_phase2_pzs.py --config configs/pzs_phase2_common.yaml \
  --mode bounded_dynamic --seed 101 --steps 1000000
python scripts/compare_phase2_results.py --root results/PZS_phase2
```

See [the incremental derivation](docs/phase2_incremental_derivation.md) and
[the frozen RMC protocol](docs/phase2_rmc_protocol.md) for the exact histogram,
scale, Metropolis, checkpoint, and diagnostic conventions.

## Reproduce the LNOC calculation

The input CIF is intentionally ignored by Git. Put it at `PDF/LNOC_opt.cif`
before running.

Windows PowerShell:

```powershell
cd 'E:\1_CODEX\dpa4-xray-HRMC'
conda env create -f environment.yml
conda activate dpa4-xray-hrmc
python scripts/calculate_xray_pdf.py --config configs/lnoc_phase1.yaml
python -m pytest -q
```

Linux Bash:

```bash
git clone <private-repository-url> dpa4-xray-HRMC
cd dpa4-xray-HRMC
# Copy the licensed input to PDF/LNOC_opt.cif.
conda env create -f environment.yml
conda activate dpa4-xray-hrmc
python scripts/calculate_xray_pdf.py --config configs/lnoc_phase1.yaml
python -m pytest -q
```

For an existing environment, use
`conda env update -f environment.yml --prune`. The editable package and test
dependencies are installed by `environment.yml`; no GPU stack or DPA4-related
dependency is pulled in.

## Default physics and outputs

The supplied run uses Q=1–24 Å⁻¹ (0.05 Å⁻¹ step), output r=0–10 Å (0.01 Å
step), `Biso=0.3 Å²`, no Lorch window, RDF bins of 0.01 Å, and a 40 Å RDF
cutoff. It also evaluates 20/30/40 Å cutoff and 0.01/0.02 Å bin convergence.

Artifacts are written to `results/LNOC/`:

- `LNOC_xray_Gr.dat`, `LNOC_SQ.dat`, and `LNOC_partial_RDF.dat`
- `form_factors.dat` and `pair_weights.dat`
- `LNOC_PDF_1p5_4p5.png` and `LNOC_PDF_1p5_10.png`
- `parameters.json`, `structure.json`, and `validation_report.md`

See [theory_and_conventions.md](docs/theory_and_conventions.md) for exact
normalization and thermal treatment. A finite RDF cutoff applied to a perfect
crystal remains an approximation; the validation report quantifies rather than
hides that sensitivity.
