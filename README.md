# DPA4 X-ray HRMC — Phase 1

This repository currently contains only the CPU forward model requested for
Phase 1: periodic CIF → partial RDFs → Q-dependent neutral-atom X-ray weights →
`S(Q)`, `F(Q)`, and reduced `G(r)`. It does **not** contain DPA4, RMC/HRMC, or
incremental PDF updates.

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
