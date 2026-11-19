#!/usr/bin/env python
"""Compare a calculated G(r) table with an experimental PDFGetX-style .gr file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dpa4_xray_hrmc.xray_pdf.experimental import (
    compare_pdfs,
    load_calculated_gr,
    plot_comparison,
    range_metrics,
    read_pdfgetx_gr,
    write_comparison_table,
    write_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calculated", required=True, help="Calculated two-column r, G(r) table")
    parser.add_argument("--experimental", required=True, help="Experimental PDFGetX/xPDFsuite .gr file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample", default="sample")
    parser.add_argument("--parameters", help="Optional parameters.json from the calculated run")
    parser.add_argument("--fit-range", nargs=2, type=float, default=(1.0, 10.0), metavar=("RMIN", "RMAX"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment = read_pdfgetx_gr(args.experimental)
    calculated_r, calculated_g = load_calculated_gr(args.calculated)
    fit_range = tuple(args.fit_range)
    ranges = ((1.0, 5.0), (1.0, 10.0))
    comparison = compare_pdfs(experiment, calculated_r, calculated_g, fit_range)
    metrics = range_metrics(comparison, ranges)
    metrics["files"] = {
        "experimental": str(experiment.path),
        "calculated": str(Path(args.calculated).resolve()),
    }
    metrics["experimental_metadata"] = experiment.metadata

    parameters = {}
    if args.parameters:
        parameters = json.loads(Path(args.parameters).read_text(encoding="utf-8"))
    experiment_q = None
    if "qmin" in experiment.metadata and "qmax" in experiment.metadata:
        experiment_q = (float(experiment.metadata["qmin"]), float(experiment.metadata["qmax"]))
    model_q = None
    if "qmin_Ainv" in parameters and "qmax_Ainv" in parameters:
        model_q = (float(parameters["qmin_Ainv"]), float(parameters["qmax_Ainv"]))
    biso = float(parameters["biso_A2"]) if "biso_A2" in parameters else None

    stem = args.sample
    write_comparison_table(output_dir / f"{stem}_experiment_comparison.dat", comparison)
    write_metrics(output_dir / f"{stem}_experiment_comparison_metrics.json", metrics)
    for lo, hi in ranges:
        plot_comparison(
            comparison,
            output_dir / f"{stem}_experiment_comparison_{lo:g}_{hi:g}.png",
            (lo, hi),
            stem,
            experiment_q,
            model_q,
            biso,
        )
    print(f"Scale: {comparison.scale:.10g}; offset: {comparison.offset_A2:.10g} A^-2")
    for name, values in metrics["ranges"].items():
        print(
            f"{name}: raw RMS ratio={values['raw_rms_ratio_calc_over_exp']:.6g}, "
            f"Pearson r={values['pearson_r']:.6g}, aligned RMSE={values['aligned_rmse_A-2']:.6g} A^-2"
        )
    print(f"Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
