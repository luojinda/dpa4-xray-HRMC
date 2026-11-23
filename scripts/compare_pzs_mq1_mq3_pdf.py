"""Compare calculated X-ray PDFs of PZS MQ1 and MQ3 over 1.4--10 A."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

from dpa4_xray_hrmc.rmc.config import load_phase2_config
from dpa4_xray_hrmc.rmc.runner import inclusive_grid
from dpa4_xray_hrmc.xray_pdf.experimental import read_pdfgetx_gr
from dpa4_xray_hrmc.xray_pdf.form_factors import q_dependent_weights
from dpa4_xray_hrmc.xray_pdf.partial_rdf import compute_partial_rdfs
from dpa4_xray_hrmc.xray_pdf.transform import forward_transform


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/PZS_MQ_3/pdf_mq1_mq3_1p4_10A"
OUT.mkdir(parents=True, exist_ok=True)

PATHS = {
    "PZS MQ1": ROOT / "PDF/PZS_quench_222.cif",
    "PZS MQ3": ROOT / "inputs/PZS_MQ_3/PZS_equilibrated.cif",
}
COLORS = {"PZS MQ1": "#377EB8", "PZS MQ3": "#E41A1C"}

cfg = replace(
    load_phase2_config(ROOT / "configs/pzs_phase2_common.yaml"),
    fit_rmin_A=1.4,
    fit_rmax_A=10.0,
)
experiment = read_pdfgetx_gr(ROOT / cfg.experimental_gr)
fit = (experiment.r_A >= cfg.fit_rmin_A) & (experiment.r_A <= cfg.fit_rmax_A)
q_grid = inclusive_grid(cfg.qmin_Ainv, cfg.qmax_Ainv, cfg.qstep_Ainv)


def calculate_curve(path: Path) -> np.ndarray:
    atoms = read(path)
    rdf = compute_partial_rdfs(atoms, cfg.rdf_rcut_A, cfg.rdf_dr_A)
    weights = q_dependent_weights(rdf.element_counts, q_grid)
    return forward_transform(
        rdf,
        weights,
        experiment.r_A,
        cfg.biso_A2,
        cfg.lorch_mod,
    ).g_r_A2


def rw(calc: np.ndarray, obs: np.ndarray) -> float:
    return float(np.sqrt(np.sum((calc - obs) ** 2) / np.sum(obs**2)))


curves = {name: calculate_curve(path) for name, path in PATHS.items()}
obs_fit = experiment.g_r_A2[fit]

metrics: dict[str, object] = {
    "fit_range_A": [cfg.fit_rmin_A, cfg.fit_rmax_A],
    "experimental_gr": cfg.experimental_gr,
    "q_range_Ainv": [cfg.qmin_Ainv, cfg.qmax_Ainv],
    "q_step_Ainv": cfg.qstep_Ainv,
    "biso_A2": cfg.biso_A2,
    "lorch_modification": cfg.lorch_mod,
    "structures": {},
}

scales: dict[str, float] = {}
for name, curve in curves.items():
    calc_fit = curve[fit]
    scale = float(np.dot(calc_fit, obs_fit) / np.dot(calc_fit, calc_fit))
    scales[name] = scale
    metrics["structures"][name] = {
        "path": str(PATHS[name].relative_to(ROOT)),
        "Rw_no_scale": rw(calc_fit, obs_fit),
        "scale_independent_optimum": scale,
        "Rw_independent_optimum_scale": rw(scale * calc_fit, obs_fit),
    }

mq1_fit = curves["PZS MQ1"][fit]
mq3_fit = curves["PZS MQ3"][fit]
metrics["curve_difference"] = {
    "RMS_G_A-2": float(np.sqrt(np.mean((mq3_fit - mq1_fit) ** 2))),
    "relative_L2_to_MQ1": float(
        np.linalg.norm(mq3_fit - mq1_fit) / np.linalg.norm(mq1_fit)
    ),
    "pearson_r_no_scale": float(np.corrcoef(mq1_fit, mq3_fit)[0, 1]),
}

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

with (OUT / "pdf_curves.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "r_A",
            "G_experiment",
            "G_MQ1_no_scale",
            "G_MQ3_no_scale",
            "G_MQ1_independent_optimum_scale",
            "G_MQ3_independent_optimum_scale",
        ]
    )
    writer.writerows(
        zip(
            experiment.r_A,
            experiment.g_r_A2,
            curves["PZS MQ1"],
            curves["PZS MQ3"],
            scales["PZS MQ1"] * curves["PZS MQ1"],
            scales["PZS MQ3"] * curves["PZS MQ3"],
            strict=True,
        )
    )

fig, axes = plt.subplots(2, 1, figsize=(11, 9), dpi=300, sharex=True)

axes[0].plot(
    experiment.r_A,
    experiment.g_r_A2,
    color="black",
    linewidth=2.0,
    label="Experiment",
)
for name in PATHS:
    value = metrics["structures"][name]["Rw_no_scale"]
    axes[0].plot(
        experiment.r_A,
        curves[name],
        color=COLORS[name],
        linewidth=1.8,
        label=rf"{name} ($R_w={value:.3f}$, $w=1$)",
    )
axes[0].set(
    ylabel=r"$G(r)$ ($\AA^{-2}$)",
    title="Common no-scale comparison",
)

axes[1].plot(
    experiment.r_A,
    experiment.g_r_A2,
    color="black",
    linewidth=2.0,
    label="Experiment",
)
for name in PATHS:
    value = metrics["structures"][name]["Rw_independent_optimum_scale"]
    scale = scales[name]
    axes[1].plot(
        experiment.r_A,
        scale * curves[name],
        color=COLORS[name],
        linewidth=1.8,
        label=rf"{name} ($R_w={value:.3f}$, $w={scale:.3f}$)",
    )
axes[1].set(
    xlim=(cfg.fit_rmin_A, cfg.fit_rmax_A),
    xlabel=r"$r$ ($\AA$)",
    ylabel=r"$G(r)$ ($\AA^{-2}$)",
    title="Each structure with its independent optimum scale",
)

for axis in axes:
    axis.grid(True, linestyle=":", alpha=0.5)
    axis.legend(fontsize=10, loc="upper right")
    axis.tick_params(direction="in", labelsize=12)

fig.suptitle("Calculated X-ray PDF: PZS MQ1 vs PZS MQ3 (1.4–10 Å)", fontsize=15)
fig.tight_layout()
fig.savefig(OUT / "pdf_mq1_mq3_1p4_10A.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(json.dumps(metrics, indent=2))
