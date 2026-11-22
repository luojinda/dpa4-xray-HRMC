"""Uniform analysis of four PZS_quench_new Inline 100k runs."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

from dpa4_xray_hrmc.analysis.phase3_structure import Phase3StructureEvaluator
from dpa4_xray_hrmc.mlip.dpa4_backend import DPA4Backend
from dpa4_xray_hrmc.rmc.config import load_phase2_config
from dpa4_xray_hrmc.rmc.objectives import ScaleMode, ScaleObjective
from dpa4_xray_hrmc.rmc.runner import inclusive_grid
from dpa4_xray_hrmc.xray_pdf.experimental import read_pdfgetx_gr
from dpa4_xray_hrmc.xray_pdf.form_factors import q_dependent_weights
from dpa4_xray_hrmc.xray_pdf.partial_rdf import compute_partial_rdfs
from dpa4_xray_hrmc.xray_pdf.transform import forward_transform


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/PZS_phase3/phase3C/PZS_quench_new"
RUN_NAMES = (
    "lora_no_scale",
    "lora_bounded_dynamic",
    "frozen_no_scale",
    "frozen_bounded_dynamic",
)
RUNS = {
    name: BASE / f"validation_100k_seed101_T100_d0p1_r1p4_10p0A_{name}"
    for name in RUN_NAMES
}
MODELS = {
    "lora": ROOT / "models/PZS_new_20261121/DPA4_PZS_new_LoRA.pt2",
    "frozen": ROOT
    / "models/PZS_new_20261121/DPA4-Neo-OMat24-v20260627_frozen.pt2",
}
OUT = ROOT / "results/PZS_quench_new/four_inline_100k_comparison"
OUT.mkdir(parents=True, exist_ok=True)

config = replace(
    load_phase2_config(ROOT / "configs/pzs_phase2_common.yaml"),
    structure="inputs/PZS_quench_new/PZS_quench_new.cif",
    fit_rmin_A=1.4,
    fit_rmax_A=10.0,
    move_max_A=0.1,
)
experiment = read_pdfgetx_gr(ROOT / config.experimental_gr)
fit_mask = (experiment.r_A >= 1.4) & (experiment.r_A <= 10.0)
q_grid = inclusive_grid(config.qmin_Ainv, config.qmax_Ainv, config.qstep_Ainv)


def calculate_curve(atoms):
    partials = compute_partial_rdfs(atoms, config.rdf_rcut_A, config.rdf_dr_A)
    weights = q_dependent_weights(partials.element_counts, q_grid)
    return forward_transform(
        partials,
        weights,
        experiment.r_A,
        config.biso_A2,
        config.lorch_mod,
    ).g_r_A2


atoms = {
    "quench_new_MQ": read(ROOT / config.structure),
    **{name: read(path / "final_structure.extxyz") for name, path in RUNS.items()},
}
curves = {name: calculate_curve(value) for name, value in atoms.items()}
objective = ScaleObjective.from_initial(
    curves["quench_new_MQ"][fit_mask], experiment.g_r_A2[fit_mask]
)
modes = {
    "quench_new_MQ": ScaleMode.BOUNDED_DYNAMIC,
    **{
        name: ScaleMode.BOUNDED_DYNAMIC if "bounded" in name else ScaleMode.NO_SCALE
        for name in RUNS
    },
}
scores = {
    name: objective.score(modes[name], curve[fit_mask])
    for name, curve in curves.items()
}
summaries = {
    name: json.loads((path / "summary.json").read_text())
    for name, path in RUNS.items()
}

evaluator = Phase3StructureEvaluator(
    ROOT,
    config,
    ROOT / "configs/pzs_phase2_5_coordination_cutoffs.json",
)
diagnostics = {name: evaluator.evaluate(value) for name, value in atoms.items()}
for value in diagnostics.values():
    value.pop("g_r_A2", None)

metrics = {
    "fit_range_A": [1.4, 10.0],
    "scale_initial": objective.s0,
    "scale_bounds": [objective.lower_scale, objective.upper_scale],
    "runs": {},
    "diagnostics": diagnostics,
    "cross_model_energy": {},
}
for name, score in scores.items():
    common = objective.common_scores(score.A, score.B)
    metrics["runs"][name] = {
        **common,
        "Rw_active": float(np.sqrt(score.loss / objective.C)),
        "scale_active": score.scale,
        "scale_raw": score.B / score.A,
    }
for name, summary in summaries.items():
    metrics["runs"][name].update(
        {
            key: summary[key]
            for key in (
                "acceptance_rate",
                "energy_eV_atom",
                "delta_energy_from_MQ_eV",
                "P_fourfold",
                "Si_fourfold",
                "hard_sphere_rejected",
                "wall_s",
                "model_sha256",
            )
        }
    )

required = set(atoms["quench_new_MQ"].get_chemical_symbols())
for model_name, model_path in MODELS.items():
    backend = DPA4Backend(model_path, required_elements=required)
    initial_energy = backend.get_energy(atoms["quench_new_MQ"])
    metrics["cross_model_energy"][model_name] = {}
    for name, structure in atoms.items():
        energy = backend.get_energy(structure)
        metrics["cross_model_energy"][model_name][name] = {
            "energy_eV": energy,
            "delta_vs_quench_new_MQ_meV_atom": (
                (energy - initial_energy) * 1000.0 / len(structure)
            ),
        }

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
with (OUT / "pdf_curves.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["r_A", "G_experiment", *[f"G_{name}" for name in curves]])
    writer.writerows(
        zip(experiment.r_A, experiment.g_r_A2, *curves.values(), strict=True)
    )

trajectory_rows = []
initial_no_scale = objective.score(
    ScaleMode.NO_SCALE, curves["quench_new_MQ"][fit_mask]
)
initial_no_scale_rw = objective.common_scores(
    initial_no_scale.A, initial_no_scale.B
)["Rw_no_scale"]
for name, path in RUNS.items():
    trace = json.loads((path / "trace.json").read_text())
    summary = summaries[name]
    if not trace or int(trace[-1]["attempted"]) != 100000:
        trace.append(summary)
    trajectory_rows.append(
        {
            "run": name,
            "attempts": 0,
            "Rw_active": metrics["runs"]["quench_new_MQ"]["Rw_active"]
            if "bounded" in name
            else initial_no_scale_rw,
            "scale": objective.s0 if "bounded" in name else 1.0,
            "delta_energy_meV_atom": 0.0,
        }
    )
    initial_energy = summary["energy_eV"] - summary["delta_energy_from_MQ_eV"]
    for record in trace:
        trajectory_rows.append(
            {
                "run": name,
                "attempts": int(record["attempted"]),
                "Rw_active": record["Rw_active"],
                "scale": record["scale_current"],
                "delta_energy_meV_atom": (
                    (record["energy_eV"] - initial_energy) * 1000.0 / len(atoms[name])
                ),
            }
        )

with (OUT / "trajectories.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0]))
    writer.writeheader()
    writer.writerows(trajectory_rows)

colors = {
    "quench_new_MQ": "#555555",
    "lora_no_scale": "#377EB8",
    "lora_bounded_dynamic": "#984EA3",
    "frozen_no_scale": "#4DAF4A",
    "frozen_bounded_dynamic": "#E41A1C",
}
labels = {
    "quench_new_MQ": "Quench new MQ",
    "lora_no_scale": "LoRA no scale",
    "lora_bounded_dynamic": "LoRA dynamic",
    "frozen_no_scale": "Frozen no scale",
    "frozen_bounded_dynamic": "Frozen dynamic",
}

figure, axes = plt.subplots(2, 1, figsize=(12, 10), dpi=300, sharex=True)
for axis, active in zip(axes, (False, True), strict=True):
    axis.plot(experiment.r_A, experiment.g_r_A2, color="black", linewidth=2, label="Experiment")
    for name, curve in curves.items():
        scale = scores[name].scale if active else 1.0
        rw_key = "Rw_active" if active else "Rw_no_scale"
        rw = metrics["runs"][name][rw_key]
        axis.plot(
            experiment.r_A,
            scale * curve,
            color=colors[name],
            linewidth=1.5,
            linestyle="--" if name == "quench_new_MQ" else "-",
            label=f"{labels[name]} (Rw={rw:.3f}, w={scale:.3f})",
        )
    axis.set(
        xlim=(1.4, 10.0),
        ylabel=r"$G(r)$ ($\AA^{-2}$)",
        title="Active strategy scale" if active else "Common no-scale diagnostic",
    )
    axis.grid(True, linestyle=":", alpha=0.45)
    axis.legend(fontsize=8, ncol=2)
axes[-1].set_xlabel(r"$r$ ($\AA$)")
figure.tight_layout()
figure.savefig(OUT / "pdf_four_runs_1p4_10A.png", dpi=300, bbox_inches="tight")
plt.close(figure)

figure, axes = plt.subplots(3, 1, figsize=(12, 12), dpi=300, sharex=True)
for name in RUNS:
    rows = [row for row in trajectory_rows if row["run"] == name]
    attempts = [row["attempts"] for row in rows]
    axes[0].plot(
        attempts,
        [row["Rw_active"] for row in rows],
        color=colors[name],
        linewidth=1.5,
        label=labels[name],
    )
    axes[1].plot(
        attempts,
        [row["delta_energy_meV_atom"] for row in rows],
        color=colors[name],
        linewidth=1.5,
    )
    axes[2].plot(
        attempts,
        [row["scale"] for row in rows],
        color=colors[name],
        linewidth=1.5,
    )
axes[0].set_ylabel("Active Rw")
axes[1].set_ylabel("Delta E (meV/atom)")
axes[2].set_ylabel("Scale w")
axes[2].set_xlabel("MC attempts")
axes[2].axhspan(
    objective.lower_scale, objective.upper_scale, color="#999999", alpha=0.12
)
axes[0].legend(fontsize=9, ncol=2)
for axis in axes:
    axis.grid(True, linestyle=":", alpha=0.45)
figure.tight_layout()
figure.savefig(OUT / "trajectories_Rw_energy_scale.png", dpi=300, bbox_inches="tight")
plt.close(figure)

geometry = []
for name, diagnostic in diagnostics.items():
    geometry.append(
        {
            "structure": name,
            "P_angle_RMS_deg": diagnostic["tetrahedra"]["P"][
                "angle_rms_from_ideal_deg"
            ]["mean"],
            "Si_angle_RMS_deg": diagnostic["tetrahedra"]["Si"][
                "angle_rms_from_ideal_deg"
            ]["mean"],
            "P_CSM_percent": diagnostic["tetrahedra"]["P"][
                "fixed_center_csm_percent"
            ]["mean"],
            "Si_CSM_percent": diagnostic["tetrahedra"]["Si"][
                "fixed_center_csm_percent"
            ]["mean"],
        }
    )
with (OUT / "tetrahedral_geometry.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(geometry[0]))
    writer.writeheader()
    writer.writerows(geometry)

figure, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
keys = (
    ("P_angle_RMS_deg", "P angle RMS (deg)"),
    ("Si_angle_RMS_deg", "Si angle RMS (deg)"),
    ("P_CSM_percent", "P CSM (%)"),
    ("Si_CSM_percent", "Si CSM (%)"),
)
for axis, (key, title) in zip(axes.flat, keys, strict=True):
    bars = axis.bar(
        [labels[row["structure"]] for row in geometry],
        [row[key] for row in geometry],
        color=[colors[row["structure"]] for row in geometry],
    )
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=20)
    axis.bar_label(bars, fmt="%.3f", fontsize=8)
    axis.grid(True, axis="y", linestyle=":", alpha=0.45)
figure.tight_layout()
figure.savefig(OUT / "tetrahedral_geometry.png", dpi=300, bbox_inches="tight")
plt.close(figure)

print(json.dumps(metrics, indent=2))
