"""Analyze the expanded-new-MQ LoRA bounded-dynamic one-million-step run."""

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
BASE = ROOT / "results/PZS_phase3/phase3C/PZS_new_20261121"
RUN_100K = BASE / "validation_100k_seed101_T100_d0p1_r1p4_10p0A_lora_bounded_dynamic"
RUN_1M = BASE / "production_1M_seed101_T100_d0p1_r1p4_10p0A_lora_bounded_dynamic"
MODEL = ROOT / "models/PZS_new_20261121/DPA4_PZS_new_LoRA.pt2"
OUT = ROOT / "results/PZS_new_20261121/lora_dynamic_1M"
OUT.mkdir(parents=True, exist_ok=True)

config = replace(
    load_phase2_config(ROOT / "configs/pzs_phase2_common.yaml"),
    structure="inputs/PZS_new_20261121/PZS_new_20261121.cif",
    fit_rmin_A=1.4,
    fit_rmax_A=10.0,
    move_max_A=0.1,
)
experiment = read_pdfgetx_gr(ROOT / config.experimental_gr)
fit_mask = (experiment.r_A >= 1.4) & (experiment.r_A <= 10.0)
q_grid = inclusive_grid(config.qmin_Ainv, config.qmax_Ainv, config.qstep_Ainv)


def curve(atoms):
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
    "new_MQ": read(ROOT / config.structure),
    "lora_dynamic_100k": read(RUN_100K / "final_structure.extxyz"),
    "lora_dynamic_1M": read(RUN_1M / "final_structure.extxyz"),
}
curves = {name: curve(value) for name, value in atoms.items()}
objective = ScaleObjective.from_initial(
    curves["new_MQ"][fit_mask], experiment.g_r_A2[fit_mask]
)
scores = {
    name: objective.score(ScaleMode.BOUNDED_DYNAMIC, value[fit_mask])
    for name, value in curves.items()
}

evaluator = Phase3StructureEvaluator(
    ROOT,
    config,
    ROOT / "configs/pzs_phase2_5_coordination_cutoffs.json",
)
diagnostics = {name: evaluator.evaluate(value) for name, value in atoms.items()}
for value in diagnostics.values():
    value.pop("g_r_A2", None)

backend = DPA4Backend(MODEL, required_elements=set(atoms["new_MQ"].get_chemical_symbols()))
initial_energy = backend.get_energy(atoms["new_MQ"])
summaries = {
    "lora_dynamic_100k": json.loads((RUN_100K / "summary.json").read_text()),
    "lora_dynamic_1M": json.loads((RUN_1M / "summary.json").read_text()),
}

metrics = {
    "fit_range_A": [1.4, 10.0],
    "scale_initial": objective.s0,
    "scale_bounds": [objective.lower_scale, objective.upper_scale],
    "runs": {},
    "diagnostics": diagnostics,
}
for name, score in scores.items():
    common = objective.common_scores(score.A, score.B)
    energy = backend.get_energy(atoms[name])
    metrics["runs"][name] = {
        **common,
        "Rw_active": float(np.sqrt(score.loss / objective.C)),
        "scale_active": score.scale,
        "energy_eV": energy,
        "delta_vs_new_MQ_meV_atom": (energy - initial_energy) * 1000.0 / len(atoms[name]),
    }
for name, summary in summaries.items():
    metrics["runs"][name].update(
        {
            key: summary[key]
            for key in (
                "attempted",
                "accepted",
                "acceptance_rate",
                "wall_s",
                "P_fourfold",
                "Si_fourfold",
                "energy_drift_eV",
                "model_sha256",
            )
        }
    )

(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
with (OUT / "pdf_curves.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["r_A", "G_experiment", *[f"G_{name}" for name in curves]])
    writer.writerows(
        zip(experiment.r_A, experiment.g_r_A2, *curves.values(), strict=True)
    )

colors = {
    "new_MQ": "#666666",
    "lora_dynamic_100k": "#377EB8",
    "lora_dynamic_1M": "#E41A1C",
}
labels = {
    "new_MQ": "New MQ",
    "lora_dynamic_100k": "LoRA dynamic 100k",
    "lora_dynamic_1M": "LoRA dynamic 1M",
}
figure, axis = plt.subplots(figsize=(12, 5), dpi=300)
axis.plot(experiment.r_A, experiment.g_r_A2, color="black", linewidth=2, label="Experiment")
for name, value in curves.items():
    score = scores[name]
    rw = metrics["runs"][name]["Rw_active"]
    axis.plot(
        experiment.r_A,
        score.scale * value,
        color=colors[name],
        linewidth=1.6,
        linestyle="--" if name == "new_MQ" else "-",
        label=f"{labels[name]} (Rw={rw:.3f}, w={score.scale:.3f})",
    )
axis.set(
    xlim=(1.4, 10.0),
    xlabel=r"$r$ ($\AA$)",
    ylabel=r"$G(r)$ ($\AA^{-2}$)",
)
axis.grid(True, linestyle=":", alpha=0.45)
axis.legend(fontsize=9, ncol=2)
figure.tight_layout()
figure.savefig(OUT / "pdf_newMQ_100k_1M.png", dpi=300, bbox_inches="tight")
plt.close(figure)

summary_1m = summaries["lora_dynamic_1M"]
trace = json.loads((RUN_1M / "trace.json").read_text())
if not trace or int(trace[-1]["attempted"]) != 1000000:
    trace.append(summary_1m)
initial_run_energy = summary_1m["energy_eV"] - summary_1m["delta_energy_from_MQ_eV"]
rows = [
    {
        "attempts": 0,
        "Rw_active": metrics["runs"]["new_MQ"]["Rw_active"],
        "scale": objective.s0,
        "delta_energy_meV_atom": 0.0,
    }
]
for record in trace:
    rows.append(
        {
            "attempts": int(record["attempted"]),
            "Rw_active": record["Rw_active"],
            "scale": record["scale_current"],
            "delta_energy_meV_atom": (
                (record["energy_eV"] - initial_run_energy) * 1000.0
                / len(atoms["lora_dynamic_1M"])
            ),
        }
    )
with (OUT / "trajectory_1M.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

figure, axes = plt.subplots(3, 1, figsize=(12, 10), dpi=300, sharex=True)
attempts = [row["attempts"] for row in rows]
axes[0].plot(attempts, [row["Rw_active"] for row in rows], color="#E41A1C")
axes[1].plot(attempts, [row["delta_energy_meV_atom"] for row in rows], color="#E41A1C")
axes[2].plot(attempts, [row["scale"] for row in rows], color="#E41A1C")
axes[0].set_ylabel("Active Rw")
axes[1].set_ylabel("Delta E (meV/atom)")
axes[2].set_ylabel("Scale w")
axes[2].set_xlabel("MC attempts")
axes[2].axhspan(
    objective.lower_scale, objective.upper_scale, color="#999999", alpha=0.12
)
for axis in axes:
    axis.grid(True, linestyle=":", alpha=0.45)
figure.tight_layout()
figure.savefig(OUT / "trajectory_1M.png", dpi=300, bbox_inches="tight")
plt.close(figure)

print(json.dumps(metrics, indent=2))
