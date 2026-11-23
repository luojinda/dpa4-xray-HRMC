"""Analyze the two PZS MQ3 LoRA/P-total Inline 100k trajectories.

The P tetrahedron definition intentionally matches the production P-total
constraint: P-O < 1.80 A plus P-Cl < 2.30 A, with total coordination four.
"""

from __future__ import annotations

import csv
import itertools
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read

from dpa4_xray_hrmc.analysis.tetrahedra import IDEAL_ANGLE_DEG, fixed_center_csm
from dpa4_xray_hrmc.rmc.config import load_phase2_config
from dpa4_xray_hrmc.rmc.coordination import (
    CoordinationCutoffs,
    IncrementalCoordination,
)
from dpa4_xray_hrmc.rmc.hard_sphere import (
    HardSphereModel,
    minimum_image_vectors,
)
from dpa4_xray_hrmc.rmc.objectives import ScaleMode, ScaleObjective
from dpa4_xray_hrmc.rmc.runner import inclusive_grid
from dpa4_xray_hrmc.xray_pdf.experimental import read_pdfgetx_gr
from dpa4_xray_hrmc.xray_pdf.form_factors import q_dependent_weights
from dpa4_xray_hrmc.xray_pdf.partial_rdf import compute_partial_rdfs
from dpa4_xray_hrmc.xray_pdf.transform import forward_transform


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/PZS_phase3/phase3C/PZS_MQ_3"
RUNS = {
    "no_scale": BASE
    / "validation_100k_seed101_T100_d0p1_r1p4_10p0A_lora_no_scale_p_total",
    "bounded_dynamic": BASE
    / "validation_100k_seed101_T100_d0p1_r1p4_10p0A_lora_bounded_dynamic_p_total",
}
OUT = BASE / "comparison_100k_lora_p_total"
OUT.mkdir(parents=True, exist_ok=True)

STRUCTURE_KEYS = ("initial_mq3", "no_scale", "bounded_dynamic")
LABELS = {
    "initial_mq3": "Initial MQ3",
    "no_scale": "No scale 100k",
    "bounded_dynamic": "Dynamic 100k",
}
COLORS = {
    "initial_mq3": "#555555",
    "no_scale": "#2F6B9A",
    "bounded_dynamic": "#C45A36",
}
PLOT_DPI = 300
P_CL_CUTOFF_A = 2.30


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    data = np.asarray(values, dtype=float)
    return {
        "count": int(data.size),
        "mean": float(data.mean()),
        "std": float(data.std()),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def ligand_environment(
    atoms: Atoms,
    center_element: str,
    ligand_cutoffs: tuple[tuple[str, float], ...],
) -> dict:
    symbols = np.asarray(atoms.get_chemical_symbols())
    cell = np.asarray(atoms.cell)
    centers = np.flatnonzero(symbols == center_element)
    bond_values = {
        f"{center_element}-{ligand}": [] for ligand, _ in ligand_cutoffs
    }
    patterns: Counter[str] = Counter()
    total_cn: list[int] = []
    angle_rms: list[float] = []
    all_angles: list[float] = []
    csms: list[float] = []
    volumes: list[float] = []

    for center in centers:
        selected_vectors: list[np.ndarray] = []
        selected_ligands: list[str] = []
        ligand_counts: Counter[str] = Counter()
        for ligand, cutoff in ligand_cutoffs:
            indices = np.flatnonzero(symbols == ligand)
            vectors = minimum_image_vectors(
                atoms.positions[indices] - atoms.positions[center], cell
            )
            distances = np.linalg.norm(vectors, axis=1)
            inside = distances < cutoff
            for vector, distance in zip(
                vectors[inside], distances[inside], strict=True
            ):
                selected_vectors.append(vector)
                selected_ligands.append(ligand)
                ligand_counts[ligand] += 1
                bond_values[f"{center_element}-{ligand}"].append(float(distance))

        cn = len(selected_vectors)
        total_cn.append(cn)
        pattern = "".join(
            f"{ligand}{ligand_counts[ligand]}"
            for ligand, _ in ligand_cutoffs
            if ligand_counts[ligand]
        )
        patterns[pattern or "none"] += 1
        if cn != 4:
            continue

        vectors = np.asarray(selected_vectors)
        site_angles = []
        for first, second in itertools.combinations(range(4), 2):
            cosine = np.dot(vectors[first], vectors[second]) / (
                np.linalg.norm(vectors[first]) * np.linalg.norm(vectors[second])
            )
            site_angles.append(
                float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
            )
        all_angles.extend(site_angles)
        angle_rms.append(
            float(
                np.sqrt(
                    np.mean((np.asarray(site_angles) - IDEAL_ANGLE_DEG) ** 2)
                )
            )
        )
        csms.append(float(fixed_center_csm(vectors)))
        volumes.append(
            abs(
                float(
                    np.linalg.det(
                        np.stack(
                            (
                                vectors[1] - vectors[0],
                                vectors[2] - vectors[0],
                                vectors[3] - vectors[0],
                            )
                        )
                    )
                )
            )
            / 6.0
        )

    return {
        "centers": int(len(centers)),
        "fourfold_count": int(np.sum(np.asarray(total_cn) == 4)),
        "fourfold_fraction": float(np.mean(np.asarray(total_cn) == 4)),
        "total_cn_histogram": {
            str(int(value)): int(count)
            for value, count in sorted(Counter(total_cn).items())
        },
        "ligand_patterns": dict(sorted(patterns.items())),
        "bond_lengths_A": {
            key: stats(values) for key, values in bond_values.items()
        },
        "angles_deg": stats(all_angles),
        "angle_rms_from_ideal_deg": stats(angle_rms),
        "fixed_center_csm_percent": stats(csms),
        "volume_A3": stats(volumes),
    }


def hard_sphere_diagnostics(atoms: Atoms, model: HardSphereModel) -> dict:
    symbols = np.asarray(atoms.get_chemical_symbols())
    cell = np.asarray(atoms.cell)
    minima = {pair: np.inf for pair in model.pairs}
    violation_count = 0
    worst_margin = np.inf
    for first in range(len(atoms) - 1):
        vectors = minimum_image_vectors(
            atoms.positions[first + 1 :] - atoms.positions[first], cell
        )
        distances = np.linalg.norm(vectors, axis=1)
        for offset, distance in enumerate(distances, start=first + 1):
            pair = tuple(sorted((str(symbols[first]), str(symbols[offset]))))
            cutoff = model.cutoffs_A[pair]
            minima[pair] = min(minima[pair], float(distance))
            worst_margin = min(worst_margin, float(distance - cutoff))
            violation_count += int(distance < cutoff - 1e-10)
    return {
        "violation_count": int(violation_count),
        "minimum_margin_A": float(worst_margin),
        "pair_minima_A": {
            f"{first}-{second}": float(minima[(first, second)])
            for first, second in model.pairs
        },
    }


config = replace(
    load_phase2_config(ROOT / "configs/pzs_phase2_common.yaml"),
    structure="inputs/PZS_MQ_3/PZS_equilibrated.cif",
    fit_rmin_A=1.4,
    fit_rmax_A=10.0,
    move_max_A=0.1,
)
experiment = read_pdfgetx_gr(ROOT / config.experimental_gr)
fit_mask = (experiment.r_A >= config.fit_rmin_A) & (
    experiment.r_A <= config.fit_rmax_A
)
q_grid = inclusive_grid(config.qmin_Ainv, config.qmax_Ainv, config.qstep_Ainv)


def pdf_curve(atoms: Atoms) -> np.ndarray:
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
    "initial_mq3": read(ROOT / config.structure),
    **{
        name: read(directory / "final_structure.extxyz")
        for name, directory in RUNS.items()
    },
}
summaries = {
    name: json.loads((directory / "summary.json").read_text())
    for name, directory in RUNS.items()
}
curves = {name: pdf_curve(structure) for name, structure in atoms.items()}
objective = ScaleObjective.from_initial(
    curves["initial_mq3"][fit_mask], experiment.g_r_A2[fit_mask]
)
score_modes = {
    "initial_mq3": ScaleMode.NO_SCALE,
    "no_scale": ScaleMode.NO_SCALE,
    "bounded_dynamic": ScaleMode.BOUNDED_DYNAMIC,
}
scores = {
    name: objective.score(score_modes[name], curve[fit_mask])
    for name, curve in curves.items()
}
initial_dynamic_score = objective.score(
    ScaleMode.BOUNDED_DYNAMIC, curves["initial_mq3"][fit_mask]
)

cutoffs = CoordinationCutoffs.from_mapping(
    json.loads((ROOT / "configs/pzs_phase2_5_coordination_cutoffs.json").read_text())
)
hard_sphere_model = HardSphereModel.from_atoms(
    atoms["initial_mq3"], config.hard_sphere_scale
)

local_metrics: dict[str, dict] = {}
for name, structure in atoms.items():
    coordination = IncrementalCoordination(
        structure,
        cutoffs,
        p_coordination_mode="P-total",
        p_cl_cutoff_A=P_CL_CUTOFF_A,
    ).current_snapshot()
    p_geometry = ligand_environment(
        structure, "P", (("O", cutoffs.p_o_A), ("Cl", P_CL_CUTOFF_A))
    )
    si_geometry = ligand_environment(
        structure,
        "Si",
        (("O", cutoffs.si_o_A), ("Cl", cutoffs.si_cl_A)),
    )
    zr_total_bins = np.arange(len(coordination.zr_hist_total))
    zr_o_bins = np.arange(len(coordination.zr_hist_o))
    zr_count = int(np.sum(coordination.zr_hist_total))
    local_metrics[name] = {
        "P": p_geometry,
        "Si": si_geometry,
        "Zr": {
            "centers": zr_count,
            "total_cn_histogram": coordination.zr_hist_total.tolist(),
            "o_cn_histogram": coordination.zr_hist_o.tolist(),
            "mean_total_cn": float(
                np.dot(zr_total_bins, coordination.zr_hist_total) / zr_count
            ),
            "mean_o_cn": float(
                np.dot(zr_o_bins, coordination.zr_hist_o) / zr_count
            ),
        },
        "coordination_penalty_components": coordination.components.as_dict(),
        "hard_sphere": hard_sphere_diagnostics(structure, hard_sphere_model),
    }

pdf_metrics = {}
for name, score in scores.items():
    common = objective.common_scores(score.A, score.B)
    pdf_metrics[name] = {
        **common,
        "Rw_active": float(np.sqrt(score.loss / objective.C)),
        "scale_active": float(score.scale),
        "scale_raw": float(score.B / score.A),
    }
pdf_metrics["initial_mq3"]["Rw_dynamic"] = float(
    np.sqrt(initial_dynamic_score.loss / objective.C)
)
pdf_metrics["initial_mq3"]["scale_dynamic"] = float(initial_dynamic_score.scale)

trajectory_rows: list[dict] = []
for name, directory in RUNS.items():
    summary = summaries[name]
    records = json.loads((directory / "trace.json").read_text())
    if not records or int(records[-1]["attempted"]) != int(summary["attempted"]):
        records.append(summary)
    initial_energy = summary["energy_eV"] - summary["delta_energy_from_MQ_eV"]
    initial_score = (
        initial_dynamic_score if name == "bounded_dynamic" else scores["initial_mq3"]
    )
    initial_common = objective.common_scores(initial_score.A, initial_score.B)
    trajectory_rows.append(
        {
            "run": name,
            "attempts": 0,
            "Rw_active": float(np.sqrt(initial_score.loss / objective.C)),
            "Rw_no_scale": float(initial_common["Rw_no_scale"]),
            "scale": float(initial_score.scale),
            "delta_energy_meV_atom": 0.0,
        }
    )
    for record in records:
        trajectory_rows.append(
            {
                "run": name,
                "attempts": int(record["attempted"]),
                "Rw_active": float(record["Rw_active"]),
                "Rw_no_scale": float(record["Rw_no_scale"]),
                "scale": float(record["scale_current"]),
                "delta_energy_meV_atom": float(
                    (record["energy_eV"] - initial_energy)
                    * 1000.0
                    / len(atoms[name])
                ),
            }
        )

with (OUT / "pdf_curves.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "r_A",
            "G_experiment",
            "G_initial_mq3_raw",
            "G_no_scale_raw",
            "G_bounded_dynamic_raw",
            "G_initial_mq3_dynamic_scaled",
            "G_bounded_dynamic_scaled",
        ]
    )
    writer.writerows(
        zip(
            experiment.r_A,
            experiment.g_r_A2,
            curves["initial_mq3"],
            curves["no_scale"],
            curves["bounded_dynamic"],
            initial_dynamic_score.scale * curves["initial_mq3"],
            scores["bounded_dynamic"].scale * curves["bounded_dynamic"],
            strict=True,
        )
    )

with (OUT / "trajectories.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0]))
    writer.writeheader()
    writer.writerows(trajectory_rows)

local_rows = []
for name in STRUCTURE_KEYS:
    item = local_metrics[name]
    local_rows.append(
        {
            "structure": name,
            "P_fourfold_fraction": item["P"]["fourfold_fraction"],
            "Si_fourfold_fraction": item["Si"]["fourfold_fraction"],
            "P_angle_RMS_deg": item["P"]["angle_rms_from_ideal_deg"]["mean"],
            "Si_angle_RMS_deg": item["Si"]["angle_rms_from_ideal_deg"]["mean"],
            "P_CSM_percent": item["P"]["fixed_center_csm_percent"]["mean"],
            "Si_CSM_percent": item["Si"]["fixed_center_csm_percent"]["mean"],
            "P_volume_A3": item["P"]["volume_A3"]["mean"],
            "Si_volume_A3": item["Si"]["volume_A3"]["mean"],
            "P_O_mean_A": item["P"]["bond_lengths_A"]["P-O"]["mean"],
            "P_Cl_mean_A": item["P"]["bond_lengths_A"]["P-Cl"]["mean"],
            "Si_O_mean_A": item["Si"]["bond_lengths_A"]["Si-O"]["mean"],
            "Si_Cl_mean_A": item["Si"]["bond_lengths_A"]["Si-Cl"]["mean"],
            "Zr_mean_total_CN": item["Zr"]["mean_total_cn"],
            "Zr_mean_O_CN": item["Zr"]["mean_o_cn"],
            "hard_sphere_violations": item["hard_sphere"]["violation_count"],
            "hard_sphere_min_margin_A": item["hard_sphere"]["minimum_margin_A"],
        }
    )

with (OUT / "local_structure_metrics.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(local_rows[0]))
    writer.writeheader()
    writer.writerows(local_rows)

max_zr_bin = max(
    max(len(local_metrics[name]["Zr"]["total_cn_histogram"]),
        len(local_metrics[name]["Zr"]["o_cn_histogram"]))
    for name in STRUCTURE_KEYS
)
with (OUT / "zr_coordination_histograms.csv").open("w", newline="") as handle:
    fieldnames = ["coordination"] + [
        f"{name}_{kind}" for name in STRUCTURE_KEYS for kind in ("total", "O")
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for cn in range(max_zr_bin):
        row = {"coordination": cn}
        for name in STRUCTURE_KEYS:
            for kind, key in (("total", "total_cn_histogram"), ("O", "o_cn_histogram")):
                values = local_metrics[name]["Zr"][key]
                row[f"{name}_{kind}"] = values[cn] if cn < len(values) else 0
        writer.writerow(row)

# PDF: one panel per production scale strategy.
figure, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=PLOT_DPI, sharex=True)
axes[0].plot(experiment.r_A, experiment.g_r_A2, color="black", lw=2.0, label="Experiment")
axes[0].plot(
    experiment.r_A,
    curves["initial_mq3"],
    color=COLORS["initial_mq3"],
    lw=1.4,
    ls="--",
    label=f"Initial MQ3 ($R_w$={pdf_metrics['initial_mq3']['Rw_no_scale']:.3f})",
)
axes[0].plot(
    experiment.r_A,
    curves["no_scale"],
    color=COLORS["no_scale"],
    lw=1.7,
    label=f"No scale 100k ($R_w$={pdf_metrics['no_scale']['Rw_active']:.3f})",
)
axes[0].set_title("No-scale Inline HRMC")

axes[1].plot(experiment.r_A, experiment.g_r_A2, color="black", lw=2.0, label="Experiment")
axes[1].plot(
    experiment.r_A,
    initial_dynamic_score.scale * curves["initial_mq3"],
    color=COLORS["initial_mq3"],
    lw=1.4,
    ls="--",
    label=(
        f"Initial MQ3 ($R_w$={pdf_metrics['initial_mq3']['Rw_dynamic']:.3f}, "
        f"$w$={initial_dynamic_score.scale:.3f})"
    ),
)
axes[1].plot(
    experiment.r_A,
    scores["bounded_dynamic"].scale * curves["bounded_dynamic"],
    color=COLORS["bounded_dynamic"],
    lw=1.7,
    label=(
        f"Dynamic 100k ($R_w$={pdf_metrics['bounded_dynamic']['Rw_active']:.3f}, "
        f"$w$={scores['bounded_dynamic'].scale:.3f})"
    ),
)
axes[1].set_title("Bounded-dynamic-scale Inline HRMC")
for axis in axes:
    axis.set_xlim(1.4, 10.0)
    axis.set_ylabel(r"$G(r)$ ($\mathrm{\AA}^{-2}$)")
    axis.grid(True, ls=":", alpha=0.45)
    axis.legend(fontsize=9, ncol=1)
axes[-1].set_xlabel(r"$r$ ($\mathrm{\AA}$)")
figure.tight_layout()
figure.savefig(OUT / "pdf_two_strategies_1p4_10A.png", bbox_inches="tight")
plt.close(figure)

# Common unscaled PDF diagnostic so that structure amplitudes are compared fairly.
figure, axis = plt.subplots(figsize=(11, 5), dpi=PLOT_DPI)
axis.plot(experiment.r_A, experiment.g_r_A2, color="black", lw=2.0, label="Experiment")
for name in STRUCTURE_KEYS:
    axis.plot(
        experiment.r_A,
        curves[name],
        color=COLORS[name],
        lw=1.7 if name != "initial_mq3" else 1.4,
        ls="--" if name == "initial_mq3" else "-",
        label=f"{LABELS[name]} ($R_w$={pdf_metrics[name]['Rw_no_scale']:.3f})",
    )
axis.set(
    xlim=(1.4, 10.0),
    xlabel=r"$r$ ($\mathrm{\AA}$)",
    ylabel=r"$G(r)$ ($\mathrm{\AA}^{-2}$)",
    title="Common no-scale PDF diagnostic",
)
axis.grid(True, ls=":", alpha=0.45)
axis.legend(fontsize=9)
figure.tight_layout()
figure.savefig(OUT / "pdf_common_no_scale_1p4_10A.png", bbox_inches="tight")
plt.close(figure)

# Four diagnostics make the strategy-dependent and common residuals explicit.
figure, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=PLOT_DPI, sharex=True)
for name in RUNS:
    rows = [row for row in trajectory_rows if row["run"] == name]
    attempts = [row["attempts"] for row in rows]
    axes[0, 0].plot(attempts, [row["Rw_active"] for row in rows], color=COLORS[name], lw=1.6, label=LABELS[name])
    axes[0, 1].plot(attempts, [row["Rw_no_scale"] for row in rows], color=COLORS[name], lw=1.6)
    axes[1, 0].plot(attempts, [row["delta_energy_meV_atom"] for row in rows], color=COLORS[name], lw=1.6)
    axes[1, 1].plot(attempts, [row["scale"] for row in rows], color=COLORS[name], lw=1.6)
axes[0, 0].set_ylabel(r"Active $R_w$")
axes[0, 0].set_title("Strategy-specific PDF residual")
axes[0, 1].set_ylabel(r"Common no-scale $R_w$")
axes[0, 1].set_title("Unscaled structural diagnostic")
axes[1, 0].set_ylabel(r"$\Delta E$ (meV/atom)")
axes[1, 0].set_xlabel("MC attempts")
axes[1, 0].set_title("DPA4 energy relative to initial MQ3")
axes[1, 1].set_ylabel("Scale $w$")
axes[1, 1].set_xlabel("MC attempts")
axes[1, 1].set_title("PDF scale evolution")
axes[1, 1].axhspan(objective.lower_scale, objective.upper_scale, color="#999999", alpha=0.13, label="Dynamic bounds")
axes[1, 1].axhline(objective.s0, color="#777777", ls=":", lw=1.0)
axes[0, 0].legend(fontsize=9)
for axis in axes.flat:
    axis.grid(True, ls=":", alpha=0.45)
    axis.set_xlim(0, 100000)
figure.tight_layout()
figure.savefig(OUT / "trajectories_Rw_energy_scale.png", bbox_inches="tight")
plt.close(figure)

# P-total/Si tetrahedral geometry, including the mixed P(O3Cl) tetrahedron.
geometry_specs = (
    ("P_fourfold_fraction", "P total fourfold (%)", 100.0),
    ("Si_fourfold_fraction", "Si total fourfold (%)", 100.0),
    ("P_angle_RMS_deg", "P tetrahedral angle RMS (deg)", 1.0),
    ("Si_angle_RMS_deg", "Si tetrahedral angle RMS (deg)", 1.0),
    ("P_CSM_percent", "P tetrahedral CSM (%)", 1.0),
    ("Si_CSM_percent", "Si tetrahedral CSM (%)", 1.0),
    ("P_volume_A3", r"P tetrahedron volume ($\mathrm{\AA}^3$)", 1.0),
    ("Si_volume_A3", r"Si tetrahedron volume ($\mathrm{\AA}^3$)", 1.0),
)
figure, axes = plt.subplots(2, 4, figsize=(14, 7.5), dpi=PLOT_DPI)
for axis, (key, title, multiplier) in zip(axes.flat, geometry_specs, strict=True):
    values = [row[key] * multiplier for row in local_rows]
    bars = axis.bar(
        [LABELS[row["structure"]] for row in local_rows],
        values,
        color=[COLORS[row["structure"]] for row in local_rows],
        width=0.68,
    )
    axis.set_title(title, fontsize=10)
    axis.tick_params(axis="x", rotation=24, labelsize=8)
    axis.grid(True, axis="y", ls=":", alpha=0.45)
    axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ymax = max(values)
    axis.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)
figure.suptitle("Local tetrahedral geometry (P-total definition)", fontsize=13)
figure.tight_layout()
figure.savefig(OUT / "local_tetrahedral_comparison.png", bbox_inches="tight")
plt.close(figure)

# Bond lengths with standard deviations across all bonds selected by the cutoffs.
bond_specs = (
    ("P", "P-O", "P-O bond length"),
    ("P", "P-Cl", "P-Cl bond length"),
    ("Si", "Si-O", "Si-O bond length"),
    ("Si", "Si-Cl", "Si-Cl bond length"),
)
figure, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=PLOT_DPI)
for axis, (center, bond, title) in zip(axes.flat, bond_specs, strict=True):
    means = [local_metrics[name][center]["bond_lengths_A"][bond]["mean"] for name in STRUCTURE_KEYS]
    errors = [local_metrics[name][center]["bond_lengths_A"][bond]["std"] for name in STRUCTURE_KEYS]
    bars = axis.bar(
        [LABELS[name] for name in STRUCTURE_KEYS],
        means,
        yerr=errors,
        capsize=3,
        color=[COLORS[name] for name in STRUCTURE_KEYS],
        width=0.68,
    )
    axis.set_title(f"{title}, mean ± SD")
    axis.set_ylabel(r"Distance ($\mathrm{\AA}$)")
    axis.tick_params(axis="x", rotation=24, labelsize=8)
    axis.grid(True, axis="y", ls=":", alpha=0.45)
    axis.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
figure.tight_layout()
figure.savefig(OUT / "bond_length_comparison.png", bbox_inches="tight")
plt.close(figure)

# Zr total and oxygen coordination distributions.
figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=PLOT_DPI)
for name in STRUCTURE_KEYS:
    total = np.asarray(local_metrics[name]["Zr"]["total_cn_histogram"])
    oxygen = np.asarray(local_metrics[name]["Zr"]["o_cn_histogram"])
    axes[0].plot(np.arange(len(total)), total, marker="o", ms=4, lw=1.5, color=COLORS[name], label=LABELS[name])
    axes[1].plot(np.arange(len(oxygen)), oxygen, marker="o", ms=4, lw=1.5, color=COLORS[name], label=LABELS[name])
axes[0].set_title("Zr total coordination distribution")
axes[1].set_title("Zr-O coordination distribution")
for axis in axes:
    axis.set_xlabel("Coordination number")
    axis.set_ylabel("Number of Zr atoms")
    axis.set_xlim(0, 12)
    axis.set_xticks(range(13))
    axis.grid(True, ls=":", alpha=0.45)
axes[0].legend(fontsize=8)
figure.tight_layout()
figure.savefig(OUT / "zr_coordination_distributions.png", bbox_inches="tight")
plt.close(figure)

metrics = {
    "fit_range_A": [1.4, 10.0],
    "model_sha256": summaries["no_scale"]["model_sha256"],
    "scale_initial": objective.s0,
    "scale_bounds": [objective.lower_scale, objective.upper_scale],
    "pdf": pdf_metrics,
    "runs": summaries,
    "local_structure": local_metrics,
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))


def fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


report_lines = [
    "# PZS MQ3: two Inline 100k HRMC runs",
    "",
    "Both runs use the LoRA DPA4 model, seed 101, T=100 K, d=0.1 A, "
    "PDF range 1.4-10 A, and the P-total coordination constraint.",
    "",
    "## PDF and energy",
    "",
    "| Structure | Active Rw | No-scale Rw | Scale | Delta E (meV/atom) | Acceptance |",
    "|---|---:|---:|---:|---:|---:|",
]
for name in ("no_scale", "bounded_dynamic"):
    summary = summaries[name]
    report_lines.append(
        f"| {LABELS[name]} | {pdf_metrics[name]['Rw_active']:.5f} | "
        f"{pdf_metrics[name]['Rw_no_scale']:.5f} | "
        f"{pdf_metrics[name]['scale_active']:.5f} | "
        f"{summary['delta_energy_from_MQ_eV'] * 1000 / len(atoms[name]):.3f} | "
        f"{summary['acceptance_rate']:.4f} |"
    )
report_lines.extend(
    [
        "",
        "The bounded-dynamic active Rw includes its fitted scale. Its common "
        "no-scale Rw is reported separately and must be used for an unscaled "
        "amplitude comparison.",
        "",
        "## Local structure",
        "",
        "P tetrahedra include both P-O (<1.80 A) and P-Cl (<2.30 A), so the "
        "initial P(O3Cl) unit is counted as four-coordinate.",
        "",
        "| Structure | P 4-fold | Si 4-fold | P angle RMS | Si angle RMS | P CSM | Si CSM | Zr CN | Zr-O CN | HS violations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
)
for row in local_rows:
    report_lines.append(
        f"| {LABELS[row['structure']]} | {100*row['P_fourfold_fraction']:.2f}% | "
        f"{100*row['Si_fourfold_fraction']:.2f}% | {fmt(row['P_angle_RMS_deg'])} | "
        f"{fmt(row['Si_angle_RMS_deg'])} | {fmt(row['P_CSM_percent'])} | "
        f"{fmt(row['Si_CSM_percent'])} | {fmt(row['Zr_mean_total_CN'])} | "
        f"{fmt(row['Zr_mean_O_CN'])} | {row['hard_sphere_violations']} |"
    )
report_lines.extend(
    [
        "",
        "Detailed ligand patterns, bond statistics, coordination histograms, "
        "and hard-sphere margins are stored in metrics.json and the CSV files.",
        "",
        "All comparisons are for one seed (101); they are not three-seed averages.",
    ]
)
(OUT / "REPORT.md").write_text("\n".join(report_lines) + "\n")

print(
    json.dumps(
        {
            "output": str(OUT),
            "pdf": pdf_metrics,
            "local_rows": local_rows,
        },
        indent=2,
    )
)
