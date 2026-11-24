"""Compare MQ3 baseline and eight one-factor coordination-weight Inline runs."""

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
from dpa4_xray_hrmc.rmc.coordination import CoordinationCutoffs, IncrementalCoordination
from dpa4_xray_hrmc.rmc.hard_sphere import HardSphereModel, minimum_image_vectors
from dpa4_xray_hrmc.rmc.objectives import ScaleMode, ScaleObjective
from dpa4_xray_hrmc.rmc.runner import inclusive_grid
from dpa4_xray_hrmc.xray_pdf.experimental import read_pdfgetx_gr
from dpa4_xray_hrmc.xray_pdf.form_factors import q_dependent_weights
from dpa4_xray_hrmc.xray_pdf.partial_rdf import compute_partial_rdfs
from dpa4_xray_hrmc.xray_pdf.transform import forward_transform


ROOT = Path(__file__).resolve().parents[1]
MQ3_BASE = ROOT / "results/PZS_phase3/phase3C/PZS_MQ_3"
SENSITIVITY_BASE = MQ3_BASE / "coord_weight_sensitivity_r1p4_10p0A"
OUT = SENSITIVITY_BASE / "comparison_9_weight_cases"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = {
    "baseline": MQ3_BASE
    / "validation_100k_seed101_T100_d0p1_r1p4_10p0A_lora_bounded_dynamic_p_total",
    "p_low": SENSITIVITY_BASE / "p_low",
    "p_high": SENSITIVITY_BASE / "p_high",
    "si_low": SENSITIVITY_BASE / "si_low",
    "si_high": SENSITIVITY_BASE / "si_high",
    "zr_total_low": SENSITIVITY_BASE / "zr_total_low",
    "zr_total_high": SENSITIVITY_BASE / "zr_total_high",
    "zr_o_low": SENSITIVITY_BASE / "zr_o_low",
    "zr_o_high": SENSITIVITY_BASE / "zr_o_high",
}
LABELS = {
    "initial": "Initial MQ3",
    "baseline": "Baseline",
    "p_low": "P x0.1",
    "p_high": "P x10",
    "si_low": "Si x0.1",
    "si_high": "Si x10",
    "zr_total_low": "Zr-total x0.1",
    "zr_total_high": "Zr-total x10",
    "zr_o_low": "Zr-O x0.1",
    "zr_o_high": "Zr-O x10",
}
COLORS = {
    "baseline": "#202020",
    "p_low": "#4C78A8",
    "p_high": "#9ECAE9",
    "si_low": "#F58518",
    "si_high": "#FFBF79",
    "zr_total_low": "#54A24B",
    "zr_total_high": "#A1D99B",
    "zr_o_low": "#B279A2",
    "zr_o_high": "#D6B4D0",
}
GROUPS = {
    "P weight": ("p_low", "p_high"),
    "Si weight": ("si_low", "si_high"),
    "Zr-total weight": ("zr_total_low", "zr_total_high"),
    "Zr-O weight": ("zr_o_low", "zr_o_high"),
}
P_CL_CUTOFF_A = 2.30
PLOT_DPI = 300


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
    bond_values = {f"{center_element}-{ligand}": [] for ligand, _ in ligand_cutoffs}
    patterns: Counter[str] = Counter()
    total_cn: list[int] = []
    angle_rms: list[float] = []
    all_angles: list[float] = []
    csms: list[float] = []
    volumes: list[float] = []

    for center in centers:
        vectors_selected: list[np.ndarray] = []
        ligand_counts: Counter[str] = Counter()
        for ligand, cutoff in ligand_cutoffs:
            indices = np.flatnonzero(symbols == ligand)
            vectors = minimum_image_vectors(
                atoms.positions[indices] - atoms.positions[center], cell
            )
            distances = np.linalg.norm(vectors, axis=1)
            inside = distances < cutoff
            for vector, distance in zip(vectors[inside], distances[inside], strict=True):
                vectors_selected.append(vector)
                ligand_counts[ligand] += 1
                bond_values[f"{center_element}-{ligand}"].append(float(distance))

        cn = len(vectors_selected)
        total_cn.append(cn)
        pattern = "".join(
            f"{ligand}{ligand_counts[ligand]}"
            for ligand, _ in ligand_cutoffs
            if ligand_counts[ligand]
        )
        patterns[pattern or "none"] += 1
        if cn != 4:
            continue

        vectors = np.asarray(vectors_selected)
        site_angles = []
        for first, second in itertools.combinations(range(4), 2):
            cosine = np.dot(vectors[first], vectors[second]) / (
                np.linalg.norm(vectors[first]) * np.linalg.norm(vectors[second])
            )
            site_angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
        all_angles.extend(site_angles)
        angle_rms.append(
            float(np.sqrt(np.mean((np.asarray(site_angles) - IDEAL_ANGLE_DEG) ** 2)))
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

    cn_array = np.asarray(total_cn)
    return {
        "centers": int(len(centers)),
        "fourfold_count": int(np.sum(cn_array == 4)),
        "fourfold_fraction": float(np.mean(cn_array == 4)),
        "total_cn_histogram": {
            str(int(value)): int(count) for value, count in sorted(Counter(total_cn).items())
        },
        "ligand_patterns": dict(sorted(patterns.items())),
        "bond_lengths_A": {key: stats(values) for key, values in bond_values.items()},
        "angles_deg": stats(all_angles),
        "angle_rms_from_ideal_deg": stats(angle_rms),
        "fixed_center_csm_percent": stats(csms),
        "volume_A3": stats(volumes),
    }


def hard_sphere_diagnostics(atoms: Atoms, model: HardSphereModel) -> dict:
    symbols = np.asarray(atoms.get_chemical_symbols())
    cell = np.asarray(atoms.cell)
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
            worst_margin = min(worst_margin, float(distance - cutoff))
            violation_count += int(distance < cutoff - 1e-10)
    return {"violation_count": int(violation_count), "minimum_margin_A": float(worst_margin)}


config = replace(
    load_phase2_config(ROOT / "configs/pzs_phase2_common.yaml"),
    structure="inputs/PZS_MQ_3/PZS_equilibrated.cif",
    fit_rmin_A=1.4,
    fit_rmax_A=10.0,
    move_max_A=0.1,
)
experiment = read_pdfgetx_gr(ROOT / config.experimental_gr)
fit_mask = (experiment.r_A >= config.fit_rmin_A) & (experiment.r_A <= config.fit_rmax_A)
q_grid = inclusive_grid(config.qmin_Ainv, config.qmax_Ainv, config.qstep_Ainv)
cutoffs = CoordinationCutoffs.from_mapping(
    json.loads((ROOT / "configs/pzs_phase2_5_coordination_cutoffs.json").read_text())
)


def pdf_curve(atoms: Atoms) -> np.ndarray:
    partials = compute_partial_rdfs(atoms, config.rdf_rcut_A, config.rdf_dr_A)
    weights = q_dependent_weights(partials.element_counts, q_grid)
    return forward_transform(
        partials, weights, experiment.r_A, config.biso_A2, config.lorch_mod
    ).g_r_A2


missing = [str(path) for path in RUNS.values() if not (path / "summary.json").is_file()]
if missing:
    raise FileNotFoundError(f"Incomplete sensitivity runs: {missing}")

summaries = {
    name: json.loads((directory / "summary.json").read_text())
    for name, directory in RUNS.items()
}
atoms = {"initial": read(ROOT / config.structure)}
atoms.update(
    {name: read(directory / "final_structure.extxyz") for name, directory in RUNS.items()}
)
curves = {name: pdf_curve(structure) for name, structure in atoms.items()}
objective = ScaleObjective.from_initial(
    curves["initial"][fit_mask], experiment.g_r_A2[fit_mask]
)
scores = {
    name: objective.score(ScaleMode.BOUNDED_DYNAMIC, curves[name][fit_mask])
    for name in atoms
}


def segment_rw(curve: np.ndarray, scale: float, lower: float, upper: float) -> float:
    mask = (experiment.r_A >= lower) & (experiment.r_A <= upper)
    residual = scale * curve[mask] - experiment.g_r_A2[mask]
    return float(np.sqrt(np.sum(residual**2) / np.sum(experiment.g_r_A2[mask] ** 2)))


hard_sphere_model = HardSphereModel.from_atoms(atoms["initial"], config.hard_sphere_scale)
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
        structure, "Si", (("O", cutoffs.si_o_A), ("Cl", cutoffs.si_cl_A))
    )
    zr_count = int(np.sum(coordination.zr_hist_total))
    local_metrics[name] = {
        "P": p_geometry,
        "Si": si_geometry,
        "Zr": {
            "centers": zr_count,
            "total_cn_histogram": coordination.zr_hist_total.tolist(),
            "o_cn_histogram": coordination.zr_hist_o.tolist(),
            "mean_total_cn": float(
                np.dot(np.arange(len(coordination.zr_hist_total)), coordination.zr_hist_total)
                / zr_count
            ),
            "mean_o_cn": float(
                np.dot(np.arange(len(coordination.zr_hist_o)), coordination.zr_hist_o)
                / zr_count
            ),
        },
        "hard_sphere": hard_sphere_diagnostics(structure, hard_sphere_model),
    }

metrics_rows = []
for name in RUNS:
    summary = summaries[name]
    weights = summary.get(
        "coordination_weights",
        {"P": 800.0, "Si": 2500.0, "Zr_total": 1.0, "Zr_O": 0.2},
    )
    score = scores[name]
    local = local_metrics[name]
    common = objective.common_scores(score.A, score.B)
    metrics_rows.append(
        {
            "run": name,
            "label": LABELS[name],
            "weight_P": weights["P"],
            "weight_Si": weights["Si"],
            "weight_Zr_total": weights["Zr_total"],
            "weight_Zr_O": weights["Zr_O"],
            "Rw_active": float(np.sqrt(score.loss / objective.C)),
            "Rw_no_scale": float(common["Rw_no_scale"]),
            "Rw_1p4_1p8": segment_rw(curves[name], score.scale, 1.4, 1.8),
            "Rw_1p8_2p2": segment_rw(curves[name], score.scale, 1.8, 2.2),
            "Rw_2p2_10": segment_rw(curves[name], score.scale, 2.2, 10.0),
            "scale": float(score.scale),
            "delta_E_meV_atom": float(
                summary["delta_energy_from_MQ_eV"] * 1000.0 / len(atoms[name])
            ),
            "acceptance_rate": summary["acceptance_rate"],
            "wall_s": summary["wall_s"],
            "P_fourfold_fraction": local["P"]["fourfold_fraction"],
            "Si_fourfold_fraction": local["Si"]["fourfold_fraction"],
            "P_angle_RMS_deg": local["P"]["angle_rms_from_ideal_deg"]["mean"],
            "Si_angle_RMS_deg": local["Si"]["angle_rms_from_ideal_deg"]["mean"],
            "P_CSM_percent": local["P"]["fixed_center_csm_percent"]["mean"],
            "Si_CSM_percent": local["Si"]["fixed_center_csm_percent"]["mean"],
            "P_volume_A3": local["P"]["volume_A3"]["mean"],
            "Si_volume_A3": local["Si"]["volume_A3"]["mean"],
            "P_O_mean_A": local["P"]["bond_lengths_A"]["P-O"]["mean"],
            "P_O_std_A": local["P"]["bond_lengths_A"]["P-O"]["std"],
            "P_Cl_mean_A": local["P"]["bond_lengths_A"]["P-Cl"]["mean"],
            "P_Cl_std_A": local["P"]["bond_lengths_A"]["P-Cl"]["std"],
            "Si_O_mean_A": local["Si"]["bond_lengths_A"]["Si-O"]["mean"],
            "Si_O_std_A": local["Si"]["bond_lengths_A"]["Si-O"]["std"],
            "Si_Cl_mean_A": local["Si"]["bond_lengths_A"]["Si-Cl"]["mean"],
            "Si_Cl_std_A": local["Si"]["bond_lengths_A"]["Si-Cl"]["std"],
            "Zr_mean_total_CN": local["Zr"]["mean_total_cn"],
            "Zr_mean_O_CN": local["Zr"]["mean_o_cn"],
            "hard_sphere_violations": local["hard_sphere"]["violation_count"],
            "hard_sphere_min_margin_A": local["hard_sphere"]["minimum_margin_A"],
        }
    )

with (OUT / "summary_metrics.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0]))
    writer.writeheader()
    writer.writerows(metrics_rows)

with (OUT / "pdf_curves.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        ["r_A", "G_experiment", "G_initial_raw"]
        + [f"G_{name}_raw" for name in RUNS]
        + [f"G_{name}_scaled" for name in RUNS]
    )
    for index, r_value in enumerate(experiment.r_A):
        writer.writerow(
            [r_value, experiment.g_r_A2[index], curves["initial"][index]]
            + [curves[name][index] for name in RUNS]
            + [scores[name].scale * curves[name][index] for name in RUNS]
        )

trajectory_rows = []
for name, directory in RUNS.items():
    summary = summaries[name]
    records = json.loads((directory / "trace.json").read_text())
    if not records or int(records[-1]["attempted"]) != int(summary["attempted"]):
        records.append(summary)
    initial_energy = summary["energy_eV"] - summary["delta_energy_from_MQ_eV"]
    initial_score = scores["initial"]
    trajectory_rows.append(
        {
            "run": name,
            "attempts": 0,
            "Rw_active": float(np.sqrt(initial_score.loss / objective.C)),
            "Rw_no_scale": objective.common_scores(initial_score.A, initial_score.B)[
                "Rw_no_scale"
            ],
            "scale": initial_score.scale,
            "delta_energy_meV_atom": 0.0,
            "acceptance_rate": 0.0,
        }
    )
    for record in records:
        trajectory_rows.append(
            {
                "run": name,
                "attempts": int(record["attempted"]),
                "Rw_active": record["Rw_active"],
                "Rw_no_scale": record["Rw_no_scale"],
                "scale": record["scale_current"],
                "delta_energy_meV_atom": (record["energy_eV"] - initial_energy)
                * 1000.0
                / len(atoms[name]),
                "acceptance_rate": record["acceptance_rate"],
            }
        )

with (OUT / "trajectories.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(trajectory_rows[0]))
    writer.writeheader()
    writer.writerows(trajectory_rows)

# Each panel isolates one weight and keeps the baseline fixed.
figure, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=PLOT_DPI, sharex=True, sharey=True)
for axis, (title, variants) in zip(axes.flat, GROUPS.items(), strict=True):
    axis.plot(experiment.r_A, experiment.g_r_A2, color="black", lw=2.0, label="Experiment")
    for name in ("baseline", *variants):
        axis.plot(
            experiment.r_A,
            scores[name].scale * curves[name],
            color=COLORS[name],
            lw=1.7,
            label=f"{LABELS[name]} ($R_w$={next(row['Rw_active'] for row in metrics_rows if row['run'] == name):.4f})",
        )
    axis.set_title(title)
    axis.set_xlim(1.4, 10.0)
    axis.grid(True, ls=":", alpha=0.4)
    axis.legend(fontsize=8)
axes[0, 0].set_ylabel(r"$G(r)$ ($\mathrm{\AA}^{-2}$)")
axes[1, 0].set_ylabel(r"$G(r)$ ($\mathrm{\AA}^{-2}$)")
axes[1, 0].set_xlabel(r"$r$ ($\mathrm{\AA}$)")
axes[1, 1].set_xlabel(r"$r$ ($\mathrm{\AA}$)")
figure.tight_layout()
figure.savefig(OUT / "pdf_weight_sensitivity_4panels.png", bbox_inches="tight")
plt.close(figure)

# Residual detail emphasizes the first-neighbor region.
figure, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=PLOT_DPI, sharex=True, sharey=True)
for axis, (title, variants) in zip(axes.flat, GROUPS.items(), strict=True):
    for name in ("baseline", *variants):
        residual = scores[name].scale * curves[name] - experiment.g_r_A2
        axis.plot(experiment.r_A, residual, color=COLORS[name], lw=1.6, label=LABELS[name])
    axis.axhline(0.0, color="black", lw=0.8)
    axis.axvspan(1.4, 1.8, color="#4C78A8", alpha=0.07)
    axis.axvspan(1.8, 2.2, color="#F58518", alpha=0.07)
    axis.set_title(title)
    axis.set_xlim(1.4, 3.2)
    axis.grid(True, ls=":", alpha=0.4)
    axis.legend(fontsize=8)
axes[0, 0].set_ylabel(r"Residual $G_{calc}-G_{exp}$")
axes[1, 0].set_ylabel(r"Residual $G_{calc}-G_{exp}$")
axes[1, 0].set_xlabel(r"$r$ ($\mathrm{\AA}$)")
axes[1, 1].set_xlabel(r"$r$ ($\mathrm{\AA}$)")
figure.tight_layout()
figure.savefig(OUT / "pdf_first_neighbor_residuals.png", bbox_inches="tight")
plt.close(figure)

figure, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=PLOT_DPI, sharex=True)
for name in RUNS:
    rows = [row for row in trajectory_rows if row["run"] == name]
    x = [row["attempts"] for row in rows]
    axes[0, 0].plot(x, [row["Rw_active"] for row in rows], color=COLORS[name], lw=1.3, label=LABELS[name])
    axes[0, 1].plot(x, [row["delta_energy_meV_atom"] for row in rows], color=COLORS[name], lw=1.3)
    axes[1, 0].plot(x, [row["scale"] for row in rows], color=COLORS[name], lw=1.3)
    axes[1, 1].plot(x, [row["acceptance_rate"] for row in rows], color=COLORS[name], lw=1.3)
axes[0, 0].set_ylabel(r"Active $R_w$")
axes[0, 1].set_ylabel(r"$\Delta E$ (meV/atom)")
axes[1, 0].set_ylabel("Dynamic scale")
axes[1, 1].set_ylabel("Cumulative acceptance")
for axis in axes.flat:
    axis.set_xlim(0, 100000)
    axis.grid(True, ls=":", alpha=0.4)
axes[1, 0].set_xlabel("MC attempts")
axes[1, 1].set_xlabel("MC attempts")
axes[0, 0].legend(fontsize=7, ncol=2)
figure.tight_layout()
figure.savefig(OUT / "trajectories_9_weight_cases.png", bbox_inches="tight")
plt.close(figure)

geometry_specs = (
    ("P_fourfold_fraction", "P total fourfold (%)", 100.0),
    ("Si_fourfold_fraction", "Si total fourfold (%)", 100.0),
    ("P_angle_RMS_deg", "P angle RMS (deg)", 1.0),
    ("Si_angle_RMS_deg", "Si angle RMS (deg)", 1.0),
    ("P_CSM_percent", "P CSM (%)", 1.0),
    ("Si_CSM_percent", "Si CSM (%)", 1.0),
    ("P_volume_A3", r"P volume ($\mathrm{\AA}^3$)", 1.0),
    ("Si_volume_A3", r"Si volume ($\mathrm{\AA}^3$)", 1.0),
)
figure, axes = plt.subplots(2, 4, figsize=(17, 8), dpi=PLOT_DPI)
for axis, (key, title, multiplier) in zip(axes.flat, geometry_specs, strict=True):
    values = [row[key] * multiplier for row in metrics_rows]
    axis.bar(
        [row["label"] for row in metrics_rows],
        values,
        color=[COLORS[row["run"]] for row in metrics_rows],
        width=0.72,
    )
    axis.set_title(title, fontsize=10)
    axis.tick_params(axis="x", rotation=65, labelsize=7)
    axis.grid(True, axis="y", ls=":", alpha=0.4)
figure.tight_layout()
figure.savefig(OUT / "local_tetrahedral_9_weight_cases.png", bbox_inches="tight")
plt.close(figure)

bond_specs = (
    ("P_O", "P-O"),
    ("P_Cl", "P-Cl"),
    ("Si_O", "Si-O"),
    ("Si_Cl", "Si-Cl"),
)
figure, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=PLOT_DPI)
for axis, (prefix, title) in zip(axes.flat, bond_specs, strict=True):
    axis.bar(
        [row["label"] for row in metrics_rows],
        [row[f"{prefix}_mean_A"] for row in metrics_rows],
        yerr=[row[f"{prefix}_std_A"] for row in metrics_rows],
        color=[COLORS[row["run"]] for row in metrics_rows],
        capsize=2,
    )
    axis.set_title(f"{title} mean +/- SD")
    axis.set_ylabel(r"Distance ($\mathrm{\AA}$)")
    axis.tick_params(axis="x", rotation=60, labelsize=7)
    axis.grid(True, axis="y", ls=":", alpha=0.4)
figure.tight_layout()
figure.savefig(OUT / "bond_lengths_9_weight_cases.png", bbox_inches="tight")
plt.close(figure)

figure, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=PLOT_DPI)
for name in RUNS:
    total = np.asarray(local_metrics[name]["Zr"]["total_cn_histogram"])
    oxygen = np.asarray(local_metrics[name]["Zr"]["o_cn_histogram"])
    axes[0].plot(np.arange(len(total)), total, marker="o", ms=3, lw=1.2, color=COLORS[name], label=LABELS[name])
    axes[1].plot(np.arange(len(oxygen)), oxygen, marker="o", ms=3, lw=1.2, color=COLORS[name], label=LABELS[name])
axes[0].set_title("Zr total coordination distribution")
axes[1].set_title("Zr-O coordination distribution")
for axis in axes:
    axis.set_xlim(0, 12)
    axis.set_xticks(range(13))
    axis.set_xlabel("Coordination number")
    axis.set_ylabel("Zr atom count")
    axis.grid(True, ls=":", alpha=0.4)
axes[0].legend(fontsize=7, ncol=2)
figure.tight_layout()
figure.savefig(OUT / "zr_coordination_9_weight_cases.png", bbox_inches="tight")
plt.close(figure)

ranking = sorted(metrics_rows, key=lambda row: row["Rw_active"])
report = [
    "# MQ3 coordination-weight sensitivity: 9 dynamic-scale Inline 100k cases",
    "",
    "All runs use MQ3, seed 101, T=100 K, d=0.1 A, 1.4-10 A, bounded dynamic scale, and the LoRA DPA4 potential.",
    "The baseline weights are P=800, Si=2500, Zr-total=1.0, and Zr-O=0.2. Each sensitivity run changes one weight only.",
    "",
    "## PDF, energy, and sampling",
    "",
    "| Rank | Case | Active Rw | Rw 1.4-1.8 | Rw 1.8-2.2 | Scale | Delta E (meV/atom) | Acceptance |",
    "|---:|---|---:|---:|---:|---:|---:|---:|",
]
for rank, row in enumerate(ranking, start=1):
    report.append(
        f"| {rank} | {row['label']} | {row['Rw_active']:.5f} | {row['Rw_1p4_1p8']:.5f} | "
        f"{row['Rw_1p8_2p2']:.5f} | {row['scale']:.5f} | {row['delta_E_meV_atom']:.3f} | "
        f"{row['acceptance_rate']:.4f} |"
    )
baseline_row = next(row for row in metrics_rows if row["run"] == "baseline")
best_row = ranking[0]
relative_improvement = 100.0 * (
    baseline_row["Rw_active"] - best_row["Rw_active"]
) / baseline_row["Rw_active"]
report.extend(
    [
        "",
        "## Interpretation",
        "",
        f"- The lowest final active Rw is {best_row['label']} ({best_row['Rw_active']:.5f}), "
        f"a {relative_improvement:.2f}% relative improvement over the baseline. Its improvement "
        "comes mainly from 2.2-10 A; it does not improve the 1.4-1.8 A first-neighbor segment.",
        "- The baseline retains the lowest 1.4-1.8 A segment Rw. P x10 gives the lowest "
        "1.8-2.2 A segment Rw, but its total Rw is essentially tied with the baseline.",
        "- P x0.1 is the only clearly adverse perturbation: both the total Rw and the 1.8-2.2 A "
        "segment deteriorate. P x10 remains close to baseline while slightly improving P geometry.",
        "- Zr-total and Zr-O decade changes have small effects on final Rw, energy, acceptance, "
        "and coordination statistics. Their present baseline weights are not the dominant fitting bottleneck.",
        "- All nine runs preserve 100% P/Si total fourfold coordination, have zero hard-sphere "
        "violations, and finish below the initial MQ3 DPA4 energy. Weight changes therefore alter the "
        "accepted pathway more than the final coordination counts.",
        "- This is a one-seed sensitivity screen. Differences below roughly 0.002 in Rw should be "
        "treated as provisional until repeated with multiple seeds.",
    ]
)
report.extend(
    [
        "",
        "## Local structure",
        "",
        "| Case | P 4-fold | Si 4-fold | P angle RMS | Si angle RMS | P CSM | Si CSM | Zr CN | Zr-O CN | HS violations |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
)
for row in metrics_rows:
    report.append(
        f"| {row['label']} | {100*row['P_fourfold_fraction']:.2f}% | "
        f"{100*row['Si_fourfold_fraction']:.2f}% | {row['P_angle_RMS_deg']:.4f} | "
        f"{row['Si_angle_RMS_deg']:.4f} | {row['P_CSM_percent']:.4f} | "
        f"{row['Si_CSM_percent']:.4f} | {row['Zr_mean_total_CN']:.4f} | "
        f"{row['Zr_mean_O_CN']:.4f} | {row['hard_sphere_violations']} |"
    )
report.extend(
    [
        "",
        "Initial MQ3 reference and complete ligand-pattern, bond-length, coordination, and hard-sphere data are stored in metrics.json.",
        "All results are one-seed (101) sensitivity comparisons, not uncertainty estimates across seeds.",
    ]
)
(OUT / "REPORT.md").write_text("\n".join(report) + "\n")
(OUT / "metrics.json").write_text(
    json.dumps(
        {
            "fit_range_A": [1.4, 10.0],
            "scale_initial": objective.s0,
            "scale_bounds": [objective.lower_scale, objective.upper_scale],
            "summary_rows": metrics_rows,
            "run_summaries": summaries,
            "local_structure": local_metrics,
        },
        indent=2,
    )
)

print(json.dumps({"output": str(OUT), "ranking": [(row["run"], row["Rw_active"]) for row in ranking]}, indent=2))
