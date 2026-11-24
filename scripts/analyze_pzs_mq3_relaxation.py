"""Compare original and fixed-cell-relaxed MQ3 PDF and local environments."""

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
RELAX = ROOT / "results/PZS_phase3/relaxations/PZS_MQ_3_lora_fixed_cell"
OUT = RELAX / "structure_comparison"
OUT.mkdir(parents=True, exist_ok=True)
STRUCTURES = {
    "initial_mq3": ROOT / "inputs/PZS_MQ_3/PZS_equilibrated.cif",
    "relaxed_mq3": RELAX / "relaxed_structure.extxyz",
}
LABELS = {"initial_mq3": "Initial MQ3", "relaxed_mq3": "Relaxed MQ3"}
COLORS = {"initial_mq3": "#555555", "relaxed_mq3": "#D55E00"}
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
    *,
    tetrahedral: bool,
) -> dict:
    symbols = np.asarray(atoms.get_chemical_symbols())
    cell = np.asarray(atoms.cell)
    centers = np.flatnonzero(symbols == center_element)
    bonds = {f"{center_element}-{ligand}": [] for ligand, _ in ligand_cutoffs}
    patterns: Counter[str] = Counter()
    total_cn: list[int] = []
    angle_rms: list[float] = []
    all_angles: list[float] = []
    csms: list[float] = []
    volumes: list[float] = []

    for center in centers:
        vectors_selected: list[np.ndarray] = []
        counts: Counter[str] = Counter()
        for ligand, cutoff in ligand_cutoffs:
            indices = np.flatnonzero(symbols == ligand)
            vectors = minimum_image_vectors(
                atoms.positions[indices] - atoms.positions[center], cell
            )
            distances = np.linalg.norm(vectors, axis=1)
            inside = distances < cutoff
            for vector, distance in zip(vectors[inside], distances[inside], strict=True):
                vectors_selected.append(vector)
                counts[ligand] += 1
                bonds[f"{center_element}-{ligand}"].append(float(distance))
        cn = len(vectors_selected)
        total_cn.append(cn)
        pattern = "".join(
            f"{ligand}{counts[ligand]}"
            for ligand, _ in ligand_cutoffs
            if counts[ligand]
        )
        patterns[pattern or "none"] += 1
        if not tetrahedral or cn != 4:
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
    result = {
        "centers": int(len(centers)),
        "total_cn_histogram": {
            str(int(value)): int(count) for value, count in sorted(Counter(total_cn).items())
        },
        "ligand_patterns": dict(sorted(patterns.items())),
        "bond_lengths_A": {key: stats(values) for key, values in bonds.items()},
    }
    if tetrahedral:
        result.update(
            {
                "fourfold_count": int(np.sum(cn_array == 4)),
                "fourfold_fraction": float(np.mean(cn_array == 4)),
                "angles_deg": stats(all_angles),
                "angle_rms_from_ideal_deg": stats(angle_rms),
                "fixed_center_csm_percent": stats(csms),
                "volume_A3": stats(volumes),
            }
        )
    return result


def hard_sphere_diagnostics(atoms: Atoms, model: HardSphereModel) -> dict:
    symbols = np.asarray(atoms.get_chemical_symbols())
    cell = np.asarray(atoms.cell)
    minima = {pair: np.inf for pair in model.pairs}
    violations = 0
    for first in range(len(atoms) - 1):
        vectors = minimum_image_vectors(
            atoms.positions[first + 1 :] - atoms.positions[first], cell
        )
        distances = np.linalg.norm(vectors, axis=1)
        for other, distance in enumerate(distances, start=first + 1):
            pair = tuple(sorted((str(symbols[first]), str(symbols[other]))))
            minima[pair] = min(minima[pair], float(distance))
            violations += int(distance < model.cutoffs_A[pair] - 1e-10)
    return {
        "violation_count": int(violations),
        "minimum_margin_A": float(
            min(minima[pair] - model.cutoffs_A[pair] for pair in model.pairs)
        ),
        "pair_minima_A": {f"{a}-{b}": minima[(a, b)] for a, b in model.pairs},
    }


config = replace(
    load_phase2_config(ROOT / "configs/pzs_phase2_common.yaml"),
    structure="inputs/PZS_MQ_3/PZS_equilibrated.cif",
    fit_rmin_A=1.4,
    fit_rmax_A=10.0,
)
experiment = read_pdfgetx_gr(ROOT / config.experimental_gr)
fit_mask = (experiment.r_A >= config.fit_rmin_A) & (experiment.r_A <= config.fit_rmax_A)
q_grid = inclusive_grid(config.qmin_Ainv, config.qmax_Ainv, config.qstep_Ainv)
cutoffs = CoordinationCutoffs.from_mapping(
    json.loads((ROOT / "configs/pzs_phase2_5_coordination_cutoffs.json").read_text())
)
atoms = {name: read(path) for name, path in STRUCTURES.items()}


def pdf_curve(structure: Atoms) -> np.ndarray:
    partials = compute_partial_rdfs(structure, config.rdf_rcut_A, config.rdf_dr_A)
    weights = q_dependent_weights(partials.element_counts, q_grid)
    return forward_transform(
        partials, weights, experiment.r_A, config.biso_A2, config.lorch_mod
    ).g_r_A2


curves = {name: pdf_curve(structure) for name, structure in atoms.items()}
objective = ScaleObjective.from_initial(
    curves["initial_mq3"][fit_mask], experiment.g_r_A2[fit_mask]
)
scores = {
    name: {
        "no_scale": objective.score(ScaleMode.NO_SCALE, curve[fit_mask]),
        "dynamic": objective.score(ScaleMode.BOUNDED_DYNAMIC, curve[fit_mask]),
    }
    for name, curve in curves.items()
}


def rw(score) -> float:
    return float(np.sqrt(score.loss / objective.C))


hard_sphere = HardSphereModel.from_atoms(atoms["initial_mq3"], config.hard_sphere_scale)
local = {}
for name, structure in atoms.items():
    coordination = IncrementalCoordination(
        structure,
        cutoffs,
        p_coordination_mode="P-total",
        p_cl_cutoff_A=P_CL_CUTOFF_A,
    ).current_snapshot()
    local[name] = {
        "P": ligand_environment(
            structure,
            "P",
            (("O", cutoffs.p_o_A), ("Cl", P_CL_CUTOFF_A)),
            tetrahedral=True,
        ),
        "Si": ligand_environment(
            structure,
            "Si",
            (("O", cutoffs.si_o_A), ("Cl", cutoffs.si_cl_A)),
            tetrahedral=True,
        ),
        "Zr": ligand_environment(
            structure,
            "Zr",
            (("O", cutoffs.zr_o_A), ("Cl", cutoffs.zr_cl_A)),
            tetrahedral=False,
        ),
        "zr_hist_total": coordination.zr_hist_total.tolist(),
        "zr_hist_o": coordination.zr_hist_o.tolist(),
        "hard_sphere": hard_sphere_diagnostics(structure, hard_sphere),
    }

energy_summary = json.loads((RELAX / "summary.json").read_text())
rows = []
for name in STRUCTURES:
    p = local[name]["P"]
    si = local[name]["Si"]
    zr = local[name]["Zr"]
    total_hist = np.asarray(local[name]["zr_hist_total"])
    o_hist = np.asarray(local[name]["zr_hist_o"])
    rows.append(
        {
            "structure": name,
            "Rw_no_scale": rw(scores[name]["no_scale"]),
            "Rw_dynamic": rw(scores[name]["dynamic"]),
            "dynamic_scale": scores[name]["dynamic"].scale,
            "energy_eV_atom": energy_summary["structures"][name]["energy_eV_atom"],
            "force_rms_eV_A": energy_summary["structures"][name]["force_rms_eV_A"],
            "force_max_eV_A": energy_summary["structures"][name]["force_max_eV_A"],
            "P_fourfold_fraction": p["fourfold_fraction"],
            "Si_fourfold_fraction": si["fourfold_fraction"],
            "P_angle_RMS_deg": p["angle_rms_from_ideal_deg"]["mean"],
            "Si_angle_RMS_deg": si["angle_rms_from_ideal_deg"]["mean"],
            "P_CSM_percent": p["fixed_center_csm_percent"]["mean"],
            "Si_CSM_percent": si["fixed_center_csm_percent"]["mean"],
            "P_volume_A3": p["volume_A3"]["mean"],
            "Si_volume_A3": si["volume_A3"]["mean"],
            "P_O_mean_A": p["bond_lengths_A"]["P-O"]["mean"],
            "P_O_std_A": p["bond_lengths_A"]["P-O"]["std"],
            "P_Cl_mean_A": p["bond_lengths_A"]["P-Cl"]["mean"],
            "P_Cl_std_A": p["bond_lengths_A"]["P-Cl"]["std"],
            "Si_O_mean_A": si["bond_lengths_A"]["Si-O"]["mean"],
            "Si_O_std_A": si["bond_lengths_A"]["Si-O"]["std"],
            "Si_Cl_mean_A": si["bond_lengths_A"]["Si-Cl"]["mean"],
            "Si_Cl_std_A": si["bond_lengths_A"]["Si-Cl"]["std"],
            "Zr_O_mean_A": zr["bond_lengths_A"]["Zr-O"]["mean"],
            "Zr_O_std_A": zr["bond_lengths_A"]["Zr-O"]["std"],
            "Zr_Cl_mean_A": zr["bond_lengths_A"]["Zr-Cl"]["mean"],
            "Zr_Cl_std_A": zr["bond_lengths_A"]["Zr-Cl"]["std"],
            "Zr_mean_total_CN": float(np.dot(np.arange(len(total_hist)), total_hist) / total_hist.sum()),
            "Zr_mean_O_CN": float(np.dot(np.arange(len(o_hist)), o_hist) / o_hist.sum()),
            "hard_sphere_violations": local[name]["hard_sphere"]["violation_count"],
            "hard_sphere_min_margin_A": local[name]["hard_sphere"]["minimum_margin_A"],
        }
    )

with (OUT / "comparison_metrics.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

with (OUT / "pdf_curves.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(
        [
            "r_A",
            "G_experiment",
            "G_initial_raw",
            "G_relaxed_raw",
            "G_initial_dynamic",
            "G_relaxed_dynamic",
        ]
    )
    writer.writerows(
        zip(
            experiment.r_A,
            experiment.g_r_A2,
            curves["initial_mq3"],
            curves["relaxed_mq3"],
            scores["initial_mq3"]["dynamic"].scale * curves["initial_mq3"],
            scores["relaxed_mq3"]["dynamic"].scale * curves["relaxed_mq3"],
            strict=True,
        )
    )

figure, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=PLOT_DPI, sharex=True)
for axis, mode, title in (
    (axes[0], "no_scale", "Raw calculated PDF"),
    (axes[1], "dynamic", "Bounded-dynamic scaled PDF"),
):
    axis.plot(experiment.r_A, experiment.g_r_A2, color="black", lw=2.0, label="Experiment")
    for name in STRUCTURES:
        score = scores[name][mode]
        axis.plot(
            experiment.r_A,
            score.scale * curves[name],
            color=COLORS[name],
            lw=1.7,
            ls="--" if name == "initial_mq3" else "-",
            label=f"{LABELS[name]} ($R_w$={rw(score):.4f}, w={score.scale:.3f})",
        )
    axis.set_xlim(1.4, 10.0)
    axis.set_ylabel(r"$G(r)$ ($\mathrm{\AA}^{-2}$)")
    axis.set_title(title)
    axis.grid(True, ls=":", alpha=0.4)
    axis.legend(fontsize=9)
axes[-1].set_xlabel(r"$r$ ($\mathrm{\AA}$)")
figure.tight_layout()
figure.savefig(OUT / "pdf_initial_vs_relaxed_1p4_10A.png", bbox_inches="tight")
plt.close(figure)

figure, axis = plt.subplots(figsize=(10, 5), dpi=PLOT_DPI)
axis.plot(experiment.r_A, experiment.g_r_A2, color="black", lw=2.0, label="Experiment")
for name in STRUCTURES:
    score = scores[name]["dynamic"]
    axis.plot(
        experiment.r_A,
        score.scale * curves[name],
        color=COLORS[name],
        lw=1.8,
        ls="--" if name == "initial_mq3" else "-",
        label=f"{LABELS[name]} ($R_w$={rw(score):.4f})",
    )
axis.set_xlim(1.4, 4.0)
axis.set_xlabel(r"$r$ ($\mathrm{\AA}$)")
axis.set_ylabel(r"$G(r)$ ($\mathrm{\AA}^{-2}$)")
axis.set_title("First-neighbor PDF comparison")
axis.grid(True, ls=":", alpha=0.4)
axis.legend(fontsize=9)
figure.tight_layout()
figure.savefig(OUT / "pdf_first_neighbors_initial_vs_relaxed.png", bbox_inches="tight")
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
figure, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=PLOT_DPI)
for axis, (key, title, multiplier) in zip(axes.flat, geometry_specs, strict=True):
    values = [row[key] * multiplier for row in rows]
    bars = axis.bar(
        [LABELS[row["structure"]] for row in rows],
        values,
        color=[COLORS[row["structure"]] for row in rows],
        width=0.65,
    )
    axis.set_title(title, fontsize=10)
    axis.tick_params(axis="x", rotation=20, labelsize=8)
    axis.grid(True, axis="y", ls=":", alpha=0.4)
    axis.bar_label(bars, fmt="%.3f", fontsize=8)
figure.tight_layout()
figure.savefig(OUT / "tetrahedral_geometry_initial_vs_relaxed.png", bbox_inches="tight")
plt.close(figure)

bond_specs = (("P_O", "P-O"), ("P_Cl", "P-Cl"), ("Si_O", "Si-O"), ("Si_Cl", "Si-Cl"), ("Zr_O", "Zr-O"), ("Zr_Cl", "Zr-Cl"))
figure, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=PLOT_DPI)
for axis, (prefix, title) in zip(axes.flat, bond_specs, strict=True):
    values = [row[f"{prefix}_mean_A"] for row in rows]
    errors = [row[f"{prefix}_std_A"] for row in rows]
    bars = axis.bar(
        [LABELS[row["structure"]] for row in rows],
        values,
        yerr=errors,
        capsize=4,
        color=[COLORS[row["structure"]] for row in rows],
    )
    axis.set_title(f"{title} mean +/- SD")
    axis.set_ylabel(r"Distance ($\mathrm{\AA}$)")
    axis.tick_params(axis="x", rotation=20, labelsize=8)
    axis.grid(True, axis="y", ls=":", alpha=0.4)
    axis.bar_label(bars, fmt="%.3f", fontsize=8)
figure.tight_layout()
figure.savefig(OUT / "bond_lengths_initial_vs_relaxed.png", bbox_inches="tight")
plt.close(figure)

figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=PLOT_DPI)
for name in STRUCTURES:
    total = np.asarray(local[name]["zr_hist_total"])
    oxygen = np.asarray(local[name]["zr_hist_o"])
    axes[0].plot(np.arange(len(total)), total, marker="o", color=COLORS[name], label=LABELS[name])
    axes[1].plot(np.arange(len(oxygen)), oxygen, marker="o", color=COLORS[name], label=LABELS[name])
axes[0].set_title("Zr total coordination")
axes[1].set_title("Zr-O coordination")
for axis in axes:
    axis.set_xlim(0, 12)
    axis.set_xticks(range(13))
    axis.set_xlabel("Coordination number")
    axis.set_ylabel("Zr atom count")
    axis.grid(True, ls=":", alpha=0.4)
axes[0].legend(fontsize=9)
figure.tight_layout()
figure.savefig(OUT / "zr_coordination_initial_vs_relaxed.png", bbox_inches="tight")
plt.close(figure)

metrics = {
    "fit_range_A": [1.4, 10.0],
    "scale_initial": objective.s0,
    "scale_bounds": [objective.lower_scale, objective.upper_scale],
    "summary_rows": rows,
    "local_structure": local,
    "energy_relaxation": energy_summary,
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

initial_row, relaxed_row = rows
report = [
    "# PZS MQ3 fixed-cell relaxation: PDF and local structure comparison",
    "",
    "Both PDFs use the same Q-dependent X-ray form factors, partial-RDF forward model, Qmax=24 A^-1, Lorch modification, and 1.4-10 A evaluation range.",
    "P coordination uses the P-total definition: P-O <1.80 A plus P-Cl <2.30 A.",
    "",
    "## Summary",
    "",
    "| Structure | Rw no scale | Rw dynamic | Scale | Energy (eV/atom) | P 4-fold | Si 4-fold | HS violations |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    report.append(
        f"| {LABELS[row['structure']]} | {row['Rw_no_scale']:.5f} | {row['Rw_dynamic']:.5f} | "
        f"{row['dynamic_scale']:.5f} | {row['energy_eV_atom']:.6f} | "
        f"{100*row['P_fourfold_fraction']:.2f}% | {100*row['Si_fourfold_fraction']:.2f}% | "
        f"{row['hard_sphere_violations']} |"
    )
report.extend(
    [
        "",
        "## Tetrahedral geometry",
        "",
        "| Structure | P angle RMS | Si angle RMS | P CSM | Si CSM | P volume | Si volume |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
)
for row in rows:
    report.append(
        f"| {LABELS[row['structure']]} | {row['P_angle_RMS_deg']:.4f} | "
        f"{row['Si_angle_RMS_deg']:.4f} | {row['P_CSM_percent']:.4f} | "
        f"{row['Si_CSM_percent']:.4f} | {row['P_volume_A3']:.4f} | {row['Si_volume_A3']:.4f} |"
    )
report.extend(
    [
        "",
        "## Interpretation",
        "",
        f"- Fixed-cell relaxation lowers the energy by {1000*(initial_row['energy_eV_atom']-relaxed_row['energy_eV_atom']):.3f} meV/atom.",
        f"- The bounded-dynamic PDF Rw changes from {initial_row['Rw_dynamic']:.5f} to {relaxed_row['Rw_dynamic']:.5f}.",
        f"- P total-fourfold changes from {100*initial_row['P_fourfold_fraction']:.2f}% to {100*relaxed_row['P_fourfold_fraction']:.2f}%; Si total-fourfold changes from {100*initial_row['Si_fourfold_fraction']:.2f}% to {100*relaxed_row['Si_fourfold_fraction']:.2f}%.",
        f"- Hard-sphere violations change from {initial_row['hard_sphere_violations']} to {relaxed_row['hard_sphere_violations']}.",
        "- Detailed bond distributions, ligand patterns, Zr coordination histograms, and pair minima are in metrics.json and comparison_metrics.csv.",
    ]
)
(OUT / "REPORT.md").write_text("\n".join(report) + "\n")

print(json.dumps({"output": str(OUT), "summary_rows": rows}, indent=2))
