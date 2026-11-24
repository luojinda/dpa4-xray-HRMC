"""Compare relaxed-MQ3 Inline HRMC runs at Teff = 50, 100, and 175 K."""

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
BASE = ROOT / "results/PZS_phase3/phase3C/PZS_MQ_3_relaxed"
RELAXED = ROOT / "results/PZS_phase3/relaxations/PZS_MQ_3_lora_fixed_cell/relaxed_structure.extxyz"
OUT = BASE / "teff_50_100_175_comparison"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = {
    50: BASE / "validation_100k_seed101_T50_d0p1_r1p4_10p0A_lora_bounded_dynamic_p_total",
    100: BASE / "validation_100k_seed101_T100_d0p1_r1p4_10p0A_lora_bounded_dynamic_p_total",
    175: BASE / "validation_100k_seed101_T175_d0p1_r1p4_10p0A_lora_bounded_dynamic_p_total",
}
COLORS = {50: "#3B6FB6", 100: "#2A9D8F", 175: "#E76F51"}
RELAXED_COLOR = "#6B7280"
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
            str(int(value)): int(count)
            for value, count in sorted(Counter(total_cn).items())
        },
        "ligand_patterns": dict(sorted(patterns.items())),
        "bond_lengths_A": {key: stats(values) for key, values in bonds.items()},
    }
    if tetrahedral:
        result.update(
            {
                "fourfold_count": int(np.sum(cn_array == 4)),
                "fourfold_fraction": float(np.mean(cn_array == 4)),
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


def histogram_mean(histogram: dict[str, int]) -> float:
    count = sum(histogram.values())
    return sum(float(value) * number for value, number in histogram.items()) / count


config = replace(
    load_phase2_config(ROOT / "configs/pzs_phase2_common.yaml"),
    structure=str(RELAXED.relative_to(ROOT)),
    fit_rmin_A=1.4,
    fit_rmax_A=10.0,
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


relaxed_atoms = read(RELAXED)
atoms = {temperature: read(path / "final_structure.extxyz") for temperature, path in RUNS.items()}
summaries = {
    temperature: json.loads((path / "summary.json").read_text())
    for temperature, path in RUNS.items()
}
traces = {
    temperature: json.loads((path / "trace.json").read_text())
    for temperature, path in RUNS.items()
}

relaxed_curve = pdf_curve(relaxed_atoms)
curves = {temperature: pdf_curve(structure) for temperature, structure in atoms.items()}
objective = ScaleObjective.from_initial(
    relaxed_curve[fit_mask], experiment.g_r_A2[fit_mask]
)
scores = {
    temperature: {
        "dynamic": objective.score(ScaleMode.BOUNDED_DYNAMIC, curve[fit_mask]),
        "no_scale": objective.score(ScaleMode.NO_SCALE, curve[fit_mask]),
        "fixed": objective.score(ScaleMode.FIXED_INITIAL, curve[fit_mask]),
    }
    for temperature, curve in curves.items()
}


def rw(score) -> float:
    return float(np.sqrt(score.loss / objective.C))


hard_sphere = HardSphereModel.from_atoms(relaxed_atoms, config.hard_sphere_scale)


def local_environment(structure: Atoms) -> dict:
    coordination = IncrementalCoordination(
        structure,
        cutoffs,
        p_coordination_mode="P-total",
        p_cl_cutoff_A=P_CL_CUTOFF_A,
    ).current_snapshot()
    return {
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


local = {"relaxed_mq3": local_environment(relaxed_atoms)}
local.update(
    {temperature: local_environment(structure) for temperature, structure in atoms.items()}
)

rows = []
for temperature in RUNS:
    summary = summaries[temperature]
    p_env = local[temperature]["P"]
    si_env = local[temperature]["Si"]
    zr_env = local[temperature]["Zr"]
    initial_energy_eV = summary["energy_eV"] - summary["delta_energy_from_MQ_eV"]
    rows.append(
        {
            "Teff_K": temperature,
            "Rw_active_reported": summary["Rw_active"],
            "Rw_dynamic_common": rw(scores[temperature]["dynamic"]),
            "Rw_no_scale_common": rw(scores[temperature]["no_scale"]),
            "Rw_fixed_common": rw(scores[temperature]["fixed"]),
            "scale_common": scores[temperature]["dynamic"].scale,
            "energy_eV_atom": summary["energy_eV_atom"],
            "delta_energy_meV_atom": 1000.0
            * (summary["energy_eV"] - initial_energy_eV)
            / len(atoms[temperature]),
            "acceptance_rate": summary["acceptance_rate"],
            "hard_sphere_rejection_rate": summary["hard_sphere_rejected"]
            / summary["attempted"],
            "P_fourfold_fraction": p_env["fourfold_fraction"],
            "Si_fourfold_fraction": si_env["fourfold_fraction"],
            "P_angle_RMS_deg": p_env["angle_rms_from_ideal_deg"]["mean"],
            "Si_angle_RMS_deg": si_env["angle_rms_from_ideal_deg"]["mean"],
            "P_CSM_percent": p_env["fixed_center_csm_percent"]["mean"],
            "Si_CSM_percent": si_env["fixed_center_csm_percent"]["mean"],
            "P_volume_A3": p_env["volume_A3"]["mean"],
            "Si_volume_A3": si_env["volume_A3"]["mean"],
            "Zr_total_CN_mean": histogram_mean(zr_env["total_cn_histogram"]),
            "hard_sphere_violations": local[temperature]["hard_sphere"]["violation_count"],
            "minimum_hard_sphere_margin_A": local[temperature]["hard_sphere"]["minimum_margin_A"],
            "wall_s": summary["wall_s"],
        }
    )

with (OUT / "final_metrics.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def local_metric_row(label: str, item: dict) -> dict:
    return {
        "structure": label,
        "P_fourfold_fraction": item["P"]["fourfold_fraction"],
        "Si_fourfold_fraction": item["Si"]["fourfold_fraction"],
        "P_angle_RMS_deg": item["P"]["angle_rms_from_ideal_deg"]["mean"],
        "Si_angle_RMS_deg": item["Si"]["angle_rms_from_ideal_deg"]["mean"],
        "P_CSM_percent": item["P"]["fixed_center_csm_percent"]["mean"],
        "Si_CSM_percent": item["Si"]["fixed_center_csm_percent"]["mean"],
        "P_volume_A3": item["P"]["volume_A3"]["mean"],
        "Si_volume_A3": item["Si"]["volume_A3"]["mean"],
        "Zr_total_CN_mean": histogram_mean(item["Zr"]["total_cn_histogram"]),
        "hard_sphere_violations": item["hard_sphere"]["violation_count"],
        "minimum_hard_sphere_margin_A": item["hard_sphere"]["minimum_margin_A"],
    }


local_rows = [local_metric_row("Relaxed MQ3", local["relaxed_mq3"])]
local_rows.extend(
    local_metric_row(f"{temperature} K", local[temperature])
    for temperature in RUNS
)
with (OUT / "local_structure_metrics.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(local_rows[0]))
    writer.writeheader()
    writer.writerows(local_rows)


(OUT / "detailed_metrics.json").write_text(
    json.dumps(
        {
            "common_scale_initial": objective.s0,
            "common_scale_bounds": [objective.lower_scale, objective.upper_scale],
            "runs": rows,
            "local": local,
        },
        indent=2,
    )
)

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})

figure, axes = plt.subplots(2, 1, figsize=(7.2, 5.7), sharex=True)
axes[0].plot(experiment.r_A, experiment.g_r_A2, color="black", lw=1.5, label="Experiment")
for temperature in RUNS:
    score = scores[temperature]["dynamic"]
    axes[0].plot(
        experiment.r_A,
        score.scale * curves[temperature],
        color=COLORS[temperature],
        lw=1.25,
        label=rf"{temperature} K ($R_w={rw(score):.3f}$, $w={score.scale:.3f}$)",
    )
    axes[1].plot(
        experiment.r_A,
        experiment.g_r_A2 - score.scale * curves[temperature],
        color=COLORS[temperature],
        lw=1.0,
        label=f"{temperature} K",
    )
axes[0].set_ylabel(r"$G(r)$ ($\mathrm{\AA}^{-2}$)")
axes[1].set_ylabel(r"$G_{exp}-G_{calc}$")
axes[1].set_xlabel(r"$r$ ($\mathrm{\AA}$)")
for axis in axes:
    axis.set_xlim(1.4, 10.0)
    axis.axhline(0.0, color="#999999", lw=0.6)
    axis.legend(frameon=False, ncol=2)
figure.tight_layout()
figure.savefig(OUT / "pdf_teff_50_100_175_1p4_10A.png", dpi=300)
figure.savefig(OUT / "pdf_teff_50_100_175_1p4_10A.pdf")
plt.close(figure)

initial_score_dynamic = objective.score(ScaleMode.BOUNDED_DYNAMIC, relaxed_curve[fit_mask])
initial_score_no_scale = objective.score(ScaleMode.NO_SCALE, relaxed_curve[fit_mask])
figure, axes = plt.subplots(2, 2, figsize=(8.0, 6.1), sharex=True)
for temperature in RUNS:
    trace = traces[temperature]
    attempts = [0] + [int(record["attempted"]) for record in trace]
    initial_energy_eV = (
        summaries[temperature]["energy_eV"]
        - summaries[temperature]["delta_energy_from_MQ_eV"]
    )
    axes[0, 0].plot(
        attempts,
        [rw(initial_score_dynamic)] + [record["Rw_active"] for record in trace],
        color=COLORS[temperature],
        label=f"{temperature} K",
    )
    axes[0, 1].plot(
        attempts,
        [rw(initial_score_no_scale)] + [record["Rw_no_scale"] for record in trace],
        color=COLORS[temperature],
    )
    axes[1, 0].plot(
        attempts,
        [0.0]
        + [
            1000.0 * (record["energy_eV"] - initial_energy_eV) / len(atoms[temperature])
            for record in trace
        ],
        color=COLORS[temperature],
    )
    axes[1, 1].plot(
        attempts,
        [objective.s0] + [record["scale_current"] for record in trace],
        color=COLORS[temperature],
    )
axes[0, 0].set_ylabel("Active Rw")
axes[0, 1].set_ylabel("No-scale Rw")
axes[1, 0].set_ylabel(r"$\Delta E$ (meV/atom)")
axes[1, 1].set_ylabel("Dynamic scale")
axes[1, 0].set_xlabel("Attempts")
axes[1, 1].set_xlabel("Attempts")
axes[0, 0].legend(frameon=False)
for axis in axes.ravel():
    axis.grid(alpha=0.2)
figure.tight_layout()
figure.savefig(OUT / "trajectories_teff_50_100_175.png", dpi=300)
figure.savefig(OUT / "trajectories_teff_50_100_175.pdf")
plt.close(figure)

structure_labels = [row["structure"] for row in local_rows]
structure_colors = [RELAXED_COLOR] + [COLORS[temperature] for temperature in RUNS]
figure, axes = plt.subplots(2, 3, figsize=(9.0, 5.5))
quantities = [
    ("P_angle_RMS_deg", "P angle RMS (deg)"),
    ("Si_angle_RMS_deg", "Si angle RMS (deg)"),
    ("P_CSM_percent", "P CSM (%)"),
    ("Si_CSM_percent", "Si CSM (%)"),
    ("Zr_total_CN_mean", "Mean Zr total CN"),
    ("minimum_hard_sphere_margin_A", "Minimum HS margin (A)"),
]
for axis, (key, ylabel) in zip(axes.ravel(), quantities, strict=True):
    axis.bar(
        structure_labels,
        [row[key] for row in local_rows],
        color=structure_colors,
    )
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=20, labelsize=8)
    axis.grid(axis="y", alpha=0.2)
figure.tight_layout()
figure.savefig(OUT / "local_structure_teff_50_100_175.png", dpi=300)
figure.savefig(OUT / "local_structure_teff_50_100_175.pdf")
plt.close(figure)

report_lines = [
    "# Relaxed MQ3 Teff comparison",
    "",
    "All runs use the same relaxed MQ3 start, seed 101, d=0.1 A, 1.4-10 A fit range, bounded-dynamic scale, DPA4 LoRA model, hard-sphere table, and coordination weights.",
    "",
    "| Teff (K) | Active Rw | No-scale Rw | Scale | E (eV/atom) | Delta E (meV/atom) | Acceptance | P 4-fold | Si 4-fold |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    report_lines.append(
        f"| {row['Teff_K']} | {row['Rw_dynamic_common']:.5f} | "
        f"{row['Rw_no_scale_common']:.5f} | {row['scale_common']:.5f} | "
        f"{row['energy_eV_atom']:.6f} | {row['delta_energy_meV_atom']:.3f} | "
        f"{100*row['acceptance_rate']:.2f}% | {100*row['P_fourfold_fraction']:.2f}% | "
        f"{100*row['Si_fourfold_fraction']:.2f}% |"
    )
(OUT / "REPORT.md").write_text("\n".join(report_lines) + "\n")

print(json.dumps({"output": str(OUT), "runs": rows}, indent=2))
