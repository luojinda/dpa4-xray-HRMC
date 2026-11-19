#!/usr/bin/env python
"""Offline MQ coordination audit using partial RDFs and CrystalNN.

CrystalNN is used only here.  The RMC hot loop uses the fixed-cutoff engine in
``rmc.coordination`` and does not import pymatgen.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dpa4_xray_hrmc_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from dpa4_xray_hrmc.rmc.coordination import CoordinationCutoffs, IncrementalCoordination
from dpa4_xray_hrmc.rmc.hard_sphere import minimum_image_vectors
from dpa4_xray_hrmc.xray_pdf.partial_rdf import compute_partial_rdfs
from dpa4_xray_hrmc.xray_pdf.structure import load_structure


AUDIT_PAIRS = (("P", "O"), ("P", "Cl"), ("Si", "O"), ("Si", "Cl"), ("Zr", "O"), ("Zr", "Cl"))
PEAK_WINDOWS = {
    "P-O": (1.0, 2.2),
    "P-Cl": (1.8, 4.2),
    "Si-O": (1.0, 2.2),
    "Si-Cl": (1.6, 3.4),
    "Zr-O": (1.5, 3.3),
    "Zr-Cl": (1.8, 3.8),
}


def pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def distance_matrix(atoms, central: str, ligand: str) -> tuple[np.ndarray, np.ndarray]:
    symbols = np.asarray(atoms.get_chemical_symbols())
    centers = np.flatnonzero(symbols == central)
    ligands = np.flatnonzero(symbols == ligand)
    matrix = np.empty((len(centers), len(ligands)), dtype=float)
    cell = np.asarray(atoms.cell, dtype=float)
    for row, center in enumerate(centers):
        matrix[row] = np.linalg.norm(
            minimum_image_vectors(atoms.positions[ligands] - atoms.positions[center], cell), axis=1
        )
    return centers, matrix


def first_shell_candidate(r: np.ndarray, g: np.ndarray, pair: str) -> dict[str, float | str]:
    smooth = gaussian_filter1d(g, sigma=2.0)
    lo, hi = PEAK_WINDOWS[pair]
    peak_mask = (r >= lo) & (r <= hi)
    peak_indices = np.flatnonzero(peak_mask)
    if not len(peak_indices):
        raise ValueError(pair)
    local_peaks, properties = find_peaks(
        smooth[peak_indices], prominence=max(0.03 * np.ptp(smooth[peak_mask]), 0.02)
    )
    if len(local_peaks):
        prominences = properties["prominences"]
        peak_index = int(peak_indices[local_peaks[int(np.argmax(prominences))]])
    else:
        peak_index = int(peak_indices[np.argmax(smooth[peak_indices])])
    minimum_mask = (r >= r[peak_index] + 0.12) & (r <= min(r[peak_index] + 1.8, 5.5))
    minimum_indices = np.flatnonzero(minimum_mask)
    local_minima, _ = find_peaks(-smooth[minimum_indices])
    if len(local_minima):
        cutoff_index = int(minimum_indices[local_minima[0]])
        method = "first smoothed RDF local minimum after dominant first-shell peak"
    else:
        cutoff_index = int(minimum_indices[np.argmin(smooth[minimum_indices])])
        method = "lowest smoothed RDF point within 1.8 A after dominant peak"
    return {
        "peak_A": float(r[peak_index]),
        "candidate_cutoff_A": float(r[cutoff_index]),
        "g_at_cutoff": float(g[cutoff_index]),
        "method": method,
    }


def distribution(values: np.ndarray) -> dict[str, object]:
    counts = Counter(int(value) for value in values)
    total = len(values)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "counts": {str(key): value for key, value in sorted(counts.items())},
        "fractions": {str(key): value / total for key, value in sorted(counts.items())},
    }


def order_statistics(matrix: np.ndarray, maximum_order: int = 12) -> list[dict[str, float | int]]:
    ordered = np.sort(matrix, axis=1)
    output = []
    for order in range(min(maximum_order, ordered.shape[1])):
        values = ordered[:, order]
        output.append(
            {
                "neighbour_order": order + 1,
                "min_A": float(np.min(values)),
                "q25_A": float(np.quantile(values, 0.25)),
                "median_A": float(np.median(values)),
                "q75_A": float(np.quantile(values, 0.75)),
                "max_A": float(np.max(values)),
            }
        )
    return output


def crystalnn_audit(atoms) -> tuple[list[dict[str, object]], dict[str, object]]:
    from pymatgen.analysis.local_env import CrystalNN
    from pymatgen.io.ase import AseAtomsAdaptor

    structure = AseAtomsAdaptor.get_structure(atoms)
    cnn = CrystalNN(
        weighted_cn=False,
        cation_anion=False,
        distance_cutoffs=(0.5, 1.0),
        x_diff_weight=3.0,
        porous_adjustment=True,
        search_cutoff=7.0,
    )
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    symbols = np.asarray(atoms.get_chemical_symbols())
    for index in np.flatnonzero(np.isin(symbols, ["P", "Si", "Zr"])):
        try:
            neighbours = cnn.get_nn_info(structure, int(index))
            ligand_counts = Counter(str(item["site"].specie.symbol) for item in neighbours)
            detail = []
            for item in neighbours:
                site = item["site"]
                detail.append(
                    {
                        "index": int(item["site_index"]),
                        "element": str(site.specie.symbol),
                        "distance_A": float(structure[int(index)].distance(site)),
                        "weight": float(item["weight"]),
                    }
                )
            rows.append(
                {
                    "atom_index": int(index),
                    "element": str(symbols[index]),
                    "O": ligand_counts["O"],
                    "Cl": ligand_counts["Cl"],
                    "total": len(neighbours),
                    "other": len(neighbours) - ligand_counts["O"] - ligand_counts["Cl"],
                    "neighbours": detail,
                }
            )
        except Exception as exc:  # audit must report per-site failures rather than hide them
            errors.append({"atom_index": int(index), "element": str(symbols[index]), "error": repr(exc)})
    summary: dict[str, object] = {
        "package_version": version("pymatgen"),
        "parameters": {
            "weighted_cn": False,
            "cation_anion": False,
            "distance_cutoffs": [0.5, 1.0],
            "x_diff_weight": 3.0,
            "porous_adjustment": True,
            "search_cutoff": 7.0,
        },
        "errors": errors,
        "by_element": {},
    }
    for element in ("P", "Si", "Zr"):
        selected = [row for row in rows if row["element"] == element]
        summary["by_element"][element] = {
            "O": distribution(np.asarray([row["O"] for row in selected], dtype=int)),
            "Cl": distribution(np.asarray([row["Cl"] for row in selected], dtype=int)),
            "total": distribution(np.asarray([row["total"] for row in selected], dtype=int)),
        }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", default="PDF/PZS_quench_222.cif")
    parser.add_argument("--output-dir", default="results/PZS_phase2_5/audit")
    parser.add_argument("--cutoffs-json", help="Optional JSON mapping of five frozen pair cutoffs")
    parser.add_argument("--skip-crystalnn", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    structure_path = (root / args.structure).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    atoms = load_structure(structure_path).atoms
    rdf = compute_partial_rdfs(atoms, 6.0, 0.02)
    radial_grid = np.arange(0.8, 6.0001, 0.01)

    pair_data: dict[str, dict[str, object]] = {}
    distance_matrices: dict[str, np.ndarray] = {}
    centers_by_pair: dict[str, np.ndarray] = {}
    for central, ligand in AUDIT_PAIRS:
        name = f"{central}-{ligand}"
        centers, matrix = distance_matrix(atoms, central, ligand)
        centers_by_pair[name] = centers
        distance_matrices[name] = matrix
        g = rdf.g[pair_key(central, ligand)]
        candidate = first_shell_candidate(rdf.r_A, g, name)
        cumulative = np.mean(matrix[:, :, None] < radial_grid[None, None, :], axis=(0, 1)) * matrix.shape[1]
        pair_data[name] = {
            **candidate,
            "order_statistics": order_statistics(matrix),
            "mean_cumulative_at_candidate": float(
                np.mean(np.count_nonzero(matrix < float(candidate["candidate_cutoff_A"]), axis=1))
            ),
        }
        np.savetxt(
            output / f"{name.replace('-', '_')}_rdf_cumulative.dat",
            np.column_stack(
                (
                    rdf.r_A,
                    g,
                    np.interp(rdf.r_A, radial_grid, cumulative),
                )
            ),
            header="r_A partial_g mean_cumulative_CN",
        )

    automatic = {
        name: float(pair_data[name]["candidate_cutoff_A"])
        for name in ("P-O", "Si-O", "Si-Cl", "Zr-O", "Zr-Cl")
    }
    if args.cutoffs_json:
        selected_cutoffs = json.loads(Path(args.cutoffs_json).read_text(encoding="utf-8"))
        cutoff_source = str(Path(args.cutoffs_json).resolve())
    else:
        selected_cutoffs = automatic
        cutoff_source = "automatic RDF first-minimum candidates; not yet frozen"
    cutoffs = CoordinationCutoffs.from_mapping(selected_cutoffs)
    engine = IncrementalCoordination(atoms, cutoffs)
    by_site_rows = []
    for local, atom_index in enumerate(engine.indices["P"]):
        by_site_rows.append({"atom_index": int(atom_index), "element": "P", "O": int(engine.p_o[local]), "Cl": 0, "total": int(engine.p_o[local])})
    for local, atom_index in enumerate(engine.indices["Si"]):
        by_site_rows.append({"atom_index": int(atom_index), "element": "Si", "O": int(engine.si_o[local]), "Cl": int(engine.si_cl[local]), "total": int(engine.si_o[local] + engine.si_cl[local])})
    for local, atom_index in enumerate(engine.indices["Zr"]):
        by_site_rows.append({"atom_index": int(atom_index), "element": "Zr", "O": int(engine.zr_o[local]), "Cl": int(engine.zr_cl[local]), "total": int(engine.zr_o[local] + engine.zr_cl[local])})
    with (output / "fixed_cutoff_coordination_by_site.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("atom_index", "element", "O", "Cl", "total"))
        writer.writeheader()
        writer.writerows(by_site_rows)

    fixed_summary = {}
    for element in ("P", "Si", "Zr"):
        selected = [row for row in by_site_rows if row["element"] == element]
        fixed_summary[element] = {
            field: distribution(np.asarray([row[field] for row in selected], dtype=int))
            for field in ("O", "Cl", "total")
        }
    crystal_rows: list[dict[str, object]] = []
    crystal_summary: dict[str, object] | None = None
    if not args.skip_crystalnn:
        crystal_rows, crystal_summary = crystalnn_audit(atoms)
        with (output / "crystalnn_coordination_by_site.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("atom_index", "element", "O", "Cl", "total", "other", "neighbours_json"))
            writer.writeheader()
            for row in crystal_rows:
                writer.writerow({**{key: row[key] for key in ("atom_index", "element", "O", "Cl", "total", "other")}, "neighbours_json": json.dumps(row["neighbours"])})

    sensitivity: dict[str, object] = {}
    for name, cutoff in selected_cutoffs.items():
        matrix = distance_matrices[name]
        sensitivity[name] = {
            f"{delta:+.2f}_A": distribution(np.count_nonzero(matrix < cutoff + delta, axis=1))
            for delta in (-0.10, -0.05, 0.0, 0.05, 0.10)
            if cutoff + delta > 0.0
        }

    report = {
        "structure": str(structure_path),
        "formula": atoms.get_chemical_formula(mode="hill"),
        "n_atoms": len(atoms),
        "pair_audit": pair_data,
        "automatic_cutoff_candidates_A": automatic,
        "selected_cutoffs_A": cutoffs.as_mapping(),
        "cutoff_source": cutoff_source,
        "fixed_cutoff_summary": fixed_summary,
        "sensitivity": sensitivity,
        "crystalnn": crystal_summary,
        "gate_checks": {
            "P_O_fourfold_fraction": fixed_summary["P"]["O"]["fractions"].get("4", 0.0),
            "Si_total_fourfold_fraction": fixed_summary["Si"]["total"]["fractions"].get("4", 0.0),
            "Si_all_64_fourfold": fixed_summary["Si"]["total"]["counts"].get("4", 0) == 64,
        },
    }
    (output / "coordination_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    colors = {"P": "#1b9e77", "Si": "#7570b3", "Zr": "#d95f02"}
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.5), constrained_layout=True)
    for axis, (central, ligand) in zip(axes.flat, AUDIT_PAIRS, strict=True):
        name = f"{central}-{ligand}"
        g = rdf.g[pair_key(central, ligand)]
        matrix = distance_matrices[name]
        cumulative = np.mean(np.count_nonzero(matrix[:, :, None] < rdf.r_A[None, None, :], axis=1), axis=0)
        axis.plot(rdf.r_A, g, color=colors[central], label="partial RDF")
        axis2 = axis.twinx()
        axis2.plot(rdf.r_A, cumulative, color="0.25", linestyle="--", label="mean cumulative CN")
        candidate = float(pair_data[name]["candidate_cutoff_A"])
        axis.axvline(candidate, color="black", linestyle=":", linewidth=1.0, label=f"candidate {candidate:.2f} A")
        if name in selected_cutoffs and not np.isclose(candidate, selected_cutoffs[name]):
            axis.axvline(selected_cutoffs[name], color="red", linestyle="-.", linewidth=1.0, label=f"selected {selected_cutoffs[name]:.2f} A")
        axis.set_xlim(0.8, 5.0)
        axis.set_title(name)
        axis.set_xlabel("r (A)")
        axis.set_ylabel("g(r)")
        axis2.set_ylabel("mean cumulative CN")
        axis.grid(alpha=0.2)
        lines, labels = axis.get_legend_handles_labels()
        axis.legend(lines, labels, frameon=False, fontsize=7)
    fig.savefig(output / "rdf_and_cumulative_coordination.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(10.5, 10.5), constrained_layout=True)
    for axis, (central, ligand) in zip(axes.flat, AUDIT_PAIRS, strict=True):
        name = f"{central}-{ligand}"
        stats = pair_data[name]["order_statistics"]
        order = np.asarray([row["neighbour_order"] for row in stats])
        median = np.asarray([row["median_A"] for row in stats])
        low = np.asarray([row["q25_A"] for row in stats])
        high = np.asarray([row["q75_A"] for row in stats])
        axis.plot(order, median, marker="o", color=colors[central])
        axis.fill_between(order, low, high, color=colors[central], alpha=0.25, label="25-75%")
        axis.axhline(float(pair_data[name]["candidate_cutoff_A"]), color="black", linestyle=":", label="candidate cutoff")
        axis.set_title(name)
        axis.set_xlabel("neighbour order")
        axis.set_ylabel("distance (A)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=7)
    fig.savefig(output / "ordered_neighbour_distances.png", dpi=300)
    plt.close(fig)
    print(json.dumps(report["gate_checks"], indent=2))
    print(json.dumps({"automatic": automatic, "selected": cutoffs.as_mapping()}, indent=2))


if __name__ == "__main__":
    main()
