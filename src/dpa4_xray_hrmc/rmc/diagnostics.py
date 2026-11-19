"""Post-run structural diagnostics that never enter the RMC objective."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from ase import Atoms
from scipy.signal import find_peaks

from dpa4_xray_hrmc.xray_pdf.partial_rdf import compute_partial_rdfs, element_pairs

from .hard_sphere import HardSphereModel, minimum_image_vectors


DIAGNOSTIC_PAIRS: tuple[tuple[str, str], ...] = (
    ("P", "O"),
    ("Si", "O"),
    ("Zr", "O"),
    ("Zr", "Cl"),
    ("Li", "O"),
    ("Li", "Cl"),
)


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def coordination_cutoffs_from_initial(
    atoms: Atoms,
    requested_pairs: tuple[tuple[str, str], ...] = DIAGNOSTIC_PAIRS,
    rmax_A: float = 6.0,
    dr_A: float = 0.02,
) -> dict[str, dict[str, float | str]]:
    """Choose one fixed first-shell cutoff from smoothed initial partial RDFs."""

    rdf = compute_partial_rdfs(atoms, rmax_A, dr_A)
    output: dict[str, dict[str, float | str]] = {}
    smoothing_width = 5
    smoothing = np.ones(smoothing_width) / smoothing_width
    for central, neighbour in requested_pairs:
        pair = _pair_key(central, neighbour)
        values = rdf.g[pair]
        smooth = np.convolve(values, smoothing, mode="same")
        search = (rdf.r_A >= 1.0) & (rdf.r_A <= 4.2)
        candidate_indices = np.flatnonzero(search)
        peaks, _ = find_peaks(smooth[candidate_indices], prominence=max(0.05 * np.ptp(smooth[search]), 0.05))
        if len(peaks):
            peak_index = int(candidate_indices[peaks[0]])
        else:
            peak_index = int(candidate_indices[np.argmax(smooth[search])])
        minimum_search = (
            (rdf.r_A >= rdf.r_A[peak_index] + 0.15)
            & (rdf.r_A <= min(rdf.r_A[peak_index] + 1.5, rmax_A - 0.1))
        )
        minimum_indices = np.flatnonzero(minimum_search)
        minimum_values = smooth[minimum_indices]
        floor = float(np.min(minimum_values))
        peak_height = float(smooth[peak_index])
        near_floor = floor + 0.02 * max(peak_height - floor, 0.0)
        cutoff_index = int(minimum_indices[np.flatnonzero(minimum_values <= near_floor)[0]])
        method = "first entry into the near-minimum RDF plateau after the first peak"
        output[f"{central}-{neighbour}"] = {
            "peak_A": float(rdf.r_A[peak_index]),
            "cutoff_A": float(rdf.r_A[cutoff_index]),
            "method": method,
        }
    return output


def _coordination_counts(atoms: Atoms, central: str, neighbour: str, cutoff_A: float) -> tuple[np.ndarray, np.ndarray]:
    symbols = np.asarray(atoms.get_chemical_symbols())
    central_indices = np.flatnonzero(symbols == central)
    neighbour_indices = np.flatnonzero(symbols == neighbour)
    counts = np.zeros(len(central_indices), dtype=int)
    distances_in_shell: list[float] = []
    cell = np.asarray(atoms.cell, dtype=float)
    for row, central_index in enumerate(central_indices):
        selected = neighbour_indices[neighbour_indices != central_index]
        vectors = minimum_image_vectors(atoms.positions[selected] - atoms.positions[central_index], cell)
        distances = np.linalg.norm(vectors, axis=1)
        shell = distances <= cutoff_A
        counts[row] = int(np.count_nonzero(shell))
        distances_in_shell.extend(distances[shell].tolist())
    return counts, np.asarray(distances_in_shell)


def _pair_minima(atoms: Atoms) -> dict[str, float]:
    symbols = np.asarray(atoms.get_chemical_symbols())
    elements = sorted(set(symbols))
    minima = {pair: np.inf for pair in element_pairs(elements)}
    cell = np.asarray(atoms.cell, dtype=float)
    for index in range(len(atoms) - 1):
        vectors = minimum_image_vectors(atoms.positions[index + 1 :] - atoms.positions[index], cell)
        distances = np.linalg.norm(vectors, axis=1)
        for other, distance in enumerate(distances, start=index + 1):
            pair = _pair_key(str(symbols[index]), str(symbols[other]))
            minima[pair] = min(minima[pair], float(distance))
    return {f"{a}-{b}": value for (a, b), value in minima.items()}


def structure_diagnostics(
    initial_atoms: Atoms,
    trial_atoms: Atoms,
    hard_sphere: HardSphereModel,
    coordination_cutoffs: dict[str, dict[str, float | str]],
) -> dict[str, Any]:
    if len(initial_atoms) != len(trial_atoms):
        raise ValueError("Initial and trial structures have different atom counts")
    if initial_atoms.get_chemical_symbols() != trial_atoms.get_chemical_symbols():
        raise ValueError("Initial and trial structures have different species order")
    cell = np.asarray(initial_atoms.cell, dtype=float)
    if not np.allclose(cell, np.asarray(trial_atoms.cell, dtype=float), atol=1e-12):
        raise ValueError("Cell changed during fixed-cell RMC")
    displacement_vectors = minimum_image_vectors(trial_atoms.positions - initial_atoms.positions, cell)
    displacement = np.linalg.norm(displacement_vectors, axis=1)
    symbols = np.asarray(initial_atoms.get_chemical_symbols())
    by_species = {
        element: {
            "mean_A": float(np.mean(displacement[symbols == element])),
            "rms_A": float(np.sqrt(np.mean(displacement[symbols == element] ** 2))),
            "max_A": float(np.max(displacement[symbols == element])),
        }
        for element in sorted(set(symbols))
    }
    pair_minimum = _pair_minima(trial_atoms)
    violations = {
        f"{a}-{b}": int(pair_minimum[f"{a}-{b}"] < hard_sphere.cutoffs_A[(a, b)] - 1e-10)
        for a, b in hard_sphere.pairs
    }
    coordination: dict[str, Any] = {}
    for name, details in coordination_cutoffs.items():
        central, neighbour = name.split("-")
        cutoff = float(details["cutoff_A"])
        counts, bond_distances = _coordination_counts(trial_atoms, central, neighbour, cutoff)
        histogram = {str(value): int(np.count_nonzero(counts == value)) for value in np.unique(counts)}
        coordination[name] = {
            "cutoff_A": cutoff,
            "count": len(counts),
            "mean_CN": float(np.mean(counts)),
            "std_CN": float(np.std(counts)),
            "min_CN": int(np.min(counts)),
            "max_CN": int(np.max(counts)),
            "distribution": histogram,
            "shell_distance_mean_A": float(np.mean(bond_distances)) if len(bond_distances) else None,
            "shell_distance_std_A": float(np.std(bond_distances)) if len(bond_distances) else None,
        }
        if central in {"P", "Si"} and neighbour == "O":
            coordination[name]["fourfold_fraction"] = float(np.mean(counts == 4))
    return {
        "n_atoms": len(trial_atoms),
        "formula": trial_atoms.get_chemical_formula(mode="hill"),
        "volume_A3": float(trial_atoms.get_volume()),
        "density_A-3": len(trial_atoms) / float(trial_atoms.get_volume()),
        "cell_A": cell.tolist(),
        "positions_all_finite": bool(np.all(np.isfinite(trial_atoms.positions))),
        "pair_minimum_A": pair_minimum,
        "hard_sphere_violations_by_pair": violations,
        "hard_sphere_violation_total": int(sum(violations.values())),
        "displacement_from_initial": {
            "mean_A": float(np.mean(displacement)),
            "rms_A": float(np.sqrt(np.mean(displacement**2))),
            "max_A": float(np.max(displacement)),
            "by_species": by_species,
        },
        "coordination": coordination,
    }
