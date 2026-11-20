"""Compare the Phase 2.5 audit CSM/volume with get_site_info_v3.py methods.

The reference script obtains tetrahedral CSM from pymatgen LocalGeometryFinder
(`T:4`) and volume from scipy.spatial.ConvexHull.  This test feeds the exact
same four periodic neighbor coordinates to those routines and to the audit's
explicit Kabsch CSM / determinant volume implementation.
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np
from ase.io import read
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
    LocalGeometryFinder,
)
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.spatial import ConvexHull


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results/PZS_phase2_5/coordination_geometry_audit"
IDEAL_TETRAHEDRON = np.asarray(
    [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
)


def minimum_image_vectors(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    fractional = delta @ np.linalg.inv(cell)
    fractional -= np.rint(fractional)
    return fractional @ cell


def tetrahedron_neighbors(atoms, center_index: int) -> tuple[list[int], np.ndarray]:
    symbols = np.asarray(atoms.get_chemical_symbols())
    center = symbols[center_index]
    definitions = {
        "P": (("O", 1.80),),
        "Si": (("O", 1.85), ("Cl", 2.30)),
    }
    indices: list[int] = []
    vectors: list[np.ndarray] = []
    for ligand, cutoff in definitions[center]:
        candidates = np.flatnonzero(symbols == ligand)
        candidate_vectors = minimum_image_vectors(
            atoms.positions[candidates] - atoms.positions[center_index], np.asarray(atoms.cell)
        )
        distances = np.linalg.norm(candidate_vectors, axis=1)
        for index, vector in zip(candidates[distances < cutoff], candidate_vectors[distances < cutoff]):
            indices.append(int(index))
            vectors.append(vector)
    return indices, np.asarray(vectors)


def audit_csm(vectors: np.ndarray) -> float:
    """CSM definition used in the existing coordination geometry audit."""
    denominator = float(np.sum(vectors * vectors))
    best = np.inf
    for permutation in itertools.permutations(range(4)):
        ideal = IDEAL_TETRAHEDRON[list(permutation)]
        u, singular_values, vt = np.linalg.svd(ideal.T @ vectors)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        scale = float(np.sum(singular_values) / np.sum(ideal * ideal))
        residual = vectors - scale * ideal @ rotation
        best = min(best, float(np.sum(residual * residual)))
    return 100.0 * best / denominator


def pymatgen_csms(atoms, center_index: int, neighbor_coords: np.ndarray) -> dict[str, float]:
    structure = AseAtomsAdaptor.get_structure(atoms)
    finder = LocalGeometryFinder(only_symbols=["T:4"])
    finder.setup_structure(structure=structure)
    finder.setup_local_geometry(isite=center_index, coords=neighbor_coords.tolist())
    result = finder.get_coordination_symmetry_measures()["T:4"]
    return {
        key: float(result[key])
        for key in (
            "csm",
            "csm_wocs_ctwocc",
            "csm_wocs_ctwcc",
            "csm_wocs_csc",
            "csm_wcs_ctwocc",
            "csm_wcs_ctwcc",
            "csm_wcs_csc",
        )
    }


def determinant_volume(coords: np.ndarray) -> float:
    return abs(float(np.linalg.det(np.stack((coords[1] - coords[0], coords[2] - coords[0], coords[3] - coords[0]))))) / 6.0


def audit_rows() -> dict[tuple[str, str, int, int, str], dict[str, str]]:
    rows = {}
    with (AUDIT / "tetrahedra_by_site.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["stage"], row["mode"], int(row["seed"]), int(row["atom_index"]), row["element"])
            rows[key] = row
    return rows


def main() -> None:
    structures = [
        ("initial_MQ", "initial", 0, ROOT / "PDF/PZS_quench_222.cif"),
        ("Phase2", "no_scale", 101, ROOT / "results/PZS_phase2/formal_1M/no_scale/seed_101/final_structure.extxyz"),
        ("Phase2.5", "no_scale", 101, ROOT / "results/PZS_phase2_5/formal_1M/no_scale/seed_101/final_structure.extxyz"),
        ("Phase2.5", "fixed_initial", 303, ROOT / "results/PZS_phase2_5/formal_1M/fixed_initial/seed_303/final_structure.extxyz"),
    ]
    expected = audit_rows()
    results = []
    for stage, mode, seed, path in structures:
        atoms = read(path)
        symbols = np.asarray(atoms.get_chemical_symbols())
        for element in ("P", "Si"):
            selected = 0
            for center_index in np.flatnonzero(symbols == element):
                _, vectors = tetrahedron_neighbors(atoms, int(center_index))
                if len(vectors) != 4:
                    continue
                neighbor_coords = atoms.positions[center_index] + vectors
                key = (stage, mode, seed, int(center_index), element)
                row = expected[key]
                pymatgen_result = pymatgen_csms(atoms, int(center_index), neighbor_coords)
                result = {
                    "stage": stage,
                    "mode": mode,
                    "seed": seed,
                    "atom_index": int(center_index),
                    "element": element,
                    "audit_csv_csm": float(row["csm_percent"]),
                    "audit_recomputed_csm": audit_csm(vectors),
                    "pymatgen_csm_default": pymatgen_result["csm"],
                    "pymatgen_csm_fixed_center": pymatgen_result["csm_wocs_csc"],
                    "pymatgen_csm_all_definitions": pymatgen_result,
                    "audit_csv_volume_A3": float(row["volume_A3"]),
                    "determinant_volume_A3": determinant_volume(neighbor_coords),
                    "convex_hull_volume_A3": float(ConvexHull(neighbor_coords).volume),
                }
                result["audit_minus_pymatgen_default_csm"] = (
                    result["audit_recomputed_csm"] - result["pymatgen_csm_default"]
                )
                result["audit_minus_pymatgen_fixed_center_csm"] = (
                    result["audit_recomputed_csm"] - result["pymatgen_csm_fixed_center"]
                )
                result["det_minus_hull_volume_A3"] = result["determinant_volume_A3"] - result["convex_hull_volume_A3"]
                results.append(result)
                selected += 1
                if selected == 2:
                    break

    output = AUDIT / "pymatgen_csm_volume_examples.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

    assert max(abs(r["audit_csv_csm"] - r["audit_recomputed_csm"]) for r in results) < 1e-10
    assert max(abs(r["audit_minus_pymatgen_fixed_center_csm"]) for r in results) < 1e-10
    assert max(abs(r["audit_csv_volume_A3"] - r["determinant_volume_A3"]) for r in results) < 1e-10
    assert max(abs(r["det_minus_hull_volume_A3"]) for r in results) < 1e-10


if __name__ == "__main__":
    main()
