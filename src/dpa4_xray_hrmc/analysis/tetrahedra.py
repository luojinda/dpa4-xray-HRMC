"""Fixed-center P/Si tetrahedron metrics used by the Phase 2.5 audit."""

from __future__ import annotations

import itertools

import numpy as np
from ase import Atoms

from dpa4_xray_hrmc.rmc.hard_sphere import minimum_image_vectors

IDEAL_ANGLE_DEG = 109.47122063449069
IDEAL = np.asarray([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)


def fixed_center_csm(vectors: np.ndarray) -> float:
    denominator = float(np.sum(vectors * vectors))
    best = np.inf
    for permutation in itertools.permutations(range(4)):
        ideal = IDEAL[list(permutation)]
        u, singular, vt = np.linalg.svd(ideal.T @ vectors)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        scale = float(np.sum(singular) / np.sum(ideal * ideal))
        best = min(best, float(np.sum((vectors - scale * ideal @ rotation) ** 2)))
    return 100.0 * best / denominator


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    data = np.asarray(values)
    return {"count": len(values), "mean": float(data.mean()), "std": float(data.std()), "min": float(data.min()), "max": float(data.max())}


def tetrahedral_metrics(atoms: Atoms) -> dict:
    symbols = np.asarray(atoms.get_chemical_symbols())
    cell = np.asarray(atoms.cell)
    definitions = {"P": (("O", 1.80),), "Si": (("O", 1.85), ("Cl", 2.30))}
    output = {}
    for element, pairs in definitions.items():
        centers = np.flatnonzero(symbols == element)
        bonds: dict[str, list[float]] = {f"{element}-{ligand}": [] for ligand, _ in pairs}
        angles: list[float] = []
        angle_rms: list[float] = []
        csms: list[float] = []
        volumes: list[float] = []
        tetrahedra = 0
        for center in centers:
            selected_vectors: list[np.ndarray] = []
            selected_kinds: list[str] = []
            for ligand, cutoff in pairs:
                indices = np.flatnonzero(symbols == ligand)
                vectors = minimum_image_vectors(atoms.positions[indices] - atoms.positions[center], cell)
                distances = np.linalg.norm(vectors, axis=1)
                for vector, distance in zip(vectors[distances < cutoff], distances[distances < cutoff], strict=True):
                    selected_vectors.append(vector)
                    selected_kinds.append(f"{element}-{ligand}")
            if len(selected_vectors) != 4:
                continue
            tetrahedra += 1
            vectors = np.asarray(selected_vectors)
            for kind, distance in zip(selected_kinds, np.linalg.norm(vectors, axis=1), strict=True):
                bonds[kind].append(float(distance))
            site_angles = []
            for i, j in itertools.combinations(range(4), 2):
                cosine = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
                site_angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
            angles.extend(site_angles)
            angle_rms.append(float(np.sqrt(np.mean((np.asarray(site_angles) - IDEAL_ANGLE_DEG) ** 2))))
            csms.append(fixed_center_csm(vectors))
            volumes.append(abs(float(np.linalg.det(np.stack((vectors[1]-vectors[0], vectors[2]-vectors[0], vectors[3]-vectors[0]))))) / 6.0)
        output[element] = {
            "centers": len(centers),
            "tetrahedral_count": tetrahedra,
            "tetrahedral_fraction": tetrahedra / len(centers),
            "bond_lengths_A": {key: _stats(values) for key, values in bonds.items()},
            "angles_deg": _stats(angles),
            "angle_rms_from_ideal_deg": _stats(angle_rms),
            "fixed_center_csm_percent": _stats(csms),
            "volume_A3": _stats(volumes),
        }
    return output
