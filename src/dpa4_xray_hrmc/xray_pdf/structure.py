"""CIF loading and explicit periodic-structure validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read


@dataclass(frozen=True)
class StructureInfo:
    atoms: Atoms
    source_path: str
    formula: str
    element_counts: dict[str, int]
    n_atoms: int
    cell_A: list[list[float]]
    cell_lengths_A: list[float]
    cell_angles_deg: list[float]
    volume_A3: float
    number_density_A3: float
    pbc: list[bool]
    occupancies: dict[str, float]
    minimum_periodic_separation_A: float

    def serializable(self) -> dict[str, object]:
        return {key: value for key, value in self.__dict__.items() if key != "atoms"}


def _occupancies_from_cif(path: Path) -> dict[str, float]:
    """Read label/occupancy pairs from the atom loop for validation only."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    tags: list[str] = []
    in_loop = False
    collecting_tags = False
    result: dict[str, float] = {}
    for raw in lines:
        line = raw.strip()
        if line == "loop_":
            in_loop = True
            collecting_tags = True
            tags = []
            continue
        if in_loop and collecting_tags and line.startswith("_"):
            tags.append(line.split()[0])
            continue
        if in_loop and tags and "_atom_site_label" in tags and line and not line.startswith("#"):
            collecting_tags = False
            if line.startswith("_") or line == "loop_":
                break
            fields = line.split()
            if len(fields) < len(tags):
                continue
            label_idx = tags.index("_atom_site_label")
            occ_idx = tags.index("_atom_site_occupancy")
            result[fields[label_idx]] = float(fields[occ_idx])
        elif in_loop and tags and not collecting_tags and (not line or line.startswith("_") or line == "loop_"):
            if "_atom_site_label" in tags:
                break
    return result


def _minimum_periodic_separation(atoms: Atoms) -> float:
    """Return the shortest non-self MIC distance, used to catch duplicates."""
    positions = atoms.get_positions()
    cell = np.asarray(atoms.cell)
    inv_cell = np.linalg.inv(cell)
    minimum = np.inf
    block = 128
    for start in range(0, len(atoms), block):
        delta = positions[None, :, :] - positions[start : start + block, None, :]
        frac = delta @ inv_cell
        frac -= np.rint(frac)
        dist = np.linalg.norm(frac @ cell, axis=2)
        rows = np.arange(start, min(start + block, len(atoms))) - start
        dist[rows, np.arange(start, min(start + block, len(atoms)))] = np.inf
        minimum = min(minimum, float(np.min(dist)))
    return minimum


def load_structure(path: str | Path) -> StructureInfo:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Structure not found: {source}")
    atoms = read(source, format="cif")
    if not isinstance(atoms, Atoms):
        raise TypeError("ASE did not return a single Atoms object")
    if len(atoms) == 0:
        raise ValueError("Structure contains no atoms")
    if not np.all(atoms.pbc):
        raise ValueError("A fully periodic structure is required")
    if atoms.cell.rank != 3 or atoms.get_volume() <= 0:
        raise ValueError("A non-singular 3D cell is required")

    occupancies = _occupancies_from_cif(source)
    if occupancies and any(abs(value - 1.0) > 1e-12 for value in occupancies.values()):
        raise ValueError("Partial occupancy/disorder is not supported in Phase 1")
    if occupancies and len(occupancies) != len(atoms):
        raise ValueError(
            f"CIF has {len(occupancies)} explicit sites but ASE returned {len(atoms)} atoms; "
            "review symmetry expansion before calculation"
        )

    minimum = _minimum_periodic_separation(atoms)
    if minimum < 1e-4:
        raise ValueError(f"Coincident periodic sites detected ({minimum:.3g} A)")
    counts = dict(sorted(Counter(atoms.get_chemical_symbols()).items()))
    cellpar = atoms.cell.cellpar()
    volume = float(atoms.get_volume())
    return StructureInfo(
        atoms=atoms,
        source_path=str(source),
        formula=atoms.get_chemical_formula(mode="hill"),
        element_counts=counts,
        n_atoms=len(atoms),
        cell_A=np.asarray(atoms.cell).tolist(),
        cell_lengths_A=cellpar[:3].tolist(),
        cell_angles_deg=cellpar[3:].tolist(),
        volume_A3=volume,
        number_density_A3=len(atoms) / volume,
        pbc=[bool(x) for x in atoms.pbc],
        occupancies=occupancies,
        minimum_periodic_separation_A=minimum,
    )
