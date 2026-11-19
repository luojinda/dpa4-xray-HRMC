"""Pair-specific hard-sphere feasibility checks for fixed periodic cells."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from dpa4_xray_hrmc.xray_pdf.partial_rdf import Pair, element_pairs


def minimum_image_vectors(vectors_A: np.ndarray, cell_A: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(cell_A)
    fractional = np.asarray(vectors_A, dtype=float) @ inverse
    fractional -= np.rint(fractional)
    return fractional @ cell_A


@dataclass(frozen=True)
class HardSphereModel:
    pairs: tuple[Pair, ...]
    minimum_input_A: dict[Pair, float]
    cutoffs_A: dict[Pair, float]
    scale: float

    @classmethod
    def from_atoms(cls, atoms: Atoms, scale: float = 0.95) -> "HardSphereModel":
        if not 0.0 < scale <= 1.0:
            raise ValueError("Hard-sphere scale must be in (0, 1]")
        symbols = np.asarray(atoms.get_chemical_symbols())
        elements = sorted(set(symbols))
        pairs = tuple(element_pairs(elements))
        minima = {pair: np.inf for pair in pairs}
        cell = np.asarray(atoms.cell, dtype=float)
        positions = atoms.positions
        for index in range(len(atoms) - 1):
            vectors = minimum_image_vectors(positions[index + 1 :] - positions[index], cell)
            distances = np.linalg.norm(vectors, axis=1)
            for offset, distance in enumerate(distances, start=index + 1):
                pair = tuple(sorted((str(symbols[index]), str(symbols[offset]))))
                minima[pair] = min(minima[pair], float(distance))  # type: ignore[index]
        missing = [pair for pair, value in minima.items() if not np.isfinite(value)]
        if missing:
            raise ValueError(f"No input distances available for pairs: {missing}")
        cutoffs = {pair: scale * value for pair, value in minima.items()}
        return cls(pairs, minima, cutoffs, float(scale))

    def check_move(self, atoms: Atoms, atom_index: int, new_position_A: np.ndarray) -> tuple[bool, float, Pair | None]:
        symbols = np.asarray(atoms.get_chemical_symbols())
        other_indices = np.delete(np.arange(len(atoms)), atom_index)
        vectors = minimum_image_vectors(
            atoms.positions[other_indices] - np.asarray(new_position_A, dtype=float),
            np.asarray(atoms.cell, dtype=float),
        )
        distances = np.linalg.norm(vectors, axis=1)
        smallest_margin = np.inf
        limiting_pair: Pair | None = None
        for other_index, distance in zip(other_indices, distances, strict=True):
            pair = tuple(sorted((str(symbols[atom_index]), str(symbols[other_index]))))
            cutoff = self.cutoffs_A[pair]  # type: ignore[index]
            margin = float(distance - cutoff)
            if margin < smallest_margin:
                smallest_margin = margin
                limiting_pair = pair  # type: ignore[assignment]
            if margin < 0.0:
                return False, smallest_margin, limiting_pair
        return True, smallest_margin, limiting_pair

    def serializable(self) -> dict[str, object]:
        return {
            "scale": self.scale,
            "pairs": {
                f"{a}-{b}": {
                    "minimum_input_A": self.minimum_input_A[(a, b)],
                    "hard_cutoff_A": self.cutoffs_A[(a, b)],
                }
                for a, b in self.pairs
            },
        }
