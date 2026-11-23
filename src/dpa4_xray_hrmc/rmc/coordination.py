"""Fixed-cutoff full and incremental coordination constraints for Phase 2.5.

CrystalNN and ChemEnv deliberately do not appear in this module.  They are
offline audit tools; the hot RMC loop uses only transparent pair cutoffs and
minimum-image distances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from ase import Atoms

from .hard_sphere import minimum_image_vectors
from .incremental_pdf import wrap_cartesian


PAIR_NAMES = ("P-O", "Si-O", "Si-Cl", "Zr-O", "Zr-Cl")


@dataclass(frozen=True)
class CoordinationCutoffs:
    p_o_A: float
    si_o_A: float
    si_cl_A: float
    zr_o_A: float
    zr_cl_A: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "CoordinationCutoffs":
        missing = [name for name in PAIR_NAMES if name not in values]
        if missing:
            raise ValueError(f"Missing coordination cutoffs: {missing}")
        cutoffs = cls(*(float(values[name]) for name in PAIR_NAMES))
        if any(value <= 0.0 for value in cutoffs.as_mapping().values()):
            raise ValueError("Coordination cutoffs must be positive")
        return cutoffs

    def as_mapping(self) -> dict[str, float]:
        return {
            "P-O": self.p_o_A,
            "Si-O": self.si_o_A,
            "Si-Cl": self.si_cl_A,
            "Zr-O": self.zr_o_A,
            "Zr-Cl": self.zr_cl_A,
        }


@dataclass(frozen=True)
class CoordinationComponents:
    p: float
    si: float
    zr_total: float
    zr_o: float

    def weighted(self, weights: Mapping[str, float]) -> float:
        return float(
            weights["P"] * self.p
            + weights["Si"] * self.si
            + weights["Zr_total"] * self.zr_total
            + weights["Zr_O"] * self.zr_o
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "C_P": self.p,
            "C_Si": self.si,
            "C_Zr_total": self.zr_total,
            "C_Zr_O": self.zr_o,
        }


@dataclass(frozen=True)
class CoordinationSnapshot:
    p_o: np.ndarray
    si_o: np.ndarray
    si_cl: np.ndarray
    zr_o: np.ndarray
    zr_cl: np.ndarray
    zr_hist_total: np.ndarray
    zr_hist_o: np.ndarray
    components: CoordinationComponents

    @property
    def si_total(self) -> np.ndarray:
        return self.si_o + self.si_cl

    @property
    def zr_total(self) -> np.ndarray:
        return self.zr_o + self.zr_cl


@dataclass(frozen=True)
class CoordinationProposal:
    atom_index: int
    old_position_A: np.ndarray
    new_position_A: np.ndarray
    trial: CoordinationSnapshot


def _histogram(values: np.ndarray, bins: int) -> np.ndarray:
    if np.any(values < 0) or np.any(values >= bins):
        raise ValueError(f"Coordination value outside configured bins [0, {bins - 1}]")
    return np.bincount(values, minlength=bins).astype(np.int64, copy=False)


def _distribution_epsilon(histogram: np.ndarray) -> np.ndarray:
    """Binomial sampling scale with a one-Zr-count finite floor.

    This is an explicit modelling tolerance, not experimental uncertainty.
    Empty and fully occupied bins receive epsilon=1/N; other bins receive
    max(sqrt(p(1-p)/N), 1/N).
    """

    count = int(np.sum(histogram))
    if count <= 0:
        raise ValueError("Empty coordination histogram")
    probability = histogram / count
    return np.maximum(np.sqrt(probability * (1.0 - probability) / count), 1.0 / count)


class IncrementalCoordination:
    """Transactional incremental coordination state for one fixed-cell structure."""

    def __init__(
        self,
        atoms: Atoms,
        cutoffs: CoordinationCutoffs,
        histogram_bins: int = 17,
        p_coordination_mode: str = "P-O",
        p_cl_cutoff_A: float | None = None,
    ) -> None:
        if not np.all(atoms.pbc):
            raise ValueError("Coordination engine requires three-dimensional PBC")
        if histogram_bins < 2:
            raise ValueError("histogram_bins must be at least two")
        self.atoms = atoms.copy()
        self.cutoffs = cutoffs
        self.histogram_bins = int(histogram_bins)
        if p_coordination_mode not in {"P-O", "P-total"}:
            raise ValueError("p_coordination_mode must be 'P-O' or 'P-total'")
        if p_coordination_mode == "P-total" and (
            p_cl_cutoff_A is None or p_cl_cutoff_A <= 0.0
        ):
            raise ValueError("P-total coordination requires a positive P-Cl cutoff")
        self.p_coordination_mode = p_coordination_mode
        self.p_cl_cutoff_A = (
            None if p_cl_cutoff_A is None else float(p_cl_cutoff_A)
        )
        self.cell_A = np.asarray(self.atoms.cell, dtype=float)
        self.symbols = np.asarray(self.atoms.get_chemical_symbols())
        self.indices = {
            element: np.flatnonzero(self.symbols == element).astype(np.int64)
            for element in ("P", "Si", "Zr", "O", "Cl")
        }
        for element in self.indices:
            if not len(self.indices[element]):
                raise ValueError(f"Structure has no {element} atoms")
        initial = self._direct_snapshot(self.atoms)
        self.p_o = initial.p_o.copy()
        self.si_o = initial.si_o.copy()
        self.si_cl = initial.si_cl.copy()
        self.zr_o = initial.zr_o.copy()
        self.zr_cl = initial.zr_cl.copy()
        self.zr_hist_total = initial.zr_hist_total.copy()
        self.zr_hist_o = initial.zr_hist_o.copy()
        self.target_zr_hist_total = initial.zr_hist_total.copy()
        self.target_zr_hist_o = initial.zr_hist_o.copy()
        self.epsilon_zr_total = _distribution_epsilon(self.target_zr_hist_total)
        self.epsilon_zr_o = _distribution_epsilon(self.target_zr_hist_o)
        self.components = self._components(
            self.p_o, self.si_o, self.si_cl, self.zr_hist_total, self.zr_hist_o
        )

    def _pair_counts(
        self, atoms: Atoms, center: str, ligand: str, cutoff_A: float
    ) -> np.ndarray:
        centers = self.indices[center]
        ligands = self.indices[ligand]
        result = np.zeros(len(centers), dtype=np.int64)
        for local_index, center_index in enumerate(centers):
            vectors = minimum_image_vectors(
                atoms.positions[ligands] - atoms.positions[center_index], self.cell_A
            )
            result[local_index] = int(np.count_nonzero(np.linalg.norm(vectors, axis=1) < cutoff_A))
        return result

    def _components(
        self,
        p_o: np.ndarray,
        si_o: np.ndarray,
        si_cl: np.ndarray,
        zr_hist_total: np.ndarray,
        zr_hist_o: np.ndarray,
    ) -> CoordinationComponents:
        p_term = float(np.mean((p_o - 4) ** 2))
        si_term = float(np.mean((si_o + si_cl - 4) ** 2))
        zr_count = float(len(self.indices["Zr"]))
        total_delta = (zr_hist_total - self.target_zr_hist_total) / zr_count
        o_delta = (zr_hist_o - self.target_zr_hist_o) / zr_count
        total_term = float(np.sum((total_delta / self.epsilon_zr_total) ** 2))
        o_term = float(np.sum((o_delta / self.epsilon_zr_o) ** 2))
        return CoordinationComponents(p_term, si_term, total_term, o_term)

    def _direct_snapshot(self, atoms: Atoms) -> CoordinationSnapshot:
        p_o = self._pair_counts(atoms, "P", "O", self.cutoffs.p_o_A)
        if self.p_coordination_mode == "P-total":
            p_o = p_o + self._pair_counts(atoms, "P", "Cl", self.p_cl_cutoff_A)
        si_o = self._pair_counts(atoms, "Si", "O", self.cutoffs.si_o_A)
        si_cl = self._pair_counts(atoms, "Si", "Cl", self.cutoffs.si_cl_A)
        zr_o = self._pair_counts(atoms, "Zr", "O", self.cutoffs.zr_o_A)
        zr_cl = self._pair_counts(atoms, "Zr", "Cl", self.cutoffs.zr_cl_A)
        zr_hist_total = _histogram(zr_o + zr_cl, self.histogram_bins)
        zr_hist_o = _histogram(zr_o, self.histogram_bins)
        if hasattr(self, "target_zr_hist_total"):
            components = self._components(p_o, si_o, si_cl, zr_hist_total, zr_hist_o)
        else:
            components = CoordinationComponents(
                float(np.mean((p_o - 4) ** 2)),
                float(np.mean((si_o + si_cl - 4) ** 2)),
                0.0,
                0.0,
            )
        return CoordinationSnapshot(p_o, si_o, si_cl, zr_o, zr_cl, zr_hist_total, zr_hist_o, components)

    def full_recompute(self, atoms: Atoms | None = None) -> CoordinationSnapshot:
        return self._direct_snapshot(self.atoms if atoms is None else atoms)

    def _updated_pair_counts(
        self,
        current: np.ndarray,
        moved_index: int,
        new_position_A: np.ndarray,
        center: str,
        ligand: str,
        cutoff_A: float,
    ) -> np.ndarray:
        trial = current.copy()
        moved_symbol = str(self.symbols[moved_index])
        if moved_symbol == center:
            local = int(np.flatnonzero(self.indices[center] == moved_index)[0])
            ligands = self.indices[ligand]
            old_vectors = minimum_image_vectors(
                self.atoms.positions[ligands] - self.atoms.positions[moved_index], self.cell_A
            )
            new_vectors = minimum_image_vectors(
                self.atoms.positions[ligands] - new_position_A, self.cell_A
            )
            old_inside = np.linalg.norm(old_vectors, axis=1) < cutoff_A
            new_inside = np.linalg.norm(new_vectors, axis=1) < cutoff_A
            trial[local] += int(np.count_nonzero(new_inside) - np.count_nonzero(old_inside))
        elif moved_symbol == ligand:
            centers = self.indices[center]
            old_vectors = minimum_image_vectors(
                self.atoms.positions[moved_index] - self.atoms.positions[centers], self.cell_A
            )
            new_vectors = minimum_image_vectors(
                new_position_A - self.atoms.positions[centers], self.cell_A
            )
            trial += (
                (np.linalg.norm(new_vectors, axis=1) < cutoff_A).astype(np.int64)
                - (np.linalg.norm(old_vectors, axis=1) < cutoff_A).astype(np.int64)
            )
        if np.any(trial < 0):
            raise RuntimeError("Incremental coordination became negative")
        return trial

    def propose_move(self, atom_index: int, new_position_A: np.ndarray) -> CoordinationProposal:
        if not 0 <= atom_index < len(self.atoms):
            raise IndexError(atom_index)
        old_position = self.atoms.positions[atom_index].copy()
        wrapped = wrap_cartesian(np.asarray(new_position_A, dtype=float), self.cell_A)
        p_o = self._updated_pair_counts(
            self.p_o, atom_index, wrapped, "P", "O", self.cutoffs.p_o_A
        )
        if self.p_coordination_mode == "P-total":
            p_o = self._updated_pair_counts(
                p_o, atom_index, wrapped, "P", "Cl", self.p_cl_cutoff_A
            )
        si_o = self._updated_pair_counts(
            self.si_o, atom_index, wrapped, "Si", "O", self.cutoffs.si_o_A
        )
        si_cl = self._updated_pair_counts(
            self.si_cl, atom_index, wrapped, "Si", "Cl", self.cutoffs.si_cl_A
        )
        zr_o = self._updated_pair_counts(
            self.zr_o, atom_index, wrapped, "Zr", "O", self.cutoffs.zr_o_A
        )
        zr_cl = self._updated_pair_counts(
            self.zr_cl, atom_index, wrapped, "Zr", "Cl", self.cutoffs.zr_cl_A
        )
        zr_hist_total = _histogram(zr_o + zr_cl, self.histogram_bins)
        zr_hist_o = _histogram(zr_o, self.histogram_bins)
        components = self._components(p_o, si_o, si_cl, zr_hist_total, zr_hist_o)
        snapshot = CoordinationSnapshot(
            p_o, si_o, si_cl, zr_o, zr_cl, zr_hist_total, zr_hist_o, components
        )
        return CoordinationProposal(atom_index, old_position, wrapped, snapshot)

    def commit(self, proposal: CoordinationProposal) -> None:
        if not np.array_equal(self.atoms.positions[proposal.atom_index], proposal.old_position_A):
            raise RuntimeError("Stale coordination proposal")
        self.atoms.positions[proposal.atom_index] = proposal.new_position_A
        trial = proposal.trial
        self.p_o[:] = trial.p_o
        self.si_o[:] = trial.si_o
        self.si_cl[:] = trial.si_cl
        self.zr_o[:] = trial.zr_o
        self.zr_cl[:] = trial.zr_cl
        self.zr_hist_total[:] = trial.zr_hist_total
        self.zr_hist_o[:] = trial.zr_hist_o
        self.components = trial.components

    def reject(self, proposal: CoordinationProposal) -> None:
        del proposal

    def current_snapshot(self) -> CoordinationSnapshot:
        return CoordinationSnapshot(
            self.p_o.copy(),
            self.si_o.copy(),
            self.si_cl.copy(),
            self.zr_o.copy(),
            self.zr_cl.copy(),
            self.zr_hist_total.copy(),
            self.zr_hist_o.copy(),
            self.components,
        )

    def restore(self, positions_A: np.ndarray, snapshot: CoordinationSnapshot) -> None:
        self.atoms.positions[:] = np.asarray(positions_A, dtype=float)
        self.p_o[:] = snapshot.p_o
        self.si_o[:] = snapshot.si_o
        self.si_cl[:] = snapshot.si_cl
        self.zr_o[:] = snapshot.zr_o
        self.zr_cl[:] = snapshot.zr_cl
        self.zr_hist_total[:] = snapshot.zr_hist_total
        self.zr_hist_o[:] = snapshot.zr_hist_o
        self.components = snapshot.components

    def validation_errors(self) -> dict[str, Any]:
        direct = self.full_recompute()
        arrays = {
            "p_o": (self.p_o, direct.p_o),
            "si_o": (self.si_o, direct.si_o),
            "si_cl": (self.si_cl, direct.si_cl),
            "zr_o": (self.zr_o, direct.zr_o),
            "zr_cl": (self.zr_cl, direct.zr_cl),
            "zr_hist_total": (self.zr_hist_total, direct.zr_hist_total),
            "zr_hist_o": (self.zr_hist_o, direct.zr_hist_o),
        }
        differences = {
            name: int(np.max(np.abs(current - expected), initial=0))
            for name, (current, expected) in arrays.items()
        }
        return {
            "max_abs_integer_error": max(differences.values(), default=0),
            "by_array": differences,
            "max_abs_component_error": float(
                max(
                    abs(self.components.p - direct.components.p),
                    abs(self.components.si - direct.components.si),
                    abs(self.components.zr_total - direct.components.zr_total),
                    abs(self.components.zr_o - direct.components.zr_o),
                )
            ),
        }

    def serializable_definition(self) -> dict[str, Any]:
        return {
            "cutoffs_A": self.cutoffs.as_mapping(),
            "p_coordination_mode": self.p_coordination_mode,
            "p_cl_cutoff_A": self.p_cl_cutoff_A,
            "histogram_bins": self.histogram_bins,
            "target_zr_hist_total": self.target_zr_hist_total.tolist(),
            "target_zr_hist_o": self.target_zr_hist_o.tolist(),
            "epsilon_zr_total": self.epsilon_zr_total.tolist(),
            "epsilon_zr_o": self.epsilon_zr_o.tolist(),
            "epsilon_definition": "max(sqrt(p0*(1-p0)/N_Zr), 1/N_Zr) per bin",
        }

