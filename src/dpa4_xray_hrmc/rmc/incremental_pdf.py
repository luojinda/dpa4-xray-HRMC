"""Transactional single-atom histogram and Q-dependent X-ray PDF updates."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from ase import Atoms

from dpa4_xray_hrmc.xray_pdf.form_factors import XrayWeights, q_dependent_weights
from dpa4_xray_hrmc.xray_pdf.partial_rdf import (
    Pair,
    PartialRDF,
    compute_partial_rdfs,
    element_pairs,
    translation_bounds,
)
from dpa4_xray_hrmc.xray_pdf.thermal import pair_debye_waller
from dpa4_xray_hrmc.xray_pdf.transform import (
    PDFResult,
    _integration_weights,
    forward_transform,
)


def minimum_cell_height(cell_A: np.ndarray) -> float:
    """Return the minimum distance between opposite faces of a triclinic cell."""

    cell = np.asarray(cell_A, dtype=float)
    inverse = np.linalg.inv(cell)
    return float(np.min(1.0 / np.linalg.norm(inverse, axis=0)))


def wrap_cartesian(position_A: np.ndarray, cell_A: np.ndarray) -> np.ndarray:
    fractional = np.asarray(position_A, dtype=float) @ np.linalg.inv(cell_A)
    fractional -= np.floor(fractional)
    return fractional @ cell_A


@dataclass(frozen=True)
class XrayPDFKernel:
    """Affine map from flattened Phase 1 pair counts to G(r)."""

    q_Ainv: np.ndarray
    output_r_A: np.ndarray
    pairs: tuple[Pair, ...]
    n_bins: int
    kernel_A2_per_count: np.ndarray
    background_A2: np.ndarray
    weights: XrayWeights
    biso_A2: float
    lorch_mod: bool

    @classmethod
    def build(
        cls,
        rdf: PartialRDF,
        q_Ainv: np.ndarray,
        output_r_A: np.ndarray,
        biso_A2: float,
        lorch_mod: bool = False,
    ) -> "XrayPDFKernel":
        q = np.asarray(q_Ainv, dtype=float)
        r_out = np.asarray(output_r_A, dtype=float)
        weights = q_dependent_weights(rdf.element_counts, q)
        pairs = tuple(sorted(rdf.g))
        radial_r = rdf.r_A
        rho0 = sum(rdf.element_counts.values()) / rdf.volume_A3
        radial_factor = 4.0 * np.pi * rho0 * radial_r**2 * rdf.dr_A
        sinc_qr = np.sinc(np.outer(q, radial_r) / np.pi)

        q_integral = q * _integration_weights(q)
        if lorch_mod:
            argument = np.pi * (q - q[0]) / (q[-1] - q[0])
            q_integral = q_integral * np.sinc(argument / np.pi)
        transform = (2.0 / np.pi) * np.sin(np.outer(r_out, q)) * q_integral[None, :]

        shell = (4.0 * np.pi / 3.0) * (rdf.edges_A[1:] ** 3 - rdf.edges_A[:-1] ** 3)
        kernel = np.empty((len(pairs), len(radial_r), len(r_out)), dtype=np.float64)
        background_sq_minus_one = np.zeros_like(q)
        partial_background = sinc_qr @ (-radial_factor)
        for pair_index, pair in enumerate(pairs):
            a, b = pair
            denominator = (
                rdf.element_counts[a]
                * (rdf.element_counts[b] / rdf.volume_A3)
                * shell
            )
            count_factor = radial_factor / denominator
            weighted_q = weights.pair_weights[pair] * pair_debye_waller(
                q, biso_A2, biso_A2
            )
            count_to_sq = weighted_q[:, None] * sinc_qr * count_factor[None, :]
            kernel[pair_index] = (transform @ count_to_sq).T
            background_sq_minus_one += weighted_q * partial_background
        background = transform @ background_sq_minus_one
        return cls(
            q,
            r_out,
            pairs,
            len(radial_r),
            kernel.reshape(len(pairs) * len(radial_r), len(r_out)),
            background,
            weights,
            float(biso_A2),
            bool(lorch_mod),
        )

    @property
    def memory_bytes(self) -> int:
        return int(self.kernel_A2_per_count.nbytes + self.background_A2.nbytes)

    def from_counts(self, counts: np.ndarray) -> np.ndarray:
        flat = np.asarray(counts, dtype=np.float64).reshape(-1)
        if len(flat) != self.kernel_A2_per_count.shape[0]:
            raise ValueError("Count array does not match kernel dimensions")
        return self.background_A2 + flat @ self.kernel_A2_per_count

    def delta_g(self, flat_indices: np.ndarray, count_deltas: np.ndarray) -> np.ndarray:
        if len(flat_indices) == 0:
            return np.zeros_like(self.output_r_A)
        return np.asarray(count_deltas, dtype=np.float64) @ self.kernel_A2_per_count[
            flat_indices
        ]


@dataclass(frozen=True)
class IncrementalProposal:
    atom_index: int
    old_position_A: np.ndarray
    new_position_A: np.ndarray
    flat_hist_indices: np.ndarray
    count_deltas: np.ndarray
    delta_g_A2: np.ndarray


class IncrementalXrayPDF:
    """Mutable PDF state with side-effect-free proposals and explicit commits."""

    def __init__(
        self,
        atoms: Atoms,
        q_Ainv: np.ndarray,
        output_r_A: np.ndarray,
        rdf_rcut_A: float,
        rdf_dr_A: float,
        biso_A2: float,
        lorch_mod: bool = False,
    ) -> None:
        self.atoms = atoms.copy()
        if not np.all(self.atoms.pbc):
            raise ValueError("Incremental PDF requires three-dimensional PBC")
        self.cell_A = np.asarray(self.atoms.cell, dtype=float)
        self.inverse_cell_Ainv = np.linalg.inv(self.cell_A)
        self.rcut_A = float(rdf_rcut_A)
        self.dr_A = float(rdf_dr_A)
        self.n_bins = int(round(self.rcut_A / self.dr_A))
        if not np.isclose(self.n_bins * self.dr_A, self.rcut_A):
            raise ValueError("rdf_rcut_A must be divisible by rdf_dr_A")
        self.minimum_image_safe = self.rcut_A < 0.5 * minimum_cell_height(self.cell_A)
        self.symbols = np.asarray(self.atoms.get_chemical_symbols())

        self.rdf = compute_partial_rdfs(self.atoms, self.rcut_A, self.dr_A)
        self.kernel = XrayPDFKernel.build(
            self.rdf, q_Ainv, output_r_A, biso_A2, lorch_mod
        )
        self.pairs = self.kernel.pairs
        self.pair_to_index = {pair: index for index, pair in enumerate(self.pairs)}
        self.counts = np.stack([self.rdf.counts[pair] for pair in self.pairs]).astype(
            np.int64, copy=True
        )
        self.g_r_A2 = self.kernel.from_counts(self.counts)
        self._translations_A: np.ndarray | None = None
        if not self.minimum_image_safe:
            bounds = translation_bounds(self.cell_A, self.rcut_A)
            shifts = np.asarray(
                list(product(*(range(-bound, bound + 1) for bound in bounds))),
                dtype=int,
            )
            self._translations_A = shifts @ self.cell_A

    def _pair_ids_and_factors(self, atom_index: int, other_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        moved_symbol = str(self.symbols[atom_index])
        pair_ids = np.empty(len(other_indices), dtype=np.int64)
        factors = np.ones(len(other_indices), dtype=np.int64)
        for offset, other_index in enumerate(other_indices):
            other_symbol = str(self.symbols[other_index])
            pair = tuple(sorted((moved_symbol, other_symbol)))
            pair_ids[offset] = self.pair_to_index[pair]  # type: ignore[index]
            if moved_symbol == other_symbol:
                factors[offset] = 2
        return pair_ids, factors

    def _distance_records(
        self,
        atom_index: int,
        position_A: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        other_indices = np.delete(np.arange(len(self.atoms), dtype=np.int64), atom_index)
        pair_ids, factors = self._pair_ids_and_factors(atom_index, other_indices)
        base = self.atoms.positions[other_indices] - position_A[None, :]
        if self.minimum_image_safe:
            fractional = base @ self.inverse_cell_Ainv
            fractional -= np.rint(fractional)
            distances = np.linalg.norm(fractional @ self.cell_A, axis=1)
            keep = distances <= self.rcut_A
            return distances[keep], pair_ids[keep], factors[keep]

        assert self._translations_A is not None
        vectors = base[:, None, :] + self._translations_A[None, :, :]
        distances = np.linalg.norm(vectors, axis=2)
        keep = distances <= self.rcut_A
        other_rows, _ = np.nonzero(keep)
        return distances[keep], pair_ids[other_rows], factors[other_rows]

    def _flat_bins(
        self, distances_A: np.ndarray, pair_ids: np.ndarray
    ) -> np.ndarray:
        bins = np.floor(distances_A / self.dr_A).astype(np.int64)
        bins = np.minimum(bins, self.n_bins - 1)
        return pair_ids * self.n_bins + bins

    @staticmethod
    def _coalesce(indices: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(indices) == 0:
            return indices.astype(np.int64), values.astype(np.int64)
        order = np.argsort(indices, kind="stable")
        sorted_indices = indices[order]
        sorted_values = values[order]
        unique, starts = np.unique(sorted_indices, return_index=True)
        summed = np.add.reduceat(sorted_values, starts)
        keep = summed != 0
        return unique[keep].astype(np.int64), summed[keep].astype(np.int64)

    def propose_move(self, atom_index: int, new_position_A: np.ndarray) -> IncrementalProposal:
        if not 0 <= atom_index < len(self.atoms):
            raise IndexError(atom_index)
        old_position = self.atoms.positions[atom_index].copy()
        new_position = wrap_cartesian(new_position_A, self.cell_A)
        old_dist, old_pairs, old_factors = self._distance_records(atom_index, old_position)
        new_dist, new_pairs, new_factors = self._distance_records(atom_index, new_position)
        old_indices = self._flat_bins(old_dist, old_pairs)
        new_indices = self._flat_bins(new_dist, new_pairs)
        flat_indices, count_deltas = self._coalesce(
            np.concatenate((old_indices, new_indices)),
            np.concatenate((-old_factors, new_factors)),
        )
        delta_g = self.kernel.delta_g(flat_indices, count_deltas)
        return IncrementalProposal(
            int(atom_index),
            old_position,
            new_position,
            flat_indices,
            count_deltas,
            delta_g,
        )

    def commit(self, proposal: IncrementalProposal) -> None:
        if not np.array_equal(self.atoms.positions[proposal.atom_index], proposal.old_position_A):
            raise RuntimeError("Proposal no longer matches current atomic position")
        flat_counts = self.counts.reshape(-1)
        flat_counts[proposal.flat_hist_indices] += proposal.count_deltas
        if np.any(flat_counts[proposal.flat_hist_indices] < 0):
            raise RuntimeError("Incremental histogram became negative")
        self.g_r_A2 += proposal.delta_g_A2
        self.atoms.positions[proposal.atom_index] = proposal.new_position_A

    def reject(self, proposal: IncrementalProposal) -> None:
        del proposal

    @property
    def positions_A(self) -> np.ndarray:
        return self.atoms.positions

    def full_recompute(self, atoms: Atoms | None = None) -> tuple[PartialRDF, PDFResult]:
        structure = self.atoms if atoms is None else atoms
        rdf = compute_partial_rdfs(structure, self.rcut_A, self.dr_A)
        weights = q_dependent_weights(rdf.element_counts, self.kernel.q_Ainv)
        result = forward_transform(
            rdf,
            weights,
            self.kernel.output_r_A,
            self.kernel.biso_A2,
            self.kernel.lorch_mod,
        )
        return rdf, result

    def validation_errors(self) -> dict[str, float]:
        rdf, result = self.full_recompute()
        reference_counts = np.stack([rdf.counts[pair] for pair in self.pairs])
        difference = self.g_r_A2 - result.g_r_A2
        return {
            "max_abs_G_A-2": float(np.max(np.abs(difference))),
            "rms_G_A-2": float(np.sqrt(np.mean(difference**2))),
            "max_abs_hist_count": int(np.max(np.abs(self.counts - reference_counts))),
        }
