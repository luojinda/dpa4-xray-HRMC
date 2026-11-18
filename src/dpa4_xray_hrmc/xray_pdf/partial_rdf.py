"""Periodic partial RDFs using explicit lattice images and ordered normalization."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from ase import Atoms
from scipy.spatial import cKDTree


Pair = tuple[str, str]


@dataclass(frozen=True)
class PartialRDF:
    r_A: np.ndarray
    edges_A: np.ndarray
    g: dict[Pair, np.ndarray]
    counts: dict[Pair, np.ndarray]
    element_counts: dict[str, int]
    volume_A3: float
    dr_A: float
    rcut_A: float
    translation_bounds: tuple[int, int, int]
    n_translations: int


def element_pairs(elements: list[str] | tuple[str, ...]) -> list[Pair]:
    ordered = sorted(elements)
    return [(a, b) for i, a in enumerate(ordered) for b in ordered[i:]]


def translation_bounds(cell: np.ndarray, rcut_A: float) -> tuple[int, int, int]:
    """Guaranteed image bounds from fractional-coordinate component limits."""
    inverse = np.linalg.inv(cell)
    # A displacement shorter than rcut has fractional component bounded by
    # rcut*|column(inv(cell))|. Atomic fractional differences lie in (-1, 1).
    bounds = np.ceil(rcut_A * np.linalg.norm(inverse, axis=0) + 1.0).astype(int)
    return tuple(int(x) for x in bounds)


def compute_partial_rdfs(atoms: Atoms, rcut_A: float, dr_A: float) -> PartialRDF:
    """Compute g_ab with n_b(r)=rho0*c_b*g_ab(r)*4*pi*r^2*dr.

    Pairs are stored only for a<=b, but the histogram is an ordered central-a
    to neighbour-b count. For a==b, only the exact zero-shift self pair is
    excluded; non-zero periodic images of the same atom are physical pairs.
    """
    if rcut_A <= 0 or dr_A <= 0:
        raise ValueError("rcut_A and dr_A must be positive")
    if not np.all(atoms.pbc):
        raise ValueError("All three PBC flags must be true")
    cell = np.asarray(atoms.cell, dtype=float)
    positions = atoms.get_positions()
    symbols = np.asarray(atoms.get_chemical_symbols())
    elements = sorted(set(symbols))
    by_element = {element: np.flatnonzero(symbols == element) for element in elements}
    counts_by_element = {element: int(len(indices)) for element, indices in by_element.items()}
    volume = float(atoms.get_volume())

    n_bins = int(round(rcut_A / dr_A))
    if not np.isclose(n_bins * dr_A, rcut_A, atol=1e-10):
        raise ValueError("rcut_A must be an integer multiple of dr_A")
    edges = np.linspace(0.0, rcut_A, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell_volumes = (4.0 * np.pi / 3.0) * (edges[1:] ** 3 - edges[:-1] ** 3)

    bounds = translation_bounds(cell, rcut_A)
    shifts = np.asarray(list(product(*(range(-n, n + 1) for n in bounds))), dtype=int)
    translations = shifts @ cell
    zero_shift_index = int(np.flatnonzero(np.all(shifts == 0, axis=1))[0])

    histograms: dict[Pair, np.ndarray] = {}
    raw_counts: dict[Pair, np.ndarray] = {}
    # Build each neighbour-species image tree once and reuse it for all central species.
    for b_index, b in enumerate(elements):
        target_indices = by_element[b]
        target_base = positions[target_indices]
        image_positions = (target_base[None, :, :] + translations[:, None, :]).reshape(-1, 3)
        image_atom_indices = np.tile(target_indices, len(shifts))
        image_shift_indices = np.repeat(np.arange(len(shifts)), len(target_indices))
        tree = cKDTree(image_positions)
        for a in elements[: b_index + 1]:
            central_indices = by_element[a]
            histogram = np.zeros(n_bins, dtype=np.int64)
            for central_index in central_indices:
                neighbours = np.asarray(
                    tree.query_ball_point(positions[central_index], rcut_A, return_sorted=False),
                    dtype=int,
                )
                if neighbours.size == 0:
                    continue
                if a == b:
                    is_self = (
                        (image_atom_indices[neighbours] == central_index)
                        & (image_shift_indices[neighbours] == zero_shift_index)
                    )
                    neighbours = neighbours[~is_self]
                distances = np.linalg.norm(image_positions[neighbours] - positions[central_index], axis=1)
                histogram += np.histogram(distances, bins=edges)[0]
            pair = (a, b)
            denominator = len(central_indices) * (len(target_indices) / volume) * shell_volumes
            histograms[pair] = histogram / denominator
            raw_counts[pair] = histogram

    return PartialRDF(
        r_A=centers,
        edges_A=edges,
        g=histograms,
        counts=raw_counts,
        element_counts=counts_by_element,
        volume_A3=volume,
        dr_A=dr_A,
        rcut_A=rcut_A,
        translation_bounds=bounds,
        n_translations=len(shifts),
    )


def truncate_rdf(rdf: PartialRDF, rcut_A: float) -> PartialRDF:
    if rcut_A > rdf.rcut_A + 1e-12:
        raise ValueError("Requested cutoff exceeds available RDF")
    n_bins = int(round(rcut_A / rdf.dr_A))
    return PartialRDF(
        r_A=rdf.r_A[:n_bins],
        edges_A=rdf.edges_A[: n_bins + 1],
        g={pair: values[:n_bins] for pair, values in rdf.g.items()},
        counts={pair: values[:n_bins] for pair, values in rdf.counts.items()},
        element_counts=rdf.element_counts,
        volume_A3=rdf.volume_A3,
        dr_A=rdf.dr_A,
        rcut_A=rcut_A,
        translation_bounds=rdf.translation_bounds,
        n_translations=rdf.n_translations,
    )


def rebin_rdf(rdf: PartialRDF, new_dr_A: float) -> PartialRDF:
    ratio_float = new_dr_A / rdf.dr_A
    ratio = int(round(ratio_float))
    if ratio < 1 or not np.isclose(ratio_float, ratio, atol=1e-10):
        raise ValueError("new_dr_A must be an integer multiple of the original bin width")
    if len(rdf.r_A) % ratio:
        raise ValueError("RDF bin count must be divisible by the rebin ratio")
    new_edges = rdf.edges_A[::ratio]
    if len(new_edges) != len(rdf.r_A) // ratio + 1:
        new_edges = np.append(new_edges, rdf.edges_A[-1])
    new_centers = 0.5 * (new_edges[:-1] + new_edges[1:])
    shell_volumes = (4.0 * np.pi / 3.0) * (new_edges[1:] ** 3 - new_edges[:-1] ** 3)
    rebinned_counts = {
        pair: values.reshape(-1, ratio).sum(axis=1) for pair, values in rdf.counts.items()
    }
    rebinned_g: dict[Pair, np.ndarray] = {}
    for pair, values in rebinned_counts.items():
        a, b = pair
        denominator = rdf.element_counts[a] * (rdf.element_counts[b] / rdf.volume_A3) * shell_volumes
        rebinned_g[pair] = values / denominator
    return PartialRDF(
        r_A=new_centers,
        edges_A=new_edges,
        g=rebinned_g,
        counts=rebinned_counts,
        element_counts=rdf.element_counts,
        volume_A3=rdf.volume_A3,
        dr_A=new_dr_A,
        rcut_A=rdf.rcut_A,
        translation_bounds=rdf.translation_bounds,
        n_translations=rdf.n_translations,
    )
