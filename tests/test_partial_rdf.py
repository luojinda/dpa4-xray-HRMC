import numpy as np
from ase import Atoms

from dpa4_xray_hrmc.xray_pdf.partial_rdf import compute_partial_rdfs, rebin_rdf


def test_uniform_random_configuration_has_unit_shell_average() -> None:
    rng = np.random.default_rng(20261118)
    cell_length = 24.0
    positions = rng.random((700, 3)) * cell_length
    atoms = Atoms("Ar700", positions=positions, cell=np.eye(3) * cell_length, pbc=True)
    rdf = compute_partial_rdfs(atoms, rcut_A=6.0, dr_A=0.2)
    mask = (rdf.r_A >= 2.0) & (rdf.r_A <= 6.0)
    edges = rdf.edges_A
    shell = (4 * np.pi / 3) * (edges[1:] ** 3 - edges[:-1] ** 3)
    shell_mean = np.average(rdf.g[("Ar", "Ar")][mask], weights=shell[mask])
    assert abs(shell_mean - 1.0) < 0.04


def test_rebin_preserves_raw_counts() -> None:
    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=np.eye(3) * 2.0, pbc=True)
    fine = compute_partial_rdfs(atoms, rcut_A=4.0, dr_A=0.01)
    coarse = rebin_rdf(fine, 0.02)
    assert coarse.counts[("Cu", "Cu")].sum() == fine.counts[("Cu", "Cu")].sum()
    assert np.isclose(coarse.rcut_A, fine.rcut_A)
