import numpy as np
from ase import Atoms

from dpa4_xray_hrmc.xray_pdf.partial_rdf import compute_partial_rdfs


def _simple_cubic(a: float = 2.0) -> Atoms:
    return Atoms("Cu", positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * a, pbc=True)


def test_simple_cubic_nearest_shell_has_six_images() -> None:
    rdf = compute_partial_rdfs(_simple_cubic(), rcut_A=2.2, dr_A=0.01)
    index = int(np.argmin(np.abs(rdf.r_A - 2.0)))
    neighbourhood = rdf.counts[("Cu", "Cu")][index - 1 : index + 2]
    assert neighbourhood.sum() == 6
    assert rdf.counts[("Cu", "Cu")].sum() == 6


def test_supercell_invariance_of_periodic_histogram() -> None:
    primitive = _simple_cubic()
    supercell = primitive.repeat((2, 2, 2))
    rdf_one = compute_partial_rdfs(primitive, rcut_A=3.0, dr_A=0.02)
    rdf_eight = compute_partial_rdfs(supercell, rcut_A=3.0, dr_A=0.02)
    assert np.allclose(rdf_one.g[("Cu", "Cu")], rdf_eight.g[("Cu", "Cu")], atol=1e-12)
