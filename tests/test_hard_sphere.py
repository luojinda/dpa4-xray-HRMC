import numpy as np
from ase import Atoms

from dpa4_xray_hrmc.rmc.hard_sphere import HardSphereModel


def test_pair_specific_hard_sphere_accepts_and_rejects() -> None:
    atoms = Atoms(
        ["Li", "O", "Li", "O"],
        positions=[[1, 1, 1], [3, 1, 1], [7, 7, 7], [9, 7, 7]],
        cell=np.eye(3) * 12,
        pbc=True,
    )
    model = HardSphereModel.from_atoms(atoms, scale=0.95)
    valid, _, _ = model.check_move(atoms, 0, np.array([1.05, 1.0, 1.0]))
    invalid, _, pair = model.check_move(atoms, 0, np.array([2.9, 1.0, 1.0]))
    assert valid
    assert not invalid
    assert pair == ("Li", "O")
