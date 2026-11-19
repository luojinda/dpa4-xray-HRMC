import numpy as np
from ase import Atoms

from dpa4_xray_hrmc.rmc.diagnostics import structure_diagnostics
from dpa4_xray_hrmc.rmc.hard_sphere import HardSphereModel


def test_structure_diagnostics_reports_fixed_cell_displacement_and_coordination() -> None:
    initial = Atoms(
        ["P", "O", "O", "O", "O", "Li", "Cl"],
        positions=[
            [5, 5, 5],
            [6.5, 5, 5],
            [3.5, 5, 5],
            [5, 6.5, 5],
            [5, 5, 6.5],
            [9, 9, 9],
            [11, 9, 9],
        ],
        cell=np.eye(3) * 15,
        pbc=True,
    )
    trial = initial.copy()
    trial.positions[5, 0] += 0.1
    hard = HardSphereModel.from_atoms(initial)
    cutoffs = {
        "P-O": {"cutoff_A": 2.0, "peak_A": 1.5, "method": "test"},
        "Li-Cl": {"cutoff_A": 2.5, "peak_A": 2.0, "method": "test"},
    }
    result = structure_diagnostics(initial, trial, hard, cutoffs)
    assert result["hard_sphere_violation_total"] == 0
    assert result["coordination"]["P-O"]["fourfold_fraction"] == 1.0
    assert np.isclose(result["displacement_from_initial"]["max_A"], 0.1)
