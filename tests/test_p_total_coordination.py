from __future__ import annotations

import numpy as np
from ase import Atoms

from dpa4_xray_hrmc.rmc.coordination import CoordinationCutoffs, IncrementalCoordination


def cutoffs() -> CoordinationCutoffs:
    return CoordinationCutoffs.from_mapping(
        {"P-O": 1.5, "Si-O": 1.5, "Si-Cl": 2.0, "Zr-O": 2.0, "Zr-Cl": 2.2}
    )


def mixed_ligand_atoms() -> Atoms:
    return Atoms(
        ["P", "Si", "Zr", "O", "O", "O", "Cl"],
        positions=[
            [5.0, 5.0, 5.0],
            [1.0, 1.0, 1.0],
            [8.0, 8.0, 8.0],
            [6.4, 5.0, 5.0],
            [5.0, 6.4, 5.0],
            [5.0, 5.0, 6.4],
            [3.2, 5.0, 5.0],
        ],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )


def assert_exact(engine: IncrementalCoordination) -> None:
    errors = engine.validation_errors()
    assert errors["max_abs_integer_error"] == 0
    assert errors["max_abs_component_error"] == 0.0


def test_optional_p_total_counts_o_and_cl_and_stays_incremental_exact() -> None:
    atoms = mixed_ligand_atoms()
    legacy = IncrementalCoordination(atoms, cutoffs())
    total = IncrementalCoordination(
        atoms,
        cutoffs(),
        p_coordination_mode="P-total",
        p_cl_cutoff_A=2.1,
    )
    assert legacy.p_o.tolist() == [3]
    assert total.p_o.tolist() == [4]
    assert total.components.p == 0.0
    assert total.serializable_definition()["p_coordination_mode"] == "P-total"

    rng = np.random.default_rng(20261123)
    for step in range(100):
        index = int(rng.integers(len(total.atoms)))
        proposal = total.propose_move(
            index, total.atoms.positions[index] + rng.uniform(-0.4, 0.4, 3)
        )
        if step % 4:
            total.commit(proposal)
        else:
            total.reject(proposal)
        assert_exact(total)
