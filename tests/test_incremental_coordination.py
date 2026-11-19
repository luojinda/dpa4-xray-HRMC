from __future__ import annotations

import numpy as np
from ase import Atoms

from dpa4_xray_hrmc.rmc.coordination import CoordinationCutoffs, IncrementalCoordination


def example_atoms() -> Atoms:
    symbols = ["P", "Si", "Zr", "O", "O", "O", "Cl", "Cl", "Cl"]
    positions = [
        [0.2, 1.0, 1.0],
        [3.0, 1.0, 1.0],
        [6.0, 1.0, 1.0],
        [9.0, 1.0, 1.0],  # PBC-near P
        [3.0, 2.4, 1.0],
        [6.0, 2.6, 1.0],
        [3.0, 1.0, 2.8],
        [6.0, 1.0, 3.0],
        [6.0, 3.0, 1.0],
    ]
    return Atoms(symbols, positions=positions, cell=[10.0, 10.0, 10.0], pbc=True)


def cutoffs() -> CoordinationCutoffs:
    return CoordinationCutoffs.from_mapping(
        {"P-O": 1.5, "Si-O": 1.5, "Si-Cl": 2.0, "Zr-O": 2.0, "Zr-Cl": 2.2}
    )


def assert_exact(engine: IncrementalCoordination) -> None:
    errors = engine.validation_errors()
    assert errors["max_abs_integer_error"] == 0
    assert errors["max_abs_component_error"] == 0.0


def test_initial_full_coordination_and_strict_cutoff() -> None:
    engine = IncrementalCoordination(example_atoms(), cutoffs())
    assert engine.p_o.tolist() == [1]
    assert engine.si_o.tolist() == [1]
    assert engine.si_cl.tolist() == [1]
    assert engine.zr_o.tolist() == [1]
    # Two Cl atoms are exactly 2.0 A from Zr and are included by the 2.2 A cutoff.
    assert engine.zr_cl.tolist() == [2]

    atoms = example_atoms()
    atoms.positions[3] = [1.7, 1.0, 1.0]  # exactly 1.5 A from P
    boundary = IncrementalCoordination(atoms, cutoffs())
    assert boundary.p_o.tolist() == [0]


def test_ligand_move_updates_multiple_centers_and_reject_is_unchanged() -> None:
    engine = IncrementalCoordination(example_atoms(), cutoffs())
    before_positions = engine.atoms.positions.copy()
    before = engine.current_snapshot()
    proposal = engine.propose_move(4, [4.5, 1.0, 1.0])
    # The moved O changes both Si-O and Zr-O membership.
    assert proposal.trial.si_o.tolist() != engine.si_o.tolist()
    assert proposal.trial.zr_o.tolist() != engine.zr_o.tolist()
    engine.reject(proposal)
    assert np.array_equal(engine.atoms.positions, before_positions)
    assert np.array_equal(engine.si_o, before.si_o)
    assert np.array_equal(engine.zr_o, before.zr_o)
    assert_exact(engine)


def test_center_and_ligand_random_moves_match_full_recompute() -> None:
    engine = IncrementalCoordination(example_atoms(), cutoffs())
    rng = np.random.default_rng(20261119)
    for step in range(250):
        index = int(rng.integers(len(engine.atoms)))
        proposal = engine.propose_move(
            index, engine.atoms.positions[index] + rng.uniform(-0.8, 0.8, 3)
        )
        if step % 3:
            engine.commit(proposal)
        else:
            engine.reject(proposal)
        assert_exact(engine)


def test_restore_recovers_integer_state() -> None:
    engine = IncrementalCoordination(example_atoms(), cutoffs())
    positions = engine.atoms.positions.copy()
    snapshot = engine.current_snapshot()
    proposal = engine.propose_move(2, [5.0, 2.0, 1.0])
    engine.commit(proposal)
    engine.restore(positions, snapshot)
    assert_exact(engine)
    assert np.array_equal(engine.atoms.positions, positions)

