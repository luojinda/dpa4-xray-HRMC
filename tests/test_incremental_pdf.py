import numpy as np
from ase import Atoms

from dpa4_xray_hrmc.rmc.incremental_pdf import IncrementalXrayPDF


def _mixed_atoms() -> Atoms:
    return Atoms(
        ["Li", "O", "Li", "O", "Si", "Si"],
        positions=[
            [1.0, 1.0, 1.0],
            [3.0, 1.2, 1.1],
            [7.0, 6.5, 5.5],
            [9.0, 6.7, 5.4],
            [4.0, 8.0, 3.0],
            [10.0, 2.0, 9.0],
        ],
        cell=[[14.0, 0.0, 0.0], [1.0, 15.0, 0.0], [0.5, 0.8, 16.0]],
        pbc=True,
    )


def test_kernel_initial_state_matches_phase1_full_recompute() -> None:
    engine = IncrementalXrayPDF(
        _mixed_atoms(), np.linspace(0.9, 8.0, 72), np.linspace(1.5, 6.0, 91), 5.0, 0.1, 0.3
    )
    errors = engine.validation_errors()
    assert errors["max_abs_hist_count"] == 0
    assert errors["max_abs_G_A-2"] < 1e-11


def test_trial_commit_reject_and_periodic_crossing_match_full() -> None:
    engine = IncrementalXrayPDF(
        _mixed_atoms(), np.linspace(0.9, 8.0, 72), np.linspace(1.5, 6.0, 91), 5.0, 0.1, 0.3
    )
    positions_before = engine.positions_A.copy()
    counts_before = engine.counts.copy()
    g_before = engine.g_r_A2.copy()
    rejected = engine.propose_move(0, np.array([13.9, 1.1, 1.0]))
    engine.reject(rejected)
    assert np.array_equal(engine.positions_A, positions_before)
    assert np.array_equal(engine.counts, counts_before)
    assert np.array_equal(engine.g_r_A2, g_before)

    proposal = engine.propose_move(0, np.array([13.9, 1.1, 1.0]))
    trial_atoms = engine.atoms.copy()
    trial_atoms.positions[0] = proposal.new_position_A
    _, full_trial = engine.full_recompute(trial_atoms)
    np.testing.assert_allclose(engine.g_r_A2 + proposal.delta_g_A2, full_trial.g_r_A2, atol=1e-11)
    engine.commit(proposal)
    errors = engine.validation_errors()
    assert errors["max_abs_hist_count"] == 0
    assert errors["max_abs_G_A-2"] < 1e-11


def test_generic_multi_image_delta_matches_full_recompute() -> None:
    atoms = Atoms(
        ["Li", "O", "Li"],
        positions=[[0.2, 0.3, 0.4], [2.0, 1.0, 1.2], [4.4, 3.8, 2.2]],
        cell=np.eye(3) * 5.0,
        pbc=True,
    )
    engine = IncrementalXrayPDF(
        atoms, np.linspace(1.0, 6.0, 51), np.linspace(1.0, 4.0, 61), 6.0, 0.2, 0.3
    )
    assert not engine.minimum_image_safe
    proposal = engine.propose_move(1, np.array([2.2, 0.8, 1.4]))
    trial = engine.atoms.copy()
    trial.positions[1] = proposal.new_position_A
    rdf, result = engine.full_recompute(trial)
    np.testing.assert_array_equal(
        engine.counts.reshape(-1)[proposal.flat_hist_indices] + proposal.count_deltas,
        np.stack([rdf.counts[pair] for pair in engine.pairs]).reshape(-1)[proposal.flat_hist_indices],
    )
    np.testing.assert_allclose(engine.g_r_A2 + proposal.delta_g_A2, result.g_r_A2, atol=1e-10)
