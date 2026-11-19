import numpy as np
from ase import Atoms

from dpa4_xray_hrmc.rmc.incremental_pdf import IncrementalXrayPDF
from dpa4_xray_hrmc.rmc.objectives import ScaleMode, ScaleObjective
from dpa4_xray_hrmc.rmc.runner import Counters, _restore_checkpoint, _save_checkpoint


def test_checkpoint_restores_histogram_pdf_scores_and_rng(tmp_path) -> None:
    atoms = Atoms(
        ["Li", "O", "Li"],
        positions=[[1, 1, 1], [3, 1, 1], [7, 7, 7]],
        cell=np.eye(3) * 12,
        pbc=True,
    )
    q = np.linspace(1, 6, 51)
    r = np.linspace(1.5, 5, 36)
    engine = IncrementalXrayPDF(atoms, q, r, 5, 0.1, 0.3)
    objective = ScaleObjective.from_initial(engine.g_r_A2, 0.7 * engine.g_r_A2)
    current = objective.score(ScaleMode.FIXED_INITIAL, engine.g_r_A2)
    proposal = engine.propose_move(0, np.array([1.1, 1.0, 1.0]))
    engine.commit(proposal)
    current = objective.score(ScaleMode.FIXED_INITIAL, engine.g_r_A2)
    rngs = tuple(np.random.default_rng(seed) for seed in (1, 2, 3))
    expected_random = [rng.random() for rng in rngs]
    path = tmp_path / "checkpoint.npz"
    _save_checkpoint(
        path,
        engine,
        Counters(attempted=1, accepted=1),
        current,
        engine.positions_A.copy(),
        engine.g_r_A2.copy(),
        current.loss,
        current.scale,
        1,
        rngs,
    )
    restored = IncrementalXrayPDF(atoms, q, r, 5, 0.1, 0.3)
    restored_rngs = tuple(np.random.default_rng(seed) for seed in (9, 9, 9))
    counters, score, *_ = _restore_checkpoint(path, restored, restored_rngs)
    assert counters.attempted == 1
    assert score == current
    np.testing.assert_array_equal(restored.positions_A, engine.positions_A)
    np.testing.assert_array_equal(restored.counts, engine.counts)
    np.testing.assert_array_equal(restored.g_r_A2, engine.g_r_A2)
    assert [rng.random() for rng in restored_rngs] != expected_random
