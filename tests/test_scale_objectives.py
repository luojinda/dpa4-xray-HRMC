import numpy as np
from scipy.optimize import minimize_scalar

from dpa4_xray_hrmc.rmc.objectives import ScaleMode, ScaleObjective


def test_three_scale_modes_and_incremental_coefficients() -> None:
    x = np.array([1.0, -0.5, 0.3, 1.4])
    y = 0.6 * x + np.array([0.02, -0.01, 0.03, -0.02])
    objective = ScaleObjective.from_initial(x, y)
    assert 0.0 < objective.s0 < 1.0
    assert objective.score(ScaleMode.NO_SCALE, x).scale == 1.0
    assert objective.score(ScaleMode.FIXED_INITIAL, x).scale == objective.s0

    delta = np.array([0.01, 0.03, -0.02, 0.04])
    current = objective.score(ScaleMode.BOUNDED_DYNAMIC, x)
    trial = objective.trial_score(ScaleMode.BOUNDED_DYNAMIC, x, current, delta)
    direct = objective.score(ScaleMode.BOUNDED_DYNAMIC, x + delta)
    assert np.isclose(trial.A, direct.A, atol=1e-14)
    assert np.isclose(trial.B, direct.B, atol=1e-14)
    assert np.isclose(trial.loss, direct.loss, atol=1e-14)


def test_bounded_scale_matches_scalar_minimization_and_clips() -> None:
    x = np.array([1.0, 2.0, -1.0])
    y = 0.5 * x
    objective = ScaleObjective.from_initial(x, y)
    changed = 0.2 * x
    score = objective.score(ScaleMode.BOUNDED_DYNAMIC, changed)
    numerical = minimize_scalar(
        lambda scale: np.sum((scale * changed - y) ** 2),
        bounds=(objective.lower_scale, objective.upper_scale),
        method="bounded",
    )
    assert np.isclose(score.scale, objective.upper_scale)
    assert np.isclose(score.scale, numerical.x, atol=1e-5)
