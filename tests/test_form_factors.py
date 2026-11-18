import numpy as np
import xraydb

from dpa4_xray_hrmc.xray_pdf.form_factors import neutral_form_factors, q_dependent_weights


def test_f0_q_zero_and_q_over_4pi_contract() -> None:
    q = np.array([0.0, 8.0, 24.0])
    factors = neutral_form_factors(["Li", "Nb"], q)
    assert np.isclose(factors["Li"][0], 3.0, rtol=0.03)
    assert np.isclose(factors["Nb"][0], 41.0, rtol=0.01)
    assert np.allclose(factors["Nb"], xraydb.f0("Nb", q / (4.0 * np.pi)))
    assert not np.allclose(factors["Nb"], xraydb.f0("Nb", q))
    assert np.all(np.diff(factors["Li"]) < 0)
    assert np.all(np.diff(factors["Nb"]) < 0)


def test_pair_weights_sum_to_one() -> None:
    q = np.linspace(1.0, 24.0, 47)
    result = q_dependent_weights({"Li": 1, "Nb": 1, "O": 1, "Cl": 4}, q)
    total = np.sum(np.stack(list(result.pair_weights.values())), axis=0)
    assert np.allclose(total, 1.0, atol=1e-12)
    assert all(np.all(weight >= 0) for weight in result.pair_weights.values())
