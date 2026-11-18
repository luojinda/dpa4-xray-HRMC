import numpy as np

from dpa4_xray_hrmc.xray_pdf.form_factors import q_dependent_weights
from dpa4_xray_hrmc.xray_pdf.partial_rdf import PartialRDF
from dpa4_xray_hrmc.xray_pdf.thermal import biso_to_uiso, pair_debye_waller
from dpa4_xray_hrmc.xray_pdf.transform import forward_transform, partial_sq_from_rdf, sine_transform


def _unit_rdf() -> PartialRDF:
    edges = np.linspace(0, 10, 1001)
    return PartialRDF(
        r_A=(edges[:-1] + edges[1:]) / 2,
        edges_A=edges,
        g={("Si", "Si"): np.ones(1000)},
        counts={("Si", "Si"): np.zeros(1000, dtype=int)},
        element_counts={"Si": 8},
        volume_A3=160.0,
        dr_A=0.01,
        rcut_A=10.0,
        translation_bounds=(2, 2, 2),
        n_translations=125,
    )


def test_unit_rdf_has_zero_reduced_signal() -> None:
    rdf = _unit_rdf()
    q = np.linspace(1, 24, 461)
    partial = partial_sq_from_rdf(rdf, q)
    assert np.allclose(partial[("Si", "Si")], 0.0)
    result = forward_transform(rdf, q_dependent_weights(rdf.element_counts, q), np.linspace(0, 10, 1001), 0.3)
    assert np.allclose(result.s_q, 1.0)
    assert np.allclose(result.f_q_Ainv, 0.0)
    assert np.allclose(result.g_r_A2, 0.0)


def test_thermal_pair_factor_and_biso_conversion() -> None:
    q = np.array([0.0, 10.0, 20.0])
    u = float(biso_to_uiso(0.3))
    assert np.isclose(u, 0.3 / (8 * np.pi**2))
    damping = pair_debye_waller(q, 0.3, 0.3)
    assert np.allclose(damping, np.exp(-(q**2) * u))
    assert damping[0] == 1.0
    assert np.all(np.diff(damping) < 0)


def test_sine_transform_matches_analytic_linear_fq() -> None:
    q = np.linspace(1.0, 24.0, 23001)
    r = np.array([0.7, 1.3, 3.2])
    numerical = sine_transform(q, q, r)
    q0, q1 = q[0], q[-1]
    analytic_integral = (
        (-q1 * np.cos(q1 * r) / r + np.sin(q1 * r) / r**2)
        - (-q0 * np.cos(q0 * r) / r + np.sin(q0 * r) / r**2)
    )
    assert np.allclose(numerical, (2.0 / np.pi) * analytic_integral, rtol=2e-6, atol=2e-6)
