from pathlib import Path

import numpy as np

from dpa4_xray_hrmc.xray_pdf.form_factors import q_dependent_weights
from dpa4_xray_hrmc.xray_pdf.partial_rdf import compute_partial_rdfs
from dpa4_xray_hrmc.xray_pdf.structure import load_structure
from dpa4_xray_hrmc.xray_pdf.transform import forward_transform


def test_real_lnoc_short_end_to_end_has_finite_nonzero_pdf() -> None:
    path = Path(__file__).parents[1] / "PDF" / "LNOC_opt.cif"
    info = load_structure(path)
    rdf = compute_partial_rdfs(info.atoms, rcut_A=5.0, dr_A=0.05)
    assert len(rdf.g) == 10
    q = np.arange(1.0, 12.0 + 0.05, 0.1)
    result = forward_transform(rdf, q_dependent_weights(rdf.element_counts, q), np.linspace(0, 6, 301), 0.3)
    zero_biso = forward_transform(rdf, q_dependent_weights(rdf.element_counts, q), np.linspace(0, 6, 301), 0.0)
    assert np.all(np.isfinite(result.g_r_A2))
    assert np.max(np.abs(result.g_r_A2)) > 0.1
    assert result.g_r_A2[0] == 0.0
    assert not np.allclose(result.g_r_A2, zero_biso.g_r_A2)
