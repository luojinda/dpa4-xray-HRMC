from pathlib import Path

import numpy as np

from dpa4_xray_hrmc.xray_pdf.structure import load_structure


def test_lnoc_cif_explicit_p1_structure() -> None:
    path = Path(__file__).parents[1] / "PDF" / "LNOC_opt.cif"
    info = load_structure(path)
    assert info.n_atoms == 672
    assert info.element_counts == {"Cl": 384, "Li": 96, "Nb": 96, "O": 96}
    assert info.formula == "Cl384Li96Nb96O96"
    assert np.isclose(info.volume_A3, 15659.087309, rtol=2e-7)
    assert info.pbc == [True, True, True]
    assert len(info.occupancies) == 672
