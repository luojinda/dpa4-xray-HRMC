import numpy as np
from dpa4_xray_hrmc.analysis.tetrahedra import fixed_center_csm


def test_ideal_tetrahedron_has_zero_fixed_center_csm():
    points=np.asarray([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],dtype=float)
    assert fixed_center_csm(points)<1e-12
