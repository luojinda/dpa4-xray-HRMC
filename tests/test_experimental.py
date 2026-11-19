import numpy as np

from dpa4_xray_hrmc.xray_pdf.experimental import compare_pdfs, range_metrics, read_pdfgetx_gr


def test_read_pdfgetx_gr_parses_header_and_data(tmp_path):
    path = tmp_path / "example.gr"
    path.write_text(
        "# xPDFsuite Configuration #\n"
        "[PDF]\n"
        "mode = xray\n"
        "qmin = 0.9\n"
        "qmax = 16.0056\n"
        "#### start data\n"
        "#L r(A) G(A^-2)\n"
        "0.00 0.0\n"
        "0.01 -0.1\n"
        "0.02 0.2\n",
        encoding="utf-8",
    )
    result = read_pdfgetx_gr(path)
    assert result.metadata["mode"] == "xray"
    assert float(result.metadata["qmax"]) == 16.0056
    np.testing.assert_allclose(result.r_A, [0.0, 0.01, 0.02])
    np.testing.assert_allclose(result.g_r_A2, [0.0, -0.1, 0.2])


def test_affine_alignment_and_metrics_recover_known_relation(tmp_path):
    r = np.linspace(0.0, 10.0, 1001)
    calculated = np.sin(2.0 * r) + 0.2 * np.cos(5.0 * r)
    observed = 0.4 * calculated + 0.1
    path = tmp_path / "synthetic.gr"
    rows = "\n".join(f"{x:.3f} {y:.12g}" for x, y in zip(r, observed, strict=True))
    path.write_text("[PDF]\nqmin = 0.9\nqmax = 16.0\n#### start data\n" + rows, encoding="utf-8")
    comparison = compare_pdfs(read_pdfgetx_gr(path), r, calculated, (1.0, 10.0))
    assert np.isclose(comparison.scale, 0.4, atol=1e-11)
    assert np.isclose(comparison.offset_A2, 0.1, atol=1e-11)
    metrics = range_metrics(comparison, ((1.0, 5.0), (1.0, 10.0)))
    assert metrics["ranges"]["1-10_A"]["aligned_rmse_A-2"] < 1e-10
    assert np.isclose(metrics["ranges"]["1-10_A"]["pearson_r"], 1.0)
