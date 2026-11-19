"""Read PDFGetX-style reduced PDFs and compare them with calculated G(r)."""

from __future__ import annotations

import configparser
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dpa4_xray_hrmc_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ExperimentalPDF:
    """An experimental reduced PDF and metadata parsed from its header."""

    r_A: np.ndarray
    g_r_A2: np.ndarray
    metadata: dict[str, str]
    path: Path


@dataclass(frozen=True)
class PDFComparison:
    """Experimental and calculated PDFs sampled on the experimental r grid."""

    r_A: np.ndarray
    experimental_A2: np.ndarray
    calculated_raw_A2: np.ndarray
    calculated_aligned_A2: np.ndarray
    scale: float
    offset_A2: float
    fit_range_A: tuple[float, float]


def read_pdfgetx_gr(path: str | Path) -> ExperimentalPDF:
    """Read a two-column PDFGetX/xPDFsuite .gr file and its optional [PDF] header."""

    source = Path(path)
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().lower() == "#### start data")
    except StopIteration as exc:
        raise ValueError(f"No '#### start data' marker in {source}") from exc

    header_lines = [line for line in lines[:start] if not line.lstrip().startswith("#")]
    parser = configparser.ConfigParser(interpolation=None)
    if header_lines:
        parser.read_string("\n".join(header_lines))
    metadata = dict(parser["PDF"]) if parser.has_section("PDF") else {}

    data_text = "\n".join(lines[start + 1 :])
    data = np.loadtxt(io.StringIO(data_text), comments="#", ndmin=2)
    if data.shape[1] < 2 or data.shape[0] < 2:
        raise ValueError(f"Expected at least two numeric columns and rows in {source}")
    r_A = np.asarray(data[:, 0], dtype=float)
    g_r_A2 = np.asarray(data[:, 1], dtype=float)
    if not np.all(np.isfinite(r_A)) or not np.all(np.isfinite(g_r_A2)):
        raise ValueError(f"Non-finite PDF data in {source}")
    if not np.all(np.diff(r_A) > 0.0):
        raise ValueError(f"r values must be strictly increasing in {source}")
    return ExperimentalPDF(r_A, g_r_A2, metadata, source.resolve())


def load_calculated_gr(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a calculated two-column r, G(r) table written by this project."""

    source = Path(path)
    data = np.loadtxt(source, comments="#", ndmin=2)
    if data.shape[1] < 2 or data.shape[0] < 2:
        raise ValueError(f"Expected at least two numeric columns and rows in {source}")
    r_A = np.asarray(data[:, 0], dtype=float)
    g_r_A2 = np.asarray(data[:, 1], dtype=float)
    if not np.all(np.isfinite(data[:, :2])) or not np.all(np.diff(r_A) > 0.0):
        raise ValueError(f"Invalid calculated PDF grid in {source}")
    return r_A, g_r_A2


def compare_pdfs(
    experimental: ExperimentalPDF,
    calculated_r_A: np.ndarray,
    calculated_g_A2: np.ndarray,
    fit_range_A: tuple[float, float] = (1.0, 10.0),
) -> PDFComparison:
    """Interpolate the model and determine one least-squares scale and offset."""

    overlap = (experimental.r_A >= calculated_r_A[0]) & (experimental.r_A <= calculated_r_A[-1])
    r_A = experimental.r_A[overlap]
    observed = experimental.g_r_A2[overlap]
    calculated = np.interp(r_A, calculated_r_A, calculated_g_A2)
    fit = (r_A >= fit_range_A[0]) & (r_A <= fit_range_A[1])
    if np.count_nonzero(fit) < 3:
        raise ValueError(f"Fewer than three comparison points in fit range {fit_range_A}")
    design = np.column_stack((calculated[fit], np.ones(np.count_nonzero(fit))))
    scale, offset = np.linalg.lstsq(design, observed[fit], rcond=None)[0]
    aligned = scale * calculated + offset
    return PDFComparison(r_A, observed, calculated, aligned, float(scale), float(offset), fit_range_A)


def range_metrics(comparison: PDFComparison, ranges_A: tuple[tuple[float, float], ...]) -> dict[str, Any]:
    """Return absolute-intensity and aligned-shape metrics for each requested range."""

    output: dict[str, Any] = {
        "alignment": {
            "fit_range_A": list(comparison.fit_range_A),
            "scale": comparison.scale,
            "offset_A-2": comparison.offset_A2,
            "meaning": "diagnostic least-squares alignment only; not applied to the raw calculation",
        },
        "ranges": {},
    }
    for lo, hi in ranges_A:
        mask = (comparison.r_A >= lo) & (comparison.r_A <= hi)
        if np.count_nonzero(mask) < 3:
            raise ValueError(f"Fewer than three points in metric range {(lo, hi)}")
        exp = comparison.experimental_A2[mask]
        raw = comparison.calculated_raw_A2[mask]
        aligned = comparison.calculated_aligned_A2[mask]
        key = f"{lo:g}-{hi:g}_A"
        exp_std = float(np.std(exp))
        raw_std = float(np.std(raw))
        output["ranges"][key] = {
            "points": int(np.count_nonzero(mask)),
            "experimental_min_A-2": float(np.min(exp)),
            "experimental_max_A-2": float(np.max(exp)),
            "experimental_rms_A-2": float(np.sqrt(np.mean(exp**2))),
            "calculated_raw_min_A-2": float(np.min(raw)),
            "calculated_raw_max_A-2": float(np.max(raw)),
            "calculated_raw_rms_A-2": float(np.sqrt(np.mean(raw**2))),
            "raw_rms_ratio_calc_over_exp": float(np.sqrt(np.mean(raw**2)) / np.sqrt(np.mean(exp**2))),
            "raw_rmse_A-2": float(np.sqrt(np.mean((raw - exp) ** 2))),
            "aligned_rmse_A-2": float(np.sqrt(np.mean((aligned - exp) ** 2))),
            "pearson_r": float(np.corrcoef(exp, raw)[0, 1]) if exp_std > 0 and raw_std > 0 else None,
        }
    return output


def write_comparison_table(path: str | Path, comparison: PDFComparison) -> None:
    data = np.column_stack(
        (
            comparison.r_A,
            comparison.experimental_A2,
            comparison.calculated_raw_A2,
            comparison.calculated_aligned_A2,
            comparison.calculated_aligned_A2 - comparison.experimental_A2,
        )
    )
    np.savetxt(
        path,
        data,
        fmt="%.10g",
        header="r_A G_experiment_A-2 G_calculated_raw_A-2 G_calculated_aligned_A-2 aligned_minus_experiment_A-2",
    )


def write_metrics(path: str | Path, metrics: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_comparison(
    comparison: PDFComparison,
    output: str | Path,
    x_range_A: tuple[float, float],
    sample: str,
    experiment_q_range_Ainv: tuple[float, float] | None,
    model_q_range_Ainv: tuple[float, float] | None,
    biso_A2: float | None,
) -> None:
    """Plot raw absolute intensity and the diagnostic affine-aligned comparison."""

    mask = (comparison.r_A >= x_range_A[0]) & (comparison.r_A <= x_range_A[1])
    exp = comparison.experimental_A2[mask]
    raw = comparison.calculated_raw_A2[mask]
    aligned = comparison.calculated_aligned_A2[mask]
    corr = float(np.corrcoef(exp, raw)[0, 1])
    raw_rmse = float(np.sqrt(np.mean((raw - exp) ** 2)))
    aligned_rmse = float(np.sqrt(np.mean((aligned - exp) ** 2)))

    fig, axes = plt.subplots(2, 1, figsize=(8.3, 7.0), sharex=True, constrained_layout=True)
    axes[0].plot(comparison.r_A, comparison.experimental_A2, color="black", linewidth=1.25, label="Experiment")
    axes[0].plot(comparison.r_A, comparison.calculated_raw_A2, color="#145da0", linewidth=1.05, label="Calculated, raw")
    axes[0].set_ylabel(r"$G(r)$ ($\AA^{-2}$)")
    axes[0].set_title(f"{sample} X-ray PDF: absolute intensity")
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(comparison.r_A, comparison.experimental_A2, color="black", linewidth=1.25, label="Experiment")
    axes[1].plot(
        comparison.r_A,
        comparison.calculated_aligned_A2,
        color="#c23b22",
        linewidth=1.05,
        linestyle="--",
        label="Calculated, scale + offset aligned",
    )
    axes[1].axhline(0.0, color="0.55", linewidth=0.6)
    axes[1].set_ylabel(r"$G(r)$ ($\AA^{-2}$)")
    axes[1].set_xlabel(r"$r$ ($\AA$)")
    axes[1].set_title("Shape comparison (diagnostic alignment only)")
    axes[1].legend(frameon=False, loc="upper right")

    for ax in axes:
        ax.set_xlim(*x_range_A)
        ax.grid(alpha=0.18, linewidth=0.5)
    all_values = np.concatenate((exp, raw))
    span = float(np.ptp(all_values))
    axes[0].set_ylim(float(np.min(all_values) - 0.06 * span), float(np.max(all_values) + 0.09 * span))
    aligned_values = np.concatenate((exp, aligned))
    aligned_span = float(np.ptp(aligned_values))
    axes[1].set_ylim(
        float(np.min(aligned_values) - 0.06 * aligned_span),
        float(np.max(aligned_values) + 0.09 * aligned_span),
    )

    q_text = []
    if experiment_q_range_Ainv is not None:
        q_text.append(rf"exp $Q={experiment_q_range_Ainv[0]:g}$--${experiment_q_range_Ainv[1]:g}\ \AA^{{-1}}$")
    if model_q_range_Ainv is not None:
        q_text.append(rf"model $Q={model_q_range_Ainv[0]:g}$--${model_q_range_Ainv[1]:g}\ \AA^{{-1}}$")
    if biso_A2 is not None:
        q_text.append(rf"$B_{{iso}}={biso_A2:g}\ \AA^2$")
    q_text.append(rf"raw RMSE={raw_rmse:.3g}, $r_P$={corr:.3f}")
    axes[0].text(
        0.012,
        0.96,
        "\n".join(q_text),
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.86, "edgecolor": "0.75"},
    )
    axes[1].text(
        0.012,
        0.96,
        rf"$G_{{aligned}}={comparison.scale:.5g}G_{{calc}}{comparison.offset_A2:+.5g}$" + "\n" + rf"aligned RMSE={aligned_rmse:.3g}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.86, "edgecolor": "0.75"},
    )
    fig.savefig(output, dpi=360)
    plt.close(fig)
