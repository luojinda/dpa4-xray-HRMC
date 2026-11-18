"""Publication-quality output plots for the calculated reduced PDF."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dpa4_xray_hrmc_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_pdf(
    r_A: np.ndarray,
    g_r_A2: np.ndarray,
    output: str | Path,
    x_range_A: tuple[float, float],
    q_range_Ainv: tuple[float, float],
    biso_A2: float,
    lorch_mod: bool,
    rcut_A: float,
    sample: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.plot(r_A, g_r_A2, color="#145da0", linewidth=1.35)
    ax.axhline(0.0, color="0.45", linewidth=0.65)
    ax.set_xlim(*x_range_A)
    mask = (r_A >= x_range_A[0]) & (r_A <= x_range_A[1])
    if np.any(mask):
        ymin, ymax = np.min(g_r_A2[mask]), np.max(g_r_A2[mask])
        margin = max(0.06 * (ymax - ymin), 0.5)
        ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_xlabel(r"$r$ ($\AA$)")
    ax.set_ylabel(r"$G(r)$ ($\AA^{-2}$)")
    ax.set_title(f"{sample} periodic structure: calculated X-ray PDF")
    lorch_text = "Lorch" if lorch_mod else "No Lorch"
    annotation = (
        rf"X-ray, $Q={q_range_Ainv[0]:g}$--${q_range_Ainv[1]:g}\ \AA^{{-1}}$"
        + "\n"
        + rf"$B_{{iso}}={biso_A2:g}\ \AA^2$, {lorch_text}, $r_c={rcut_A:g}\ \AA$"
    )
    ax.text(
        0.985,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    ax.grid(alpha=0.18, linewidth=0.5)
    fig.savefig(output, dpi=360)
    plt.close(fig)
