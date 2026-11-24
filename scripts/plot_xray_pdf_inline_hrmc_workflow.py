"""Create a publication-ready schematic of a generic X-ray PDF + Inline HRMC workflow."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "xray_pdf_inline_hrmc_workflow"

NAVY = "#173B57"
BLUE = "#3977A8"
TEAL = "#2A9D8F"
ORANGE = "#E76F51"
GOLD = "#E9B949"
INK = "#22313D"
MUTED = "#657785"
LINE = "#B8C4CC"
PAPER = "#FFFFFF"
PALE_BLUE = "#EFF6FA"
PALE_TEAL = "#EDF8F6"
PALE_ORANGE = "#FFF2ED"
PALE_GOLD = "#FFF8E5"
PALE_GRAY = "#F5F7F8"


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": PAPER,
    }
)


def box(ax, xy, wh, face, edge, radius=0.018, lw=1.1, z=2):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=LINE, lw=1.25, rad=0.0, style="-|>", z=4):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def title(ax, x, y, panel, heading, color):
    ax.text(
        x,
        y,
        panel,
        ha="center",
        va="center",
        color="white",
        fontsize=8.2,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.24", facecolor=color, edgecolor=color),
        zorder=8,
    )
    ax.text(
        x + 0.025,
        y,
        heading,
        ha="left",
        va="center",
        color=INK,
        fontsize=8.0,
        fontweight="bold",
        zorder=8,
    )


def tiny_curve(ax, x, y, w, h, color, kind="pdf"):
    xx = np.linspace(0, 1, 240)
    if kind == "iq":
        yy = 0.53 + 0.20 * np.sin(15 * np.pi * xx) * np.exp(-1.6 * xx)
        yy += 0.08 * np.sin(43 * np.pi * xx) * np.exp(-2.8 * xx)
    elif kind == "energy":
        yy = 0.72 - 0.48 * (1 - np.exp(-3.0 * xx)) + 0.025 * np.sin(8 * np.pi * xx)
    else:
        yy = np.zeros_like(xx) + 0.43
        peaks = [(0.12, 0.27, 0.026), (0.25, 0.55, 0.038), (0.45, 0.28, 0.048), (0.68, 0.20, 0.065), (0.88, 0.12, 0.075)]
        for mu, amp, sig in peaks:
            yy += amp * np.exp(-0.5 * ((xx - mu) / sig) ** 2)
        yy -= 0.10 * np.exp(-0.5 * ((xx - 0.34) / 0.055) ** 2)
    ax.plot(x + w * xx, y + h * yy, color=color, lw=1.45, clip_on=False, zorder=7)
    ax.plot([x, x + w], [y + 0.10 * h, y + 0.10 * h], color=LINE, lw=0.55, zorder=6)


def atom_cluster(ax, center, scale=1.0):
    cx, cy = center
    pts = np.array(
        [
            (-0.62, 0.22), (-0.38, -0.35), (-0.15, 0.52), (0.05, -0.08),
            (0.31, 0.38), (0.55, -0.30), (0.74, 0.20), (-0.70, -0.46),
            (0.15, -0.58), (0.48, 0.67), (-0.34, 0.76), (0.80, -0.66),
        ]
    )
    radii = [0.010, 0.015, 0.011, 0.017, 0.012, 0.010, 0.016, 0.009, 0.012, 0.009, 0.013, 0.011]
    colors = [TEAL, ORANGE, BLUE, GOLD, TEAL, NAVY, ORANGE, BLUE, GOLD, TEAL, NAVY, ORANGE]
    for i, (px, py) in enumerate(pts):
        for j in range(i):
            qx, qy = pts[j]
            if (px - qx) ** 2 + (py - qy) ** 2 < 0.34:
                ax.plot(
                    [cx + scale * 0.045 * px, cx + scale * 0.045 * qx],
                    [cy + scale * 0.048 * py, cy + scale * 0.048 * qy],
                    color="#CBD4DA",
                    lw=0.7,
                    zorder=5,
                )
        ax.add_patch(
            Circle(
                (cx + scale * 0.045 * px, cy + scale * 0.048 * py),
                scale * radii[i],
                facecolor=colors[i],
                edgecolor="white",
                linewidth=0.45,
                zorder=7,
            )
        )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.09, 4.72), facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Header
    ax.text(
        0.035,
        0.955,
        "X-ray PDF-constrained Inline Hybrid Reverse Monte Carlo",
        color=NAVY,
        fontsize=14.2,
        fontweight="bold",
        va="center",
    )
    ax.text(
        0.035,
        0.922,
        "A general workflow for chemically complex amorphous and disordered materials",
        color=MUTED,
        fontsize=7.8,
        va="center",
    )
    ax.plot([0.035, 0.965], [0.895, 0.895], color=NAVY, lw=1.15)

    # Input panels
    input_y, input_h = 0.705, 0.155
    input_w = 0.285
    x1, x2, x3 = 0.035, 0.3575, 0.680
    box(ax, (x1, input_y), (input_w, input_h), PALE_BLUE, BLUE)
    box(ax, (x2, input_y), (input_w, input_h), PALE_TEAL, TEAL)
    box(ax, (x3, input_y), (input_w, input_h), PALE_GOLD, GOLD)
    title(ax, x1 + 0.021, input_y + input_h - 0.022, "A", "Experimental total scattering", BLUE)
    title(ax, x2 + 0.021, input_y + input_h - 0.022, "B", "Initial atomistic model", TEAL)
    title(ax, x3 + 0.021, input_y + input_h - 0.022, "C", "Physical & chemical priors", GOLD)

    tiny_curve(ax, x1 + 0.020, input_y + 0.038, 0.074, 0.065, BLUE, "iq")
    ax.text(x1 + 0.058, input_y + 0.018, r"$I(Q)$", ha="center", color=MUTED, fontsize=7.0)
    arrow(ax, (x1 + 0.103, input_y + 0.072), (x1 + 0.130, input_y + 0.072), BLUE, 1.0)
    ax.text(x1 + 0.147, input_y + 0.077, r"$S(Q)$", ha="center", va="bottom", color=INK, fontsize=7.4)
    ax.text(x1 + 0.147, input_y + 0.055, r"$F(Q)=Q[S(Q)-1]$", ha="center", color=MUTED, fontsize=6.6)
    arrow(ax, (x1 + 0.178, input_y + 0.072), (x1 + 0.205, input_y + 0.072), BLUE, 1.0)
    tiny_curve(ax, x1 + 0.213, input_y + 0.038, 0.054, 0.065, ORANGE, "pdf")
    ax.text(x1 + 0.240, input_y + 0.018, r"$G_{\mathrm{exp}}(r)$", ha="center", color=MUTED, fontsize=7.0)

    atom_cluster(ax, (x2 + 0.065, input_y + 0.072), 1.15)
    ax.text(x2 + 0.122, input_y + 0.092, "Multicomponent supercell", color=INK, fontsize=6.7, fontweight="bold")
    ax.text(x2 + 0.122, input_y + 0.067, "composition · density · PBC", color=MUTED, fontsize=6.1)
    ax.text(x2 + 0.122, input_y + 0.042, "crystal / MQ / random model", color=MUTED, fontsize=6.1)

    ax.text(x3 + 0.025, input_y + 0.095, r"$E_{\rm ML}(\mathbf{R})$", color=NAVY, fontsize=8.0, fontweight="bold")
    ax.text(x3 + 0.104, input_y + 0.095, "ML / empirical potential", color=MUTED, fontsize=6.0)
    ax.text(x3 + 0.025, input_y + 0.065, r"$d_{ij}>d^{\rm min}_{ij}$", color=ORANGE, fontsize=7.7, fontweight="bold")
    ax.text(x3 + 0.126, input_y + 0.065, "minimum-distance exclusion", color=MUTED, fontsize=5.8)
    ax.text(x3 + 0.025, input_y + 0.035, r"$C_m(\mathbf{R})$", color=TEAL, fontsize=7.7, fontweight="bold")
    ax.text(x3 + 0.091, input_y + 0.035, "coordination / motif priors", color=MUTED, fontsize=5.9)

    # Converging arrows into HRMC loop
    for xx in (x1 + input_w / 2, x2 + input_w / 2, x3 + input_w / 2):
        arrow(ax, (xx, input_y - 0.002), (xx, 0.665), NAVY, 1.05)

    # Main HRMC container
    loop_x, loop_y, loop_w, loop_h = 0.035, 0.255, 0.930, 0.400
    box(ax, (loop_x, loop_y), (loop_w, loop_h), "#FBFCFD", NAVY, radius=0.022, lw=1.35)
    ax.add_patch(Rectangle((loop_x, loop_y + loop_h - 0.055), loop_w, 0.055, facecolor=NAVY, edgecolor=NAVY, zorder=3))
    ax.text(loop_x + 0.020, loop_y + loop_h - 0.0275, "INLINE HRMC OPTIMIZATION LOOP", color="white", fontsize=9.4, fontweight="bold", va="center", zorder=8)
    ax.text(loop_x + loop_w - 0.020, loop_y + loop_h - 0.0275, "experiment + chemistry + energetics in every trial", color="#DCE8EF", fontsize=6.9, ha="right", va="center", zorder=8)

    node_y, node_h = 0.393, 0.160
    widths = [0.130, 0.130, 0.190, 0.215, 0.125]
    gaps = 0.020
    xs = [0.060]
    for ww in widths[:-1]:
        xs.append(xs[-1] + ww + gaps)
    faces = [PALE_TEAL, PALE_ORANGE, PALE_BLUE, PALE_GOLD, PALE_TEAL]
    edges = [TEAL, ORANGE, BLUE, GOLD, TEAL]
    heads = ["1  Trial move", "2  Fast screening", "3  X-ray PDF forward", "4  Composite increment", "5  Metropolis test"]
    for xx, ww, face, edge, head in zip(xs, widths, faces, edges, heads):
        box(ax, (xx, node_y), (ww, node_h), face, edge, radius=0.015, lw=1.0)
        ax.text(xx + 0.010, node_y + node_h - 0.028, head, color=edge, fontsize=6.7, fontweight="bold", va="center")

    # Node 1
    atom_cluster(ax, (xs[0] + 0.045, node_y + 0.073), 0.76)
    arrow(ax, (xs[0] + 0.064, node_y + 0.073), (xs[0] + 0.102, node_y + 0.103), ORANGE, 1.25)
    ax.text(xs[0] + 0.010, node_y + 0.018, r"$\mathbf{r}_i' = \mathbf{r}_i + \Delta\mathbf{r}$", color=INK, fontsize=6.8)
    ax.text(xs[0] + 0.010, node_y + 0.043, r"random atom; $|\Delta\mathbf{r}|\leq d_{\max}$", color=MUTED, fontsize=5.7)

    # Node 2
    ax.text(xs[1] + 0.010, node_y + 0.095, "Hard-sphere check", color=INK, fontsize=6.4, fontweight="bold")
    ax.text(xs[1] + 0.010, node_y + 0.070, "incremental coordination", color=INK, fontsize=5.9)
    ax.text(xs[1] + 0.010, node_y + 0.044, "reject forbidden overlaps", color=ORANGE, fontsize=5.8)
    ax.text(xs[1] + 0.010, node_y + 0.018, r"$\Delta C_m$: affected pairs only", color=MUTED, fontsize=5.6)

    # Node 3
    ax.text(xs[2] + 0.010, node_y + 0.105, r"Pairs involving atom $i$ only", color=INK, fontsize=6.0, fontweight="bold")
    ax.text(xs[2] + 0.010, node_y + 0.079, r"$\Delta g_{\alpha\beta}(r)\;\rightarrow\;\Delta S_X(Q)$", color=BLUE, fontsize=6.6)
    ax.text(xs[2] + 0.010, node_y + 0.052, r"Q-dependent $f_\alpha f_\beta$ weights", color=INK, fontsize=5.9)
    ax.text(xs[2] + 0.010, node_y + 0.026, r"FT $(Q_{\max})\rightarrow G_{\rm calc}(r)$", color=INK, fontsize=6.0)
    ax.text(xs[2] + 0.010, node_y + 0.005, r"scale $w$: none / fixed / bounded", color=MUTED, fontsize=5.5)

    # Node 4
    eq = (
        r"$\Delta\Omega=$" + "\n" +
        r"$\frac{\Delta\chi^2_{\rm PDF}}{2\sigma_{\rm PDF}^2}+\sum_m\lambda_m\Delta C_m$" + "\n" +
        r"$+\;\Delta E_{\rm ML}/(k_{\rm B}T_{\rm eff})$"
    )
    ax.text(xs[3] + 0.012, node_y + 0.077, eq, color=INK, fontsize=6.2, linespacing=1.30, va="center")
    ax.text(xs[3] + 0.012, node_y + 0.014, "PDF · chemistry · stability", color=MUTED, fontsize=5.8)

    # Node 5
    ax.text(xs[4] + widths[4] / 2, node_y + 0.098, r"$P_{\rm acc}=$", ha="center", color=INK, fontsize=6.5)
    ax.text(xs[4] + widths[4] / 2, node_y + 0.071, r"$\min[1,e^{-\Delta\Omega}]$", ha="center", color=NAVY, fontsize=7.5, fontweight="bold")
    ax.text(xs[4] + widths[4] / 2, node_y + 0.037, "accept → commit", ha="center", color=TEAL, fontsize=6.0, fontweight="bold")
    ax.text(xs[4] + widths[4] / 2, node_y + 0.015, "reject → rollback", ha="center", color=ORANGE, fontsize=6.0, fontweight="bold")

    # Arrows across nodes
    for i in range(4):
        arrow(ax, (xs[i] + widths[i] + 0.002, node_y + node_h / 2), (xs[i + 1] - 0.002, node_y + node_h / 2), NAVY, 1.25)

    # Iteration loop at bottom of container
    ax.text(0.505, 0.337, "accepted state", color=TEAL, fontsize=6.8, fontweight="bold", ha="center")
    arrow(ax, (xs[4] + widths[4] / 2, node_y - 0.005), (xs[4] + widths[4] / 2, 0.325), TEAL, 1.25)
    arrow(ax, (xs[4] + widths[4] / 2, 0.325), (xs[0] + widths[0] / 2, 0.325), TEAL, 1.25)
    arrow(ax, (xs[0] + widths[0] / 2, 0.325), (xs[0] + widths[0] / 2, node_y - 0.005), TEAL, 1.25)
    ax.text(0.505, 0.300, "repeat until target attempts / convergence / ensemble size", color=MUTED, fontsize=6.7, ha="center")

    # Outputs
    out_y, out_h = 0.065, 0.142
    out_w = 0.285
    for xx, face, edge in [(x1, PALE_BLUE, BLUE), (x2, PALE_GOLD, GOLD), (x3, PALE_TEAL, TEAL)]:
        box(ax, (xx, out_y), (out_w, out_h), face, edge, radius=0.015, lw=1.0)
    ax.text(x1 + 0.014, out_y + out_h - 0.026, "SCATTERING AGREEMENT", color=BLUE, fontsize=7.4, fontweight="bold")
    tiny_curve(ax, x1 + 0.018, out_y + 0.025, 0.095, 0.065, ORANGE, "pdf")
    tiny_curve(ax, x1 + 0.018, out_y + 0.025, 0.095, 0.065, NAVY, "pdf")
    ax.text(x1 + 0.132, out_y + 0.074, r"$G_{\rm exp}(r)$ vs $G_{\rm calc}(r)$", color=INK, fontsize=6.1)
    ax.text(x1 + 0.132, out_y + 0.047, r"$R_w$, fit range, scale $w$", color=MUTED, fontsize=5.9)
    ax.text(x1 + 0.132, out_y + 0.022, "distance-resolved residual", color=MUTED, fontsize=5.9)

    ax.text(x2 + 0.014, out_y + out_h - 0.026, "TRAJECTORY & ROBUSTNESS", color="#A27300", fontsize=7.4, fontweight="bold")
    tiny_curve(ax, x2 + 0.018, out_y + 0.025, 0.095, 0.065, ORANGE, "energy")
    ax.text(x2 + 0.132, out_y + 0.074, r"$E/N$, $R_w$, scale, acceptance", color=INK, fontsize=6.1)
    ax.text(x2 + 0.132, out_y + 0.047, "multiple seeds / uncertainty", color=MUTED, fontsize=5.9)
    ax.text(x2 + 0.132, out_y + 0.022, "incremental-vs-full check", color=MUTED, fontsize=5.9)

    ax.text(x3 + 0.014, out_y + out_h - 0.026, "REFINED STRUCTURAL ENSEMBLE", color=TEAL, fontsize=7.4, fontweight="bold")
    atom_cluster(ax, (x3 + 0.064, out_y + 0.063), 1.0)
    ax.text(x3 + 0.127, out_y + 0.076, "partial RDFs + coordination", color=INK, fontsize=6.1)
    ax.text(x3 + 0.127, out_y + 0.049, "bond lengths / angles / motifs", color=MUTED, fontsize=5.9)
    ax.text(x3 + 0.127, out_y + 0.023, "structures + ensemble statistics", color=MUTED, fontsize=5.9)

    # Downstream arrows
    for xx in (x1 + out_w / 2, x2 + out_w / 2, x3 + out_w / 2):
        arrow(ax, (xx, loop_y - 0.002), (xx, out_y + out_h + 0.002), NAVY, 1.05)

    stem = OUT / "xray_pdf_inline_hrmc_workflow"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
