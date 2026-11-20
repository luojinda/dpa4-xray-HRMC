"""Compare selected final 100k Alternating/Inline trajectories and geometry."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

from dpa4_xray_hrmc.analysis.tetrahedra import tetrahedral_metrics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/PZS_phase3"
ALT = RESULTS / "phase3B/validation_100k_seed303_A500_F10"
INLINE = RESULTS / "phase3C/validation_100k_seed303_T100"
OUT = RESULTS / "calibration"
PLOTS = OUT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
N_ATOMS = 896


def load_trajectory(directory: Path, method: str):
    trace = json.loads((directory / "trace.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if int(trace[-1]["attempted"]) != int(summary["attempted"]):
        trace.append(summary)
    rows = []
    for row in trace:
        rows.append({
            "method": method,
            "attempts": int(row["attempted"]),
            "Rw_no_scale": float(row["Rw_no_scale"]),
            "energy_eV": float(row["energy_eV"]),
        })
    rows.sort(key=lambda item: item["attempts"])
    return rows


alternating = load_trajectory(ALT, "Alternating 500/10 seed303 final")
inline = load_trajectory(INLINE, "Inline 100 K seed303 final")
mq_energy = alternating[0]["energy_eV"]
mq_rw = alternating[0]["Rw_no_scale"]
if inline[0]["attempts"] != 0:
    inline.insert(0, {
        "method": "Inline 100 K seed303 final", "attempts": 0,
        "Rw_no_scale": mq_rw, "energy_eV": mq_energy,
    })
all_rows = alternating + inline
for row in all_rows:
    row["delta_E_meV_atom"] = (row["energy_eV"] - mq_energy) * 1000.0 / N_ATOMS

with (OUT / "selected_final_100k_trajectories.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
    writer.writeheader(); writer.writerows(all_rows)

fig, left = plt.subplots(figsize=(11, 5), dpi=300)
right = left.twinx()
styles = {
    "Alternating 500/10 seed303 final": ("#15572C", "Alternating"),
    "Inline 100 K seed303 final": ("#FF0025", "Inline"),
}
for rows in (alternating, inline):
    method = rows[0]["method"]
    color, label = styles[method]
    x = np.asarray([row["attempts"] for row in rows])
    rw = np.asarray([row["Rw_no_scale"] for row in rows])
    de = np.asarray([(row["energy_eV"] - mq_energy) * 1000.0 / N_ATOMS for row in rows])
    left.plot(x, rw, color=color, lw=2.0, label=f"{label} $R_w$")
    right.plot(x, de, color=color, lw=1.5, ls="--", label=rf"{label} $\Delta E$")
left.scatter([0], [mq_rw], color="black", marker="*", s=120, zorder=5, label="MQ")
left.set_xlabel("MC attempts", fontsize=14)
left.set_ylabel(r"$R_w$ (no scale, 1.5--10 A)", fontsize=14)
right.set_ylabel(r"$\Delta E$ (meV/atom, relative to MQ)", fontsize=14)
left.set_xlim(0, 100000)
left.tick_params(direction="in", labelsize=14)
right.tick_params(direction="in", labelsize=14)
left.grid(True, linestyle=":", alpha=0.5)
handles1, labels1 = left.get_legend_handles_labels()
handles2, labels2 = right.get_legend_handles_labels()
left.legend(handles1 + handles2, labels1 + labels2, loc="center right", frameon=True, fontsize=12)
left.set_title("Selected final 100k trajectories: PDF residual and DPA4 energy", fontsize=14, pad=10)
fig.tight_layout()
fig.savefig(PLOTS / "selected_final_100k_Rw_deltaE_vs_attempts.png", dpi=300, bbox_inches="tight")
plt.close(fig)

structures = {
    "MQ": read(ROOT / "PDF/PZS_quench_222.cif"),
    "Alternating\nseed303 final": read(ALT / "final_structure.extxyz"),
    "Inline 100 K\nseed303 final": read(INLINE / "final_structure.extxyz"),
}
geometry_rows = []
for label, atoms in structures.items():
    values = tetrahedral_metrics(atoms)
    geometry_rows.append({
        "structure": label.replace("\n", " "),
        "P_angle_RMS_deg": values["P"]["angle_rms_from_ideal_deg"]["mean"],
        "Si_angle_RMS_deg": values["Si"]["angle_rms_from_ideal_deg"]["mean"],
        "P_CSM_percent": values["P"]["fixed_center_csm_percent"]["mean"],
        "Si_CSM_percent": values["Si"]["fixed_center_csm_percent"]["mean"],
    })
with (OUT / "selected_final_100k_geometry.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(geometry_rows[0]))
    writer.writeheader(); writer.writerows(geometry_rows)

metrics = [
    ("P_angle_RMS_deg", "P angle RMS (deg)"),
    ("Si_angle_RMS_deg", "Si angle RMS (deg)"),
    ("P_CSM_percent", "P CSM (%)"),
    ("Si_CSM_percent", "Si CSM (%)"),
]
fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=300)
colors = ["#555555", "#15572C", "#FF0025"]
labels = list(structures)
for ax, (key, title) in zip(axes.flat, metrics, strict=True):
    values = [row[key] for row in geometry_rows]
    bars = ax.bar(labels, values, color=colors, width=0.65)
    ax.set_title(title, fontsize=14, pad=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.tick_params(direction="in", axis="both", labelsize=12)
    ax.set_ylim(0, max(values) * 1.18)
fig.suptitle("Tetrahedral geometry: MQ and selected final 100k structures", fontsize=14)
fig.tight_layout()
fig.savefig(PLOTS / "selected_final_100k_tetrahedral_geometry.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(json.dumps({
    "trajectory_plot": str(PLOTS / "selected_final_100k_Rw_deltaE_vs_attempts.png"),
    "geometry_plot": str(PLOTS / "selected_final_100k_tetrahedral_geometry.png"),
    "geometry": geometry_rows,
}, indent=2))
