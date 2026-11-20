"""Create compact parameter-calibration and 100k validation figures."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
directory = root / "results/PZS_phase3/calibration"
runs = json.loads((directory / "calibration_evaluations.json").read_text())
plots = directory / "plots"; plots.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors = {"Alternating": "tab:blue", "Inline": "tab:orange"}
for run in runs:
    marker = "o" if run["attempted"] == 10000 else "s"
    ax.scatter(run["delta_energy_from_MQ_eV"], run["Rw_no_scale"],
               color=colors[run["phase"]], marker=marker, s=65)
    label = run["tag"].replace("calibration_seed101_", "").replace("validation_100k_", "")
    ax.annotate(label, (run["delta_energy_from_MQ_eV"], run["Rw_no_scale"]),
                xytext=(4, 4), textcoords="offset points", fontsize=7)
ax.set_xlabel("DPA4 energy relative to MQ (eV / 896 atoms)")
ax.set_ylabel(r"$R_w$ (no scale, 1.5--10 A)")
ax.grid(alpha=.25); fig.tight_layout()
fig.savefig(plots / "energy_pdf_pareto.png", dpi=220); plt.close(fig)

selected = [r for r in runs if ("A500_F10" in r["tag"] or "T100" in r["tag"])]
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
labels = [r["tag"].replace("calibration_", "C_").replace("validation_100k_", "V_") for r in selected]
x = np.arange(len(selected))
for ax, key, ylabel in zip(axes,
    ("Rw_no_scale", "delta_energy_from_MQ_eV", "force_rms_eV_A"),
    (r"$R_w$", "Delta E from MQ (eV)", "Force RMS (eV/A)")):
    ax.bar(x, [r[key] for r in selected], color=[colors[r["phase"]] for r in selected])
    ax.set_ylabel(ylabel); ax.set_xticks(x, labels, rotation=90, fontsize=6); ax.grid(axis="y", alpha=.2)
fig.tight_layout(); fig.savefig(plots / "selected_10k_100k_comparison.png", dpi=220); plt.close(fig)
print(plots)
