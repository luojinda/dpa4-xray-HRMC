"""Compare local environments in the three PZS MQ structures."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from dpa4_xray_hrmc.analysis.tetrahedra import tetrahedral_metrics
from dpa4_xray_hrmc.rmc.coordination import (
    CoordinationCutoffs,
    IncrementalCoordination,
)
from dpa4_xray_hrmc.xray_pdf.partial_rdf import compute_partial_rdfs


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/PZS_quench_new/three_mq_audit"
OUT.mkdir(parents=True, exist_ok=True)

PATHS = {
    "Original MQ": ROOT / "PDF/PZS_quench_222.cif",
    "Expanded new MQ": ROOT / "inputs/PZS_new_20261121/PZS_new_20261121.cif",
    "Quench new MQ": ROOT / "inputs/PZS_quench_new/PZS_quench_new.cif",
}
PAIRS = (("Zr", "Cl"), ("Zr", "O"), ("P", "O"), ("Si", "Cl"), ("Si", "O"))
COLORS = {
    "Original MQ": "#555555",
    "Expanded new MQ": "#377EB8",
    "Quench new MQ": "#E41A1C",
}
LINESTYLES = {
    "Original MQ": "--",
    "Expanded new MQ": "-.",
    "Quench new MQ": "-",
}


def pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


cutoffs = CoordinationCutoffs.from_mapping(
    json.loads((ROOT / "configs/pzs_phase2_5_coordination_cutoffs.json").read_text())
)
structures = {name: read(path) for name, path in PATHS.items()}
rdfs = {
    name: compute_partial_rdfs(atoms, 6.0, 0.02)
    for name, atoms in structures.items()
}

audit: dict[str, object] = {
    "coordination_cutoffs_A": cutoffs.as_mapping(),
    "rdf_dr_A": 0.02,
    "structures": {},
}

for name, atoms in structures.items():
    cn = IncrementalCoordination(atoms, cutoffs).current_snapshot()
    tetra = tetrahedral_metrics(atoms)
    audit["structures"][name] = {
        "path": str(PATHS[name].relative_to(ROOT)),
        "atoms": len(atoms),
        "composition": dict(Counter(atoms.get_chemical_symbols())),
        "cell_A": np.asarray(atoms.cell).tolist(),
        "volume_A3": atoms.get_volume(),
        "density_g_cm3": atoms.get_masses().sum() * 1.66053906660 / atoms.get_volume(),
        "P_CN_counts": dict(
            zip(*[x.tolist() for x in np.unique(cn.p_o, return_counts=True)], strict=True)
        ),
        "Si_CN_counts": dict(
            zip(*[x.tolist() for x in np.unique(cn.si_total, return_counts=True)], strict=True)
        ),
        "P_fourfold": float(np.mean(cn.p_o == 4)),
        "Si_fourfold": float(np.mean(cn.si_total == 4)),
        "Zr_total_hist": cn.zr_hist_total.tolist(),
        "Zr_O_hist": cn.zr_hist_o.tolist(),
        "tetrahedra": {
            element: {
                "tetrahedral_count": tetra[element]["tetrahedral_count"],
                "angle_RMS_deg": tetra[element]["angle_rms_from_ideal_deg"]["mean"],
                "CSM_percent": tetra[element]["fixed_center_csm_percent"]["mean"],
                "volume_A3": tetra[element]["volume_A3"]["mean"],
                "bonds": {
                    pair: {field: values[field] for field in ("mean", "std", "min", "max")}
                    for pair, values in tetra[element]["bond_lengths_A"].items()
                },
            }
            for element in ("P", "Si")
        },
    }

(OUT / "three_mq_local_environment_audit.json").write_text(
    json.dumps(audit, indent=2), encoding="utf-8"
)

fig, axes_array = plt.subplots(3, 2, figsize=(13, 14), dpi=300)
axes = axes_array.flat
for ax, (central, neighbor) in zip(axes, PAIRS, strict=False):
    key = pair_key(central, neighbor)
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, rdf in rdfs.items():
        g_r = rdf.g[key]
        n_r = np.cumsum(rdf.counts[key]) / rdf.element_counts[central]
        series[name] = (g_r, n_r)
        ax.plot(
            rdf.r_A,
            g_r,
            color=COLORS[name],
            linestyle=LINESTYLES[name],
            linewidth=1.6,
            label=name,
        )

    inset = inset_axes(ax, width="44%", height="43%", loc="upper right", borderpad=1.0)
    for name, (_, n_r) in series.items():
        inset.plot(
            rdfs[name].r_A,
            n_r,
            color=COLORS[name],
            linestyle=LINESTYLES[name],
            linewidth=1.2,
        )

    ax.set(
        xlim=(0.8, 5.0),
        xlabel=r"$r$ ($\AA$)",
        ylabel=rf"$g_{{{central}-{neighbor}}}(r)$",
        title=f"{central}-{neighbor}",
    )
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left")
    inset.set(
        xlim=(0.8, 5.0),
        xlabel=r"$r$",
        ylabel=rf"$N_{{{neighbor}\ around\ {central}}}$",
    )
    inset.tick_params(labelsize=7)
    inset.grid(True, linestyle=":", alpha=0.4)

    table_path = OUT / f"{central}_{neighbor}_rdf_CN.csv"
    names = list(PATHS)
    header = ["r_A"]
    for name in names:
        slug = name.lower().replace(" ", "_")
        header.extend((f"g_{slug}", f"N_{slug}"))
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(
            zip(
                rdfs[names[0]].r_A,
                *[array for name in names for array in series[name]],
                strict=True,
            )
        )

axes[-1].axis("off")
fig.suptitle("Local pair environments in three PZS MQ structures", fontsize=16)
fig.tight_layout()
fig.savefig(OUT / "three_mq_partial_rdf_and_CN.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(json.dumps(audit, indent=2))
