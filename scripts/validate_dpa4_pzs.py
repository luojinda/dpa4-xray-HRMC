#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from ase.io import read

from dpa4_xray_hrmc.mlip.dpa4_backend import DPA4Backend


def force_metrics(forces: np.ndarray) -> tuple[float, float, int]:
    norms = np.linalg.norm(forces, axis=1)
    return float(np.sqrt(np.mean(norms**2))), float(np.max(norms)), int(np.argmax(norms))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "results/PZS_phase3/phase3A"
    out.mkdir(parents=True, exist_ok=True)
    backend = DPA4Backend(os.environ["DPA4_MODEL_PATH"], required_elements=("Cl", "Li", "O", "P", "Si", "Zr"))
    structures = [("MQ", root / "PDF/PZS_quench_222.cif")]
    for seed in (101, 202, 303):
        structures.append((f"phase2_5_no_scale_final_seed_{seed}", root / f"results/PZS_phase2_5/formal_1M/no_scale/seed_{seed}/final_structure.extxyz"))
    rows = []
    mq_energy = None
    for label, path in structures:
        atoms = read(path)
        result = backend.evaluate(atoms)
        rms, maximum, index = force_metrics(result.forces_eV_A)
        if mq_energy is None:
            mq_energy = result.energy_eV
        rows.append({"label": label, "path": str(path), "energy_eV": result.energy_eV, "energy_eV_atom": result.energy_eV/len(atoms), "delta_E_vs_MQ_eV": result.energy_eV-mq_energy, "delta_E_vs_MQ_eV_atom": (result.energy_eV-mq_energy)/len(atoms), "force_rms_eV_A": rms, "force_max_eV_A": maximum, "force_max_atom": index, "wall_s": result.wall_s})
    mq = read(structures[0][1])
    repeats = [backend.evaluate(mq) for _ in range(3)]
    repeatability = {"energy_range_eV": max(x.energy_eV for x in repeats)-min(x.energy_eV for x in repeats), "force_max_abs_range_eV_A": float(np.max(np.ptp(np.stack([x.forces_eV_A for x in repeats]), axis=0)))}
    rng = np.random.default_rng(20261120)
    perturbed = []
    for magnitude in (0.005, 0.010, 0.020):
        trial = mq.copy(); trial.positions += rng.normal(scale=magnitude, size=trial.positions.shape)
        result = backend.evaluate(trial); rms, maximum, index = force_metrics(result.forces_eV_A)
        perturbed.append({"rms_displacement_A": magnitude, "energy_eV": result.energy_eV, "delta_E_eV": result.energy_eV-mq_energy, "force_rms_eV_A": rms, "force_max_eV_A": maximum, "force_max_atom": index})
    base = backend.evaluate(mq)
    fd_rows = []
    cases = [(0, 0), (160, 1), (224, 2), (832, 0)]
    for atom_index, axis in cases:
        for delta in (1e-2, 1e-3):
            plus = mq.copy(); minus = mq.copy()
            plus.positions[atom_index, axis] += delta; minus.positions[atom_index, axis] -= delta
            ep = backend.evaluate(plus).energy_eV; em = backend.evaluate(minus).energy_eV
            fd_force = -(ep-em)/(2*delta); analytic = float(base.forces_eV_A[atom_index, axis])
            fd_rows.append({"atom_index": atom_index, "element": mq[atom_index].symbol, "axis": axis, "delta_A": delta, "analytic_force_eV_A": analytic, "fd_force_eV_A": fd_force, "abs_error_eV_A": abs(fd_force-analytic), "relative_error": abs(fd_force-analytic)/max(abs(analytic), 1e-12)})
    with (out / "single_point_energy_force.csv").open("w", newline="") as handle:
        writer=csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    with (out / "force_fd_check.csv").open("w", newline="") as handle:
        writer=csv.DictWriter(handle, fieldnames=fd_rows[0]); writer.writeheader(); writer.writerows(fd_rows)
    summary={"repeatability":repeatability,"perturbations":perturbed,"finite_difference":fd_rows}
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"single_points":rows,**summary}, indent=2))


if __name__ == "__main__": main()
