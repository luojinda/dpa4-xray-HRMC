"""Fixed-cell ASE/FIRE relaxation of MQ3 with the PZS LoRA DPA4 model."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from ase.io import read, write

from dpa4_xray_hrmc.mlip.dpa4_backend import DPA4Backend
from dpa4_xray_hrmc.mlip.fire_relaxation import run_fixed_cell_fire
from dpa4_xray_hrmc.rmc.hard_sphere import minimum_image_vectors


ROOT = Path(__file__).resolve().parents[1]
MQ3_PATH = ROOT / "inputs/PZS_MQ_3/PZS_equilibrated.cif"
BASELINE_PATH = (
    ROOT
    / "results/PZS_phase3/phase3C/PZS_MQ_3/"
    "validation_100k_seed101_T100_d0p1_r1p4_10p0A_lora_bounded_dynamic_p_total/"
    "final_structure.extxyz"
)
OUT = ROOT / "results/PZS_phase3/relaxations/PZS_MQ_3_lora_fixed_cell"
OUT.mkdir(parents=True, exist_ok=True)

FMAX_EV_A = 0.05
MAX_STEPS = 2000
MAXSTEP_A = 0.05
DT_FS_LIKE = 0.05


def force_metrics(forces: np.ndarray) -> tuple[float, float]:
    norms = np.linalg.norm(forces, axis=1)
    return float(np.sqrt(np.mean(norms**2))), float(np.max(norms))


def main() -> None:
    model_path = Path(os.environ["DPA4_MODEL_PATH"])
    backend = DPA4Backend(
        model_path,
        required_elements=("Cl", "Li", "O", "P", "Si", "Zr"),
    )
    initial = read(MQ3_PATH)
    baseline = read(BASELINE_PATH)
    initial_cell = np.asarray(initial.cell).copy()
    initial_positions = initial.positions.copy()

    initial_eval = backend.evaluate(initial)
    baseline_eval = backend.evaluate(baseline)
    initial_force_rms, initial_force_max = force_metrics(initial_eval.forces_eV_A)
    baseline_force_rms, baseline_force_max = force_metrics(baseline_eval.forces_eV_A)

    result = run_fixed_cell_fire(
        initial,
        backend,
        MAX_STEPS,
        fmax_eV_A=FMAX_EV_A,
        maxstep_A=MAXSTEP_A,
        dt_fs_like=DT_FS_LIKE,
        logfile=str(OUT / "fire.log"),
    )
    final_eval = backend.evaluate(result.atoms)
    final_force_rms, final_force_max = force_metrics(final_eval.forces_eV_A)

    if not np.array_equal(initial_cell, np.asarray(result.atoms.cell)):
        raise RuntimeError("Fixed-cell relaxation changed the cell")
    if initial.get_chemical_symbols() != result.atoms.get_chemical_symbols():
        raise RuntimeError("Relaxation changed atom ordering or species")

    displacements = np.linalg.norm(
        minimum_image_vectors(result.atoms.positions - initial_positions, initial_cell),
        axis=1,
    )
    atom_count = len(initial)
    structures = {
        "initial_mq3": {
            "energy_eV": initial_eval.energy_eV,
            "energy_eV_atom": initial_eval.energy_eV / atom_count,
            "force_rms_eV_A": initial_force_rms,
            "force_max_eV_A": initial_force_max,
        },
        "relaxed_mq3": {
            "energy_eV": final_eval.energy_eV,
            "energy_eV_atom": final_eval.energy_eV / atom_count,
            "force_rms_eV_A": final_force_rms,
            "force_max_eV_A": final_force_max,
        },
        "baseline_hrmc": {
            "energy_eV": baseline_eval.energy_eV,
            "energy_eV_atom": baseline_eval.energy_eV / atom_count,
            "force_rms_eV_A": baseline_force_rms,
            "force_max_eV_A": baseline_force_max,
        },
    }
    summary = {
        "input_structure": str(MQ3_PATH.relative_to(ROOT)),
        "baseline_structure": str(BASELINE_PATH.relative_to(ROOT)),
        "model": backend.manifest(),
        "optimizer": {
            "name": "ASE FIRE",
            "fixed_cell": True,
            "fmax_eV_A": FMAX_EV_A,
            "max_steps": MAX_STEPS,
            "maxstep_A": MAXSTEP_A,
            "dt_fs_like": DT_FS_LIKE,
            "executed_steps": result.executed_steps,
            "converged": result.converged,
            "wall_s": result.wall_s,
        },
        "structures": structures,
        "energy_differences": {
            "relaxed_minus_initial_eV": final_eval.energy_eV - initial_eval.energy_eV,
            "relaxed_minus_initial_meV_atom": (
                final_eval.energy_eV - initial_eval.energy_eV
            )
            * 1000.0
            / atom_count,
            "baseline_minus_initial_eV": baseline_eval.energy_eV
            - initial_eval.energy_eV,
            "baseline_minus_initial_meV_atom": (
                baseline_eval.energy_eV - initial_eval.energy_eV
            )
            * 1000.0
            / atom_count,
            "relaxed_minus_baseline_eV": final_eval.energy_eV
            - baseline_eval.energy_eV,
            "relaxed_minus_baseline_meV_atom": (
                final_eval.energy_eV - baseline_eval.energy_eV
            )
            * 1000.0
            / atom_count,
        },
        "displacement_from_initial": {
            "rms_A": float(np.sqrt(np.mean(displacements**2))),
            "mean_A": float(np.mean(displacements)),
            "max_A": float(np.max(displacements)),
        },
        "cell_A": np.asarray(result.atoms.cell).tolist(),
        "cell_exactly_unchanged": True,
    }

    write(OUT / "relaxed_structure.extxyz", result.atoms, format="extxyz")
    write(OUT / "relaxed_structure.cif", result.atoms, format="cif")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    with (OUT / "energy_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "structure",
                "energy_eV",
                "energy_eV_atom",
                "force_rms_eV_A",
                "force_max_eV_A",
            ),
        )
        writer.writeheader()
        for name, values in structures.items():
            writer.writerow({"structure": name, **values})

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
