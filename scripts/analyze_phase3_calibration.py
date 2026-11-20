"""Uniform post-processing for Phase 3 calibration runs."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import numpy as np
from ase.io import read

from dpa4_xray_hrmc.analysis.phase3_structure import Phase3StructureEvaluator
from dpa4_xray_hrmc.mlip.dpa4_backend import DPA4Backend
from dpa4_xray_hrmc.rmc.config import load_phase2_config


def flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, (list, tuple, np.ndarray)):
        result[prefix] = json.dumps(np.asarray(value).tolist())
    elif isinstance(value, np.generic):
        result[prefix] = value.item()
    else:
        result[prefix] = value
    return result


root = Path(__file__).resolve().parents[1]
out = root / "results/PZS_phase3/calibration"
out.mkdir(parents=True, exist_ok=True)
config = load_phase2_config(root / "configs/pzs_phase2_common.yaml")
evaluator = Phase3StructureEvaluator(
    root, config, root / "configs/pzs_phase2_5_coordination_cutoffs.json"
)
backend = DPA4Backend(os.environ["DPA4_MODEL_PATH"])
mq = read(root / config.structure)
mq_energy = backend.get_energy(mq)

runs = []
for phase, pattern in (("Alternating", "phase3B/calibration_*"),
                       ("Inline", "phase3C/calibration_*"),
                       ("Alternating", "phase3B/validation_100k_*"),
                       ("Inline", "phase3C/validation_100k_*")):
    for directory in sorted((root / "results/PZS_phase3").glob(pattern)):
        summary_path = directory / "summary.json"
        structure_path = directory / "final_structure.extxyz"
        if not summary_path.exists() or not structure_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        atoms = read(structure_path)
        ef = backend.evaluate(atoms)
        force_norms = np.linalg.norm(ef.forces_eV_A, axis=1)
        diagnostics = evaluator.evaluate(atoms)
        diagnostics.pop("g_r_A2", None)
        record = {
            "phase": phase, "tag": directory.name,
            "path": str(directory.relative_to(root)),
            **summary,
            "recomputed_energy_eV": ef.energy_eV,
            "delta_energy_from_MQ_eV": ef.energy_eV - mq_energy,
            "force_rms_eV_A": float(np.sqrt(np.mean(force_norms**2))),
            "force_max_eV_A": float(np.max(force_norms)),
            "diagnostics": diagnostics,
        }
        runs.append(record)

flat = [flatten(record) for record in runs]
fields = sorted({key for row in flat for key in row})
with (out / "calibration_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader(); writer.writerows(flat)
(out / "calibration_evaluations.json").write_text(json.dumps(runs, indent=2))

common_wall_s = min(float(run["wall_s"]) for run in runs)
matched = []
for run in runs:
    directory = root / run["path"]
    trace = json.loads((directory / "trace.json").read_text())
    candidates = [row for row in trace if float(row.get("wall_s", 0.0)) <= common_wall_s]
    candidates.append({key: run[key] for key in
                       ("attempted", "accepted", "wall_s", "Rw_no_scale",
                        "energy_eV", "P_fourfold", "Si_fourfold")
                       if key in run})
    candidates = [row for row in candidates if float(row.get("wall_s", 0.0)) <= common_wall_s]
    chosen = max(candidates, key=lambda row: float(row.get("wall_s", 0.0)))
    matched.append({"phase": run["phase"], "tag": run["tag"],
                    "common_wall_s": common_wall_s, **chosen})
matched_flat = [flatten(row) for row in matched]
matched_fields = sorted({key for row in matched_flat for key in row})
with (out / "wall_matched_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=matched_fields)
    writer.writeheader(); writer.writerows(matched_flat)
print(json.dumps({"runs": len(runs), "common_wall_s": common_wall_s,
                  "output": str(out)}, indent=2))
