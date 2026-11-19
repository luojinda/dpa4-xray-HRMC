#!/usr/bin/env python
"""Cross-score completed Phase 2 chains and generate comparison tables/plots."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dpa4_xray_hrmc_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

from dpa4_xray_hrmc.rmc.config import load_phase2_config
from dpa4_xray_hrmc.rmc.diagnostics import coordination_cutoffs_from_initial, structure_diagnostics
from dpa4_xray_hrmc.rmc.hard_sphere import HardSphereModel
from dpa4_xray_hrmc.rmc.objectives import ScaleMode
from dpa4_xray_hrmc.xray_pdf.structure import load_structure


MODES = tuple(mode.value for mode in ScaleMode)


def _flatten(prefix: str, values: dict[str, object], output: dict[str, object]) -> None:
    for key, value in values.items():
        if isinstance(value, dict):
            _flatten(f"{prefix}{key}_", value, output)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            output[f"{prefix}{key}"] = value


def _ensure_diagnostics(chain_dir: Path, initial_atoms, hard, cutoffs) -> dict[str, object]:
    path = chain_dir / "diagnostics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    final_atoms = read(chain_dir / "final_structure.extxyz")
    best_atoms = read(chain_dir / "best_structure.extxyz")
    diagnostics = {
        "coordination_cutoffs_from_initial": cutoffs,
        "final": structure_diagnostics(initial_atoms, final_atoms, hard, cutoffs),
        "best": structure_diagnostics(initial_atoms, best_atoms, hard, cutoffs),
    }
    path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Root containing mode/seed_* directories")
    parser.add_argument("--config", default="configs/pzs_phase2_common.yaml")
    parser.add_argument("--output-dir", help="Defaults to the root directory")
    args = parser.parse_args()
    chain_root = Path(args.root).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else chain_root
    output.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent
    config = load_phase2_config(config_path)
    initial_atoms = load_structure(project_root / config.structure).atoms
    hard = HardSphereModel.from_atoms(initial_atoms, config.hard_sphere_scale)
    cutoffs = coordination_cutoffs_from_initial(initial_atoms)

    rows: list[dict[str, object]] = []
    traces: dict[tuple[str, int], np.ndarray] = {}
    diagnostics_rows: list[dict[str, object]] = []
    curve_candidates: dict[str, list[tuple[float, Path, int]]] = defaultdict(list)
    for mode in MODES:
        for chain_dir in sorted((chain_root / mode).glob("seed_*")):
            summary_path = chain_dir / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            seed = int(summary["seed"])
            row: dict[str, object] = {
                "mode": mode,
                "seed": seed,
                "final_step": summary["final_step"],
                "final_loss_mode": summary["final_loss"],
                "final_scale": summary["final_scale"],
                "best_loss_mode": summary["best_loss"],
                "best_scale": summary["best_scale"],
                "best_step": summary["best_step"],
            }
            _flatten("final_", summary["final_common_metrics"], row)
            _flatten("best_", summary["best_common_metrics"], row)
            counters = summary["timings"]["counters"]
            row.update(
                {
                    "accepted": counters["accepted"],
                    "hard_sphere_rejected": counters["hard_sphere_rejected"],
                    "metropolis_rejected": counters["metropolis_rejected"],
                    "acceptance_rate": counters["accepted"] / counters["attempted"],
                    "attempts_per_s": summary["timings"]["attempts_per_s_sampling"],
                    "total_wall_s": summary["timings"]["total_wall_s"],
                }
            )
            rows.append(row)
            curve_candidates[mode].append(
                (float(summary["best_common_metrics"]["Rw_shape_only"]), chain_dir, seed)
            )
            trace = np.genfromtxt(chain_dir / "trace.csv", delimiter=",", names=True)
            traces[(mode, seed)] = np.atleast_1d(trace)
            diagnostics = _ensure_diagnostics(chain_dir, initial_atoms, hard, cutoffs)
            for state_name in ("final", "best"):
                state = diagnostics[state_name]
                diag_row: dict[str, object] = {
                    "mode": mode,
                    "seed": seed,
                    "state": state_name,
                    "hard_sphere_violation_total": state["hard_sphere_violation_total"],
                    "displacement_rms_A": state["displacement_from_initial"]["rms_A"],
                    "displacement_max_A": state["displacement_from_initial"]["max_A"],
                }
                for pair_name, values in state["coordination"].items():
                    diag_row[f"{pair_name}_mean_CN"] = values["mean_CN"]
                    if "fourfold_fraction" in values:
                        diag_row[f"{pair_name}_fourfold_fraction"] = values["fourfold_fraction"]
                diagnostics_rows.append(diag_row)

    if not rows:
        raise SystemExit(f"No completed chains under {chain_root}")
    fields = sorted({key for row in rows for key in row})
    with (output / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    diag_fields = sorted({key for row in diagnostics_rows for key in row})
    with (output / "structural_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=diag_fields)
        writer.writeheader()
        writer.writerows(diagnostics_rows)

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    aggregate_keys = (
        "final_Rw_no_scale",
        "final_Rw_fixed_initial",
        "final_Rw_bounded_dynamic",
        "final_Rw_shape_only",
        "best_Rw_no_scale",
        "best_Rw_fixed_initial",
        "best_Rw_bounded_dynamic",
        "best_Rw_shape_only",
        "acceptance_rate",
        "attempts_per_s",
    )
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        aggregate[mode] = {}
        for key in aggregate_keys:
            values = np.asarray([float(row[key]) for row in selected])
            aggregate[mode][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "count": len(values),
            }
    (output / "comparison_statistics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )

    colors = {"no_scale": "#1b9e77", "fixed_initial": "#7570b3", "bounded_dynamic": "#d95f02"}
    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for (mode, seed), trace in traces.items():
        axis.plot(trace["attempted_step"], trace["loss_mode"], color=colors[mode], alpha=0.45, linewidth=0.9, label=f"{mode}, {seed}")
    axis.set_xlabel("Attempted moves")
    axis.set_ylabel("Mode sampling loss")
    axis.set_yscale("log")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=7, ncol=3)
    fig.savefig(output / "loss_vs_attempts.png", dpi=300)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for (mode, seed), trace in traces.items():
        axis.plot(trace["attempted_step"], trace["s_current"], color=colors[mode], alpha=0.55, linewidth=0.9, label=f"{mode}, {seed}")
    axis.set_xlabel("Attempted moves")
    axis.set_ylabel("Sampling scale")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=7, ncol=3)
    fig.savefig(output / "scale_vs_attempts.png", dpi=300)
    plt.close(fig)

    selected_curves: dict[str, tuple[np.ndarray, int]] = {}
    for mode, candidates in curve_candidates.items():
        _, directory, seed = min(candidates, key=lambda item: item[0])
        selected_curves[mode] = (np.loadtxt(directory / "best_calc_Gr.dat"), seed)
    for lo, hi, name in ((1.5, 4.5, "comparison_1p5_4p5A.png"), (1.5, 10.0, "comparison_1p5_10A.png")):
        fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
        first = next(iter(selected_curves.values()))[0]
        mask = (first[:, 0] >= lo) & (first[:, 0] <= hi)
        axis.plot(first[mask, 0], first[mask, 5], color="black", linewidth=1.3, label="Experiment ZXS_1")
        for mode, (curve, seed) in selected_curves.items():
            axis.plot(curve[mask, 0], curve[mask, 2], color=colors[mode], linewidth=1.0, label=f"{mode}, best seed {seed}")
        axis.axhline(0.0, color="0.5", linewidth=0.6)
        axis.set_xlim(lo, hi)
        axis.set_xlabel(r"$r$ ($\AA$)")
        axis.set_ylabel(r"$G(r)$ ($\AA^{-2}$)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
        fig.savefig(output / name, dpi=300)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    labels = list(MODES)
    means = [aggregate[mode]["attempts_per_s"]["mean"] for mode in labels]
    errors = [aggregate[mode]["attempts_per_s"]["std"] for mode in labels]
    axis.bar(labels, means, yerr=errors, color=[colors[mode] for mode in labels], alpha=0.85)
    axis.set_ylabel("Attempts / s")
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(output / "speed_comparison.png", dpi=300)
    plt.close(fig)
    print(f"Completed chains: {len(rows)}")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
