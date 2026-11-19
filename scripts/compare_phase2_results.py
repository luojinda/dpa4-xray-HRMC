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
from dpa4_xray_hrmc.rmc.runner import inclusive_grid
from dpa4_xray_hrmc.xray_pdf.experimental import read_pdfgetx_gr
from dpa4_xray_hrmc.xray_pdf.form_factors import q_dependent_weights
from dpa4_xray_hrmc.xray_pdf.partial_rdf import compute_partial_rdfs
from dpa4_xray_hrmc.xray_pdf.structure import load_structure
from dpa4_xray_hrmc.xray_pdf.transform import forward_transform


MODES = tuple(mode.value for mode in ScaleMode)
DIAGNOSTIC_PAIRS = (("P", "O"), ("Si", "O"), ("Zr", "O"), ("Zr", "Cl"), ("Li", "O"), ("Li", "Cl"))


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))  # type: ignore[return-value]


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


def _ensure_partial_rdfs(chain_dir: Path, initial_atoms) -> None:
    """Persist initial/final/best RDF curves for every completed chain."""

    path = chain_dir / "diagnostic_partial_rdfs.npz"
    if path.exists():
        return
    states = {
        "initial": initial_atoms,
        "final": read(chain_dir / "final_structure.extxyz"),
        "best": read(chain_dir / "best_structure.extxyz"),
    }
    output: dict[str, np.ndarray] = {}
    for state_name, atoms in states.items():
        rdf = compute_partial_rdfs(atoms, rcut_A=6.0, dr_A=0.02)
        output.setdefault("r_A", rdf.r_A)
        for central, neighbour in DIAGNOSTIC_PAIRS:
            output[f"{state_name}_{central}_{neighbour}"] = rdf.g[_pair_key(central, neighbour)]
    np.savez_compressed(path, **output)


def _curve_metrics(path: Path) -> dict[str, float]:
    data = np.loadtxt(path)
    r = data[:, 0]
    scaled = data[:, 2]
    observed = data[:, 5]
    output: dict[str, float] = {}
    for lo, hi, name in (
        (1.5, 4.5, "short"),
        (4.5, 10.0, "medium"),
        (1.5, 10.0, "total"),
    ):
        mask = (r >= lo) & (r <= hi)
        residual = scaled[mask] - observed[mask]
        output[f"rms_{name}_A-2"] = float(np.sqrt(np.mean(residual**2)))
        denominator = float(np.sum((observed[mask] - np.mean(observed[mask])) ** 2))
        output[f"R2_{name}"] = float(1.0 - np.sum(residual**2) / denominator)
    first = (r >= 2.2) & (r <= 2.7)
    second = (r >= 3.1) & (r <= 3.8)
    output["first_peak_r_A"] = float(r[first][np.argmax(scaled[first])])
    output["first_peak_G_A-2"] = float(np.max(scaled[first]))
    output["second_peak_r_A"] = float(r[second][np.argmax(scaled[second])])
    output["second_peak_G_A-2"] = float(np.max(scaled[second]))
    output["second_over_first_peak"] = (
        output["second_peak_G_A-2"] / output["first_peak_G_A-2"]
    )
    return output


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
    experiment = read_pdfgetx_gr(project_root / config.experimental_gr)
    fit = (experiment.r_A >= config.fit_rmin_A) & (experiment.r_A <= config.fit_rmax_A)
    fit_r = experiment.r_A[fit]
    fit_target = experiment.g_r_A2[fit]
    q = inclusive_grid(config.qmin_Ainv, config.qmax_Ainv, config.qstep_Ainv)
    initial_rdf = compute_partial_rdfs(initial_atoms, config.rdf_rcut_A, config.rdf_dr_A)
    initial_weights = q_dependent_weights(initial_rdf.element_counts, q)
    initial_curve = forward_transform(
        initial_rdf, initial_weights, fit_r, config.biso_A2, config.lorch_mod
    ).g_r_A2
    s0 = float(np.dot(initial_curve, fit_target) / np.dot(initial_curve, initial_curve))
    hard = HardSphereModel.from_atoms(initial_atoms, config.hard_sphere_scale)
    cutoffs = coordination_cutoffs_from_initial(initial_atoms)

    rows: list[dict[str, object]] = []
    scale_rows: list[dict[str, object]] = []
    traces: dict[tuple[str, int], np.ndarray] = {}
    diagnostics_rows: list[dict[str, object]] = []
    initial_diagnostics = structure_diagnostics(initial_atoms, initial_atoms, hard, cutoffs)
    initial_row: dict[str, object] = {
        "mode": "initial_MQ",
        "seed": 0,
        "state": "initial",
        "hard_sphere_violation_total": initial_diagnostics["hard_sphere_violation_total"],
        "displacement_rms_A": 0.0,
        "displacement_max_A": 0.0,
    }
    for pair_name, values in initial_diagnostics["coordination"].items():
        initial_row[f"{pair_name}_mean_CN"] = values["mean_CN"]
        if "fourfold_fraction" in values:
            initial_row[f"{pair_name}_fourfold_fraction"] = values["fourfold_fraction"]
    diagnostics_rows.append(initial_row)
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
            _flatten("final_curve_", _curve_metrics(chain_dir / "final_calc_Gr.dat"), row)
            _flatten("best_curve_", _curve_metrics(chain_dir / "best_calc_Gr.dat"), row)
            counters = summary["timings"]["counters"]
            timing_values = summary["timings"]
            row.update(
                {
                    "accepted": counters["accepted"],
                    "hard_sphere_rejected": counters["hard_sphere_rejected"],
                    "metropolis_rejected": counters["metropolis_rejected"],
                    "acceptance_rate": counters["accepted"] / counters["attempted"],
                    "attempts_per_s": summary["timings"]["attempts_per_s_sampling"],
                    "total_wall_s": summary["timings"]["total_wall_s"],
                    "sampling_wall_s": timing_values["sampling_wall_s"],
                    "legal_pdf_evaluations_per_s": timing_values["legal_pdf_evaluations_per_s_sampling"],
                    "accepts_per_s": timing_values["accepts_per_s_sampling"],
                }
            )
            for timing_name in (
                "initialize_s",
                "hard_sphere_s",
                "delta_pdf_s",
                "score_s",
                "commit_s",
                "validation_s",
                "checkpoint_s",
                "output_s",
            ):
                row[timing_name] = timing_values[timing_name]
            rows.append(row)
            curve_candidates[mode].append(
                (float(summary["best_common_metrics"]["Rw_shape_only"]), chain_dir, seed)
            )
            trace = np.genfromtxt(chain_dir / "trace.csv", delimiter=",", names=True)
            traces[(mode, seed)] = np.atleast_1d(trace)
            trace_scales = np.atleast_1d(trace)["s_current"]
            lower, upper = (float(value) for value in summary["scale_bounds"])
            scale_rows.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "s0": summary["s0"],
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "scale_min": float(np.min(trace_scales)),
                    "scale_mean": float(np.mean(trace_scales)),
                    "scale_max": float(np.max(trace_scales)),
                    "scale_final": summary["final_scale"],
                    "scale_best": summary["best_scale"],
                    "fraction_at_lower_bound": float(np.mean(np.isclose(trace_scales, lower, rtol=0.0, atol=1e-12))),
                    "fraction_at_upper_bound": float(np.mean(np.isclose(trace_scales, upper, rtol=0.0, atol=1e-12))),
                }
            )
            diagnostics = _ensure_diagnostics(chain_dir, initial_atoms, hard, cutoffs)
            _ensure_partial_rdfs(chain_dir, initial_atoms)
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
    with (output / "scale_evolution.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scale_rows[0]))
        writer.writeheader()
        writer.writerows(scale_rows)

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
        "final_scale",
        "best_scale",
        "final_curve_rms_short_A-2",
        "final_curve_rms_medium_A-2",
        "final_curve_R2_short",
        "final_curve_R2_medium",
        "final_curve_first_peak_r_A",
        "final_curve_second_peak_r_A",
        "final_curve_second_over_first_peak",
        "best_curve_rms_short_A-2",
        "best_curve_rms_medium_A-2",
        "best_curve_R2_short",
        "best_curve_R2_medium",
        "best_curve_first_peak_r_A",
        "best_curve_second_peak_r_A",
        "best_curve_second_over_first_peak",
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

    selected_curves: dict[str, tuple[np.ndarray, int, Path]] = {}
    for mode, candidates in curve_candidates.items():
        _, directory, seed = min(candidates, key=lambda item: item[0])
        selected_curves[mode] = (np.loadtxt(directory / "best_calc_Gr.dat"), seed, directory)
    for lo, hi, name in ((1.5, 4.5, "comparison_1p5_4p5A.png"), (1.5, 10.0, "comparison_1p5_10A.png")):
        fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
        first = next(iter(selected_curves.values()))[0]
        mask = (first[:, 0] >= lo) & (first[:, 0] <= hi)
        axis.plot(first[mask, 0], first[mask, 5], color="black", linewidth=1.3, label="Experiment ZXS_1")
        initial_mask = (fit_r >= lo) & (fit_r <= hi)
        axis.plot(fit_r[initial_mask], s0 * initial_curve[initial_mask], color="0.45", linestyle="--", linewidth=1.0, label="Initial MQ, s0-scaled")
        for mode, (curve, seed, _) in selected_curves.items():
            axis.plot(curve[mask, 0], curve[mask, 2], color=colors[mode], linewidth=1.0, label=f"{mode}, best seed {seed}")
        axis.axhline(0.0, color="0.5", linewidth=0.6)
        axis.set_xlim(lo, hi)
        axis.set_xlabel(r"$r$ ($\AA$)")
        axis.set_ylabel(r"$G(r)$ ($\AA^{-2}$)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
        fig.savefig(output / name, dpi=300)
        plt.close(fig)

    for column, ylabel, name in (
        (1, r"Unscaled $G_{calc}(r)$ ($\AA^{-2}$)", "comparison_unscaled_1p5_10A.png"),
        (3, r"$s_0 G_{calc}(r)$ ($\AA^{-2}$)", "comparison_fixed_s0_1p5_10A.png"),
    ):
        fig, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
        first = next(iter(selected_curves.values()))[0]
        mask = (first[:, 0] >= 1.5) & (first[:, 0] <= 10.0)
        axis.plot(first[mask, 0], first[mask, 5], color="black", linewidth=1.3, label="Experiment ZXS_1")
        baseline = initial_curve if column == 1 else s0 * initial_curve
        axis.plot(fit_r, baseline, color="0.45", linestyle="--", linewidth=1.0, label="Initial MQ")
        for mode, (curve, seed, _) in selected_curves.items():
            axis.plot(curve[mask, 0], curve[mask, column], color=colors[mode], linewidth=1.0, label=f"{mode}, best seed {seed}")
        axis.axhline(0.0, color="0.5", linewidth=0.6)
        axis.set_xlim(1.5, 10.0)
        axis.set_xlabel(r"$r$ ($\AA$)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
        fig.savefig(output / name, dpi=300)
        plt.close(fig)

    initial_diag_rdf = compute_partial_rdfs(initial_atoms, rcut_A=6.0, dr_A=0.02)
    fig, axes = plt.subplots(3, 2, figsize=(10.0, 10.0), constrained_layout=True, sharex=True)
    for axis, (central, neighbour) in zip(axes.flat, DIAGNOSTIC_PAIRS, strict=True):
        pair = _pair_key(central, neighbour)
        axis.plot(initial_diag_rdf.r_A, initial_diag_rdf.g[pair], color="black", linestyle="--", linewidth=1.0, label="Initial MQ")
        for mode, (_, seed, directory) in selected_curves.items():
            atoms = read(directory / "best_structure.extxyz")
            rdf = compute_partial_rdfs(atoms, rcut_A=6.0, dr_A=0.02)
            axis.plot(rdf.r_A, rdf.g[pair], color=colors[mode], linewidth=0.9, label=f"{mode}, {seed}")
        axis.set_title(f"{central}-{neighbour}")
        axis.set_xlim(0.8, 6.0)
        axis.set_ylabel(r"$g_{\alpha\beta}(r)$")
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel(r"$r$ ($\AA$)")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.savefig(output / "partial_rdf_best_comparison.png", dpi=300)
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
