#!/usr/bin/env python
"""Run the Phase 2 cutoff study and incremental/full correctness gate."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dpa4_xray_hrmc_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dpa4_xray_hrmc.rmc.config import load_phase2_config
from dpa4_xray_hrmc.rmc.incremental_pdf import IncrementalXrayPDF
from dpa4_xray_hrmc.rmc.runner import inclusive_grid
from dpa4_xray_hrmc.xray_pdf.experimental import read_pdfgetx_gr
from dpa4_xray_hrmc.xray_pdf.form_factors import q_dependent_weights
from dpa4_xray_hrmc.xray_pdf.partial_rdf import compute_partial_rdfs, truncate_rdf
from dpa4_xray_hrmc.xray_pdf.structure import load_structure
from dpa4_xray_hrmc.xray_pdf.transform import forward_transform


def error_metrics(reference: np.ndarray, trial: np.ndarray) -> dict[str, float]:
    difference = np.asarray(trial) - np.asarray(reference)
    return {
        "max_abs_A-2": float(np.max(np.abs(difference))),
        "rms_A-2": float(np.sqrt(np.mean(difference**2))),
        "relative_rms": float(np.sqrt(np.mean(difference**2)) / np.sqrt(np.mean(reference**2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pzs_phase2_common.yaml")
    parser.add_argument("--output-dir", default="results/PZS_phase2/correctness")
    parser.add_argument("--trial-moves", type=int, default=100)
    parser.add_argument("--mixed-moves", type=int, default=10000)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parent.parent
    config = load_phase2_config(config_path)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    info = load_structure(root / config.structure)
    experiment = read_pdfgetx_gr(root / config.experimental_gr)
    fit = (experiment.r_A >= config.fit_rmin_A) & (experiment.r_A <= config.fit_rmax_A)
    r_fit = experiment.r_A[fit]
    q = inclusive_grid(config.qmin_Ainv, config.qmax_Ainv, config.qstep_Ainv)

    cutoff_started = time.perf_counter()
    reference_rdf = compute_partial_rdfs(info.atoms, 40.0, config.rdf_dr_A)
    weights = q_dependent_weights(reference_rdf.element_counts, q)
    cutoff_curves: dict[float, np.ndarray] = {}
    for cutoff in (12.0, 20.0, 30.0, 40.0):
        rdf = reference_rdf if cutoff == 40.0 else truncate_rdf(reference_rdf, cutoff)
        cutoff_curves[cutoff] = forward_transform(
            rdf, weights, r_fit, config.biso_A2, config.lorch_mod
        ).g_r_A2
    cutoff_metrics = {
        f"{cutoff:g}_vs_40_A": error_metrics(cutoff_curves[40.0], curve)
        for cutoff, curve in cutoff_curves.items()
        if cutoff != 40.0
    }
    cutoff_seconds = time.perf_counter() - cutoff_started

    init_started = time.perf_counter()
    engine = IncrementalXrayPDF(
        info.atoms,
        q,
        r_fit,
        config.rdf_rcut_A,
        config.rdf_dr_A,
        config.biso_A2,
        config.lorch_mod,
    )
    initialize_seconds = time.perf_counter() - init_started
    initial_errors = engine.validation_errors()
    rng = np.random.default_rng(20261119)
    trial_max_errors: list[float] = []
    trial_rms_errors: list[float] = []
    reject_unchanged = True
    trial_started = time.perf_counter()
    for move_index in range(args.trial_moves):
        atom_index = int(rng.integers(len(engine.atoms)))
        new_position = engine.positions_A[atom_index] + rng.uniform(
            -config.move_max_A, config.move_max_A, 3
        )
        proposal = engine.propose_move(atom_index, new_position)
        trial_atoms = engine.atoms.copy()
        trial_atoms.positions[atom_index] = proposal.new_position_A
        _, full_trial = engine.full_recompute(trial_atoms)
        difference = engine.g_r_A2 + proposal.delta_g_A2 - full_trial.g_r_A2
        trial_max_errors.append(float(np.max(np.abs(difference))))
        trial_rms_errors.append(float(np.sqrt(np.mean(difference**2))))
        if move_index % 2 == 0:
            engine.commit(proposal)
        else:
            position_before = engine.positions_A.copy()
            counts_before = engine.counts.copy()
            g_before = engine.g_r_A2.copy()
            engine.reject(proposal)
            reject_unchanged &= (
                np.array_equal(position_before, engine.positions_A)
                and np.array_equal(counts_before, engine.counts)
                and np.array_equal(g_before, engine.g_r_A2)
            )
    trial_seconds = time.perf_counter() - trial_started
    after_trial_errors = engine.validation_errors()

    mixed_started = time.perf_counter()
    mixed_accepted = 0
    for _ in range(args.mixed_moves):
        atom_index = int(rng.integers(len(engine.atoms)))
        new_position = engine.positions_A[atom_index] + rng.uniform(
            -0.5 * config.move_max_A, 0.5 * config.move_max_A, 3
        )
        proposal = engine.propose_move(atom_index, new_position)
        if rng.random() < 0.5:
            engine.commit(proposal)
            mixed_accepted += 1
        else:
            engine.reject(proposal)
    mixed_seconds = time.perf_counter() - mixed_started
    after_mixed_errors = engine.validation_errors()

    report = {
        "structure": {
            "formula": info.formula,
            "atoms": info.n_atoms,
            "cell_A": info.cell_A,
            "density_A-3": info.number_density_A3,
            "minimum_periodic_separation_A": info.minimum_periodic_separation_A,
        },
        "cutoff_study": {
            "seconds": cutoff_seconds,
            "metrics": cutoff_metrics,
        },
        "incremental": {
            "rcut_A": config.rdf_rcut_A,
            "dr_A": config.rdf_dr_A,
            "minimum_image_safe": engine.minimum_image_safe,
            "kernel_memory_bytes": engine.kernel.memory_bytes,
            "initialize_seconds": initialize_seconds,
            "initial_errors": initial_errors,
            "trial_moves": args.trial_moves,
            "trial_seconds": trial_seconds,
            "trial_max_abs_G_A-2": max(trial_max_errors, default=0.0),
            "trial_max_rms_G_A-2": max(trial_rms_errors, default=0.0),
            "reject_byte_unchanged": reject_unchanged,
            "after_trial_errors": after_trial_errors,
            "mixed_moves": args.mixed_moves,
            "mixed_accepted": mixed_accepted,
            "mixed_seconds": mixed_seconds,
            "mixed_attempts_per_s": args.mixed_moves / mixed_seconds,
            "after_mixed_errors": after_mixed_errors,
        },
    }
    (output / "incremental_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savetxt(
        output / "cutoff_curves.dat",
        np.column_stack((r_fit, *[cutoff_curves[value] for value in (12.0, 20.0, 30.0, 40.0)])),
        header="r_A G_rcut12_A-2 G_rcut20_A-2 G_rcut30_A-2 G_rcut40_A-2",
    )
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.8), sharex=True, constrained_layout=True)
    for cutoff, color in zip((12.0, 20.0, 30.0, 40.0), ("#d95f02", "#7570b3", "#1b9e77", "black"), strict=True):
        axes[0].plot(r_fit, cutoff_curves[cutoff], label=f"rcut={cutoff:g} A", linewidth=1.0, color=color)
        if cutoff != 40.0:
            axes[1].plot(r_fit, cutoff_curves[cutoff] - cutoff_curves[40.0], label=f"{cutoff:g}-40 A", linewidth=0.9, color=color)
    axes[0].set_ylabel(r"$G(r)$ ($\AA^{-2}$)")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].axhline(0.0, color="0.5", linewidth=0.6)
    axes[1].set_ylabel(r"$\Delta G$ ($\AA^{-2}$)")
    axes[1].set_xlabel(r"$r$ ($\AA$)")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)
    fig.savefig(output / "cutoff_convergence.png", dpi=300)
    plt.close(fig)

    _, final_full = engine.full_recompute()
    difference = engine.g_r_A2 - final_full.g_r_A2
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.8), sharex=True, constrained_layout=True)
    axes[0].plot(r_fit, final_full.g_r_A2, color="black", linewidth=1.2, label="Phase 1 full")
    axes[0].plot(r_fit, engine.g_r_A2, color="#d95f02", linestyle="--", linewidth=0.9, label="Incremental state")
    axes[0].set_ylabel(r"$G(r)$ ($\AA^{-2}$)")
    axes[0].legend(frameon=False)
    axes[1].plot(r_fit, difference, color="#145da0", linewidth=0.9)
    axes[1].axhline(0.0, color="0.5", linewidth=0.6)
    axes[1].set_ylabel(r"incremental - full ($\AA^{-2}$)")
    axes[1].set_xlabel(r"$r$ ($\AA$)")
    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)
    fig.savefig(output / "incremental_vs_full.png", dpi=300)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
