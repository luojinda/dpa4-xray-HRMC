#!/usr/bin/env python
"""Microbenchmark incremental proposals against Phase 1 full recomputation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dpa4_xray_hrmc.rmc.config import load_phase2_config
from dpa4_xray_hrmc.rmc.incremental_pdf import IncrementalXrayPDF
from dpa4_xray_hrmc.rmc.runner import inclusive_grid
from dpa4_xray_hrmc.xray_pdf.experimental import read_pdfgetx_gr
from dpa4_xray_hrmc.xray_pdf.structure import load_structure


def distribution(values: list[float]) -> dict[str, float]:
    data = np.asarray(values)
    return {
        "mean_ms": float(1000.0 * np.mean(data)),
        "median_ms": float(1000.0 * np.median(data)),
        "p95_ms": float(1000.0 * np.quantile(data, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pzs_phase2_common.yaml")
    parser.add_argument("--output", default="results/PZS_phase2/benchmarks/incremental_benchmark.json")
    parser.add_argument("--incremental-moves", type=int, default=1000)
    parser.add_argument("--full-moves", type=int, default=100)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    root = config_path.parent.parent
    config = load_phase2_config(config_path)
    info = load_structure(root / config.structure)
    experiment = read_pdfgetx_gr(root / config.experimental_gr)
    fit = (experiment.r_A >= config.fit_rmin_A) & (experiment.r_A <= config.fit_rmax_A)
    q = inclusive_grid(config.qmin_Ainv, config.qmax_Ainv, config.qstep_Ainv)
    init_started = time.perf_counter()
    engine = IncrementalXrayPDF(
        info.atoms, q, experiment.r_A[fit], config.rdf_rcut_A, config.rdf_dr_A, config.biso_A2, config.lorch_mod
    )
    initialize_seconds = time.perf_counter() - init_started
    rng = np.random.default_rng(1729)
    indices = rng.integers(len(info.atoms), size=args.incremental_moves)
    displacements = rng.uniform(-config.move_max_A, config.move_max_A, (args.incremental_moves, 3))
    incremental_times: list[float] = []
    proposals = []
    for atom_index, displacement in zip(indices, displacements, strict=True):
        tick = time.perf_counter()
        proposal = engine.propose_move(int(atom_index), engine.positions_A[int(atom_index)] + displacement)
        incremental_times.append(time.perf_counter() - tick)
        proposals.append(proposal)
    full_times: list[float] = []
    max_errors: list[float] = []
    for proposal in proposals[: args.full_moves]:
        trial = engine.atoms.copy()
        trial.positions[proposal.atom_index] = proposal.new_position_A
        tick = time.perf_counter()
        _, full = engine.full_recompute(trial)
        full_times.append(time.perf_counter() - tick)
        max_errors.append(float(np.max(np.abs(engine.g_r_A2 + proposal.delta_g_A2 - full.g_r_A2))))
    result = {
        "initialize_seconds": initialize_seconds,
        "kernel_memory_bytes": engine.kernel.memory_bytes,
        "minimum_image_safe": engine.minimum_image_safe,
        "incremental_moves": args.incremental_moves,
        "incremental": distribution(incremental_times),
        "full_moves": min(args.full_moves, len(proposals)),
        "full": distribution(full_times),
        "speedup_pdf_only_median": float(np.median(full_times) / np.median(incremental_times)),
        "max_abs_incremental_vs_full_A-2": max(max_errors, default=0.0),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
