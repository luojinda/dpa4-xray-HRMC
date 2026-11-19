"""Checkpointable Pure-RMC runner with incremental X-ray PDF scoring."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase import Atoms
from ase.io import write

from dpa4_xray_hrmc.xray_pdf.experimental import read_pdfgetx_gr
from dpa4_xray_hrmc.xray_pdf.form_factors import q_dependent_weights
from dpa4_xray_hrmc.xray_pdf.partial_rdf import compute_partial_rdfs
from dpa4_xray_hrmc.xray_pdf.structure import load_structure
from dpa4_xray_hrmc.xray_pdf.transform import forward_transform

from .config import Phase2Config
from .hard_sphere import HardSphereModel
from .incremental_pdf import IncrementalXrayPDF
from .objectives import ScaleMode, ScaleObjective, Score


def inclusive_grid(start: float, stop: float, step: float) -> np.ndarray:
    count = int(round((stop - start) / step))
    if not np.isclose(start + count * step, stop, atol=1e-10):
        raise ValueError(f"Grid [{start}, {stop}] is not divisible by {step}")
    return np.linspace(start, stop, count + 1)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class TimingTotals:
    initialize_s: float = 0.0
    hard_sphere_s: float = 0.0
    delta_pdf_s: float = 0.0
    score_s: float = 0.0
    commit_s: float = 0.0
    validation_s: float = 0.0
    checkpoint_s: float = 0.0
    output_s: float = 0.0


@dataclass
class Counters:
    attempted: int = 0
    accepted: int = 0
    hard_sphere_rejected: int = 0
    metropolis_rejected: int = 0
    legal_pdf_evaluations: int = 0


@dataclass(frozen=True)
class ChainResult:
    output_directory: str
    mode: str
    seed: int
    attempted: int
    accepted: int
    hard_sphere_rejected: int
    final_loss: float
    final_scale: float
    best_loss: float
    best_scale: float
    best_step: int
    wall_seconds: float


def _write_structure_pair(directory: Path, stem: str, atoms: Atoms) -> None:
    write(directory / f"{stem}.extxyz", atoms, format="extxyz")
    write(directory / f"{stem}.cif", atoms, format="cif")


def _curve_table(
    path: Path,
    atoms: Atoms,
    config: Phase2Config,
    q_Ainv: np.ndarray,
    experiment_r_A: np.ndarray,
    experiment_g_A2: np.ndarray,
    objective: ScaleObjective,
    mode: ScaleMode,
) -> dict[str, float]:
    rdf = compute_partial_rdfs(atoms, config.rdf_rcut_A, config.rdf_dr_A)
    weights = q_dependent_weights(rdf.element_counts, q_Ainv)
    result = forward_transform(rdf, weights, experiment_r_A, config.biso_A2, config.lorch_mod)
    fit = (experiment_r_A >= config.fit_rmin_A) & (experiment_r_A <= config.fit_rmax_A)
    score = objective.score(mode, result.g_r_A2[fit])
    fixed = objective.score(ScaleMode.FIXED_INITIAL, result.g_r_A2[fit])
    positive = objective.common_scores(score.A, score.B)["scale_unbounded_positive"]
    data = np.column_stack(
        (
            experiment_r_A,
            result.g_r_A2,
            score.scale * result.g_r_A2,
            objective.s0 * result.g_r_A2,
            positive * result.g_r_A2,
            experiment_g_A2,
            score.scale * result.g_r_A2 - experiment_g_A2,
        )
    )
    np.savetxt(
        path,
        data,
        fmt="%.10g",
        header=(
            "r_A Gcalc_unscaled_A-2 Gcalc_mode_scaled_A-2 Gcalc_s0_scaled_A-2 "
            "Gcalc_positive_opt_A-2 Gexp_A-2 residual_mode_A-2"
        ),
    )
    metrics = objective.common_scores(score.A, score.B)
    metrics.update({"mode_scale": score.scale, "mode_loss": score.loss, "fixed_scale": fixed.scale})
    return metrics


def _save_checkpoint(
    path: Path,
    engine: IncrementalXrayPDF,
    counters: Counters,
    current: Score,
    best_positions_A: np.ndarray,
    best_g_A2: np.ndarray,
    best_loss: float,
    best_scale: float,
    best_step: int,
    rngs: tuple[np.random.Generator, np.random.Generator, np.random.Generator],
) -> None:
    metadata = {
        "counters": asdict(counters),
        "current": asdict(current),
        "best_loss": best_loss,
        "best_scale": best_scale,
        "best_step": best_step,
        "rng_states": [_json_compatible(rng.bit_generator.state) for rng in rngs],
    }
    np.savez_compressed(
        path,
        positions_A=engine.positions_A,
        counts=engine.counts,
        g_fit_A2=engine.g_r_A2,
        best_positions_A=best_positions_A,
        best_g_fit_A2=best_g_A2,
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def _restore_checkpoint(
    path: Path,
    engine: IncrementalXrayPDF,
    rngs: tuple[np.random.Generator, np.random.Generator, np.random.Generator],
) -> tuple[Counters, Score, np.ndarray, np.ndarray, float, float, int]:
    saved = np.load(path, allow_pickle=False)
    engine.atoms.positions[:] = saved["positions_A"]
    engine.counts[:] = saved["counts"]
    engine.g_r_A2[:] = saved["g_fit_A2"]
    metadata = json.loads(str(saved["metadata_json"]))
    for rng, state in zip(rngs, metadata["rng_states"], strict=True):
        rng.bit_generator.state = state
    counters = Counters(**metadata["counters"])
    current = Score(**metadata["current"])
    return (
        counters,
        current,
        saved["best_positions_A"].copy(),
        saved["best_g_fit_A2"].copy(),
        float(metadata["best_loss"]),
        float(metadata["best_scale"]),
        int(metadata["best_step"]),
    )


def run_chain(
    config: Phase2Config,
    mode: ScaleMode | str,
    seed: int,
    target_attempts: int,
    project_root: str | Path = ".",
    output_root_override: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
) -> ChainResult:
    config.validate()
    selected_mode = ScaleMode(mode)
    root = Path(project_root).resolve()
    structure_path = (root / config.structure).resolve()
    experiment_path = (root / config.experimental_gr).resolve()
    output_root = Path(output_root_override or (root / config.output_root)).resolve()
    output_dir = output_root / selected_mode.value / f"seed_{seed}"
    checkpoints_dir = output_dir / "checkpoints"
    snapshots_dir = output_dir / "snapshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    snapshots_dir.mkdir(exist_ok=True)

    started = time.perf_counter()
    timings = TimingTotals()
    init_started = time.perf_counter()
    info = load_structure(structure_path)
    experiment = read_pdfgetx_gr(experiment_path)
    fit = (experiment.r_A >= config.fit_rmin_A) & (experiment.r_A <= config.fit_rmax_A)
    fit_r = experiment.r_A[fit]
    fit_target = experiment.g_r_A2[fit]
    q = inclusive_grid(config.qmin_Ainv, config.qmax_Ainv, config.qstep_Ainv)
    engine = IncrementalXrayPDF(
        info.atoms,
        q,
        fit_r,
        config.rdf_rcut_A,
        config.rdf_dr_A,
        config.biso_A2,
        config.lorch_mod,
    )
    objective = ScaleObjective.from_initial(engine.g_r_A2, fit_target)
    hard_sphere = HardSphereModel.from_atoms(info.atoms, config.hard_sphere_scale)
    current = objective.score(selected_mode, engine.g_r_A2)
    timings.initialize_s = time.perf_counter() - init_started

    seed_sequences = np.random.SeedSequence(seed).spawn(3)
    rng_atom, rng_displacement, rng_acceptance = tuple(
        np.random.default_rng(sequence) for sequence in seed_sequences
    )
    rngs = (rng_atom, rng_displacement, rng_acceptance)
    counters = Counters()
    best_positions = engine.positions_A.copy()
    best_g = engine.g_r_A2.copy()
    best_loss = current.loss
    best_scale = current.scale
    best_step = 0
    if resume_checkpoint is not None:
        (
            counters,
            current,
            best_positions,
            best_g,
            best_loss,
            best_scale,
            best_step,
        ) = _restore_checkpoint(Path(resume_checkpoint), engine, rngs)
        drift = engine.validation_errors()
        if drift["max_abs_G_A-2"] > 1e-8 or drift["max_abs_hist_count"] != 0:
            raise RuntimeError(f"Checkpoint failed full validation: {drift}")

    resolved = {
        **config.to_dict(),
        "mode": selected_mode.value,
        "seed": seed,
        "target_attempts": target_attempts,
        "s0": objective.s0,
        "scale_bounds": [objective.lower_scale, objective.upper_scale],
        "minimum_image_safe": engine.minimum_image_safe,
        "kernel_memory_bytes": engine.kernel.memory_bytes,
    }
    (output_dir / "run_config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    manifest = {
        "structure": {"path": str(structure_path), "sha256": file_sha256(structure_path)},
        "experiment": {"path": str(experiment_path), "sha256": file_sha256(experiment_path)},
        "git_commit": _git_commit(root),
        "python": sys.version,
        "platform": platform.platform(),
    }
    (output_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    environment_lines = [
        f"python={platform.python_version()}",
        *[
            f"{package}={version(package)}"
            for package in ("numpy", "scipy", "ase", "matplotlib", "xraydb", "PyYAML", "psutil")
        ],
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}",
        f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}",
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}",
    ]
    (output_dir / "environment.txt").write_text("\n".join(environment_lines) + "\n", encoding="utf-8")
    (output_dir / "hard_sphere.json").write_text(
        json.dumps(hard_sphere.serializable(), indent=2), encoding="utf-8"
    )
    _write_structure_pair(output_dir, "initial_structure", info.atoms)

    trace_path = output_dir / "trace.csv"
    trace_fields = [
        "attempted_step",
        "accepted_total",
        "hs_rejected_total",
        "metropolis_rejected_total",
        "acceptance_rate",
        "s_current",
        "loss_mode",
        "loss_no_scale",
        "loss_fixed_initial",
        "loss_bounded_dynamic",
        "Rw_no_scale",
        "Rw_fixed_initial",
        "Rw_bounded_dynamic",
        "Rw_shape_only",
        "wall_s",
    ]
    trace_exists = trace_path.exists() and counters.attempted > 0
    trace_stream = trace_path.open("a" if trace_exists else "w", newline="", encoding="utf-8")
    trace_writer = csv.DictWriter(trace_stream, fieldnames=trace_fields)
    if not trace_exists:
        trace_writer.writeheader()
    drift_records: list[dict[str, float | int]] = []

    loop_started = time.perf_counter()
    while counters.attempted < target_attempts:
        counters.attempted += 1
        atom_index = int(rng_atom.integers(len(engine.atoms)))
        displacement = rng_displacement.uniform(-config.move_max_A, config.move_max_A, 3)
        new_position = engine.positions_A[atom_index] + displacement

        tick = time.perf_counter()
        valid, _, _ = hard_sphere.check_move(engine.atoms, atom_index, new_position)
        timings.hard_sphere_s += time.perf_counter() - tick
        if not valid:
            counters.hard_sphere_rejected += 1
        else:
            tick = time.perf_counter()
            proposal = engine.propose_move(atom_index, new_position)
            timings.delta_pdf_s += time.perf_counter() - tick
            counters.legal_pdf_evaluations += 1

            tick = time.perf_counter()
            trial = objective.trial_score(selected_mode, engine.g_r_A2, current, proposal.delta_g_A2)
            delta_loss = trial.loss - current.loss
            accept = delta_loss <= 0.0 or np.log(rng_acceptance.random()) < (
                -delta_loss / (2.0 * config.sigma_rmc_A2**2)
            )
            timings.score_s += time.perf_counter() - tick
            if accept:
                tick = time.perf_counter()
                engine.commit(proposal)
                timings.commit_s += time.perf_counter() - tick
                current = trial
                counters.accepted += 1
                if current.loss < best_loss:
                    best_loss = current.loss
                    best_scale = current.scale
                    best_step = counters.attempted
                    best_positions = engine.positions_A.copy()
                    best_g = engine.g_r_A2.copy()
            else:
                engine.reject(proposal)
                counters.metropolis_rejected += 1

        if counters.attempted % config.trace_interval == 0 or counters.attempted == target_attempts:
            common = objective.common_scores(current.A, current.B)
            trace_writer.writerow(
                {
                    "attempted_step": counters.attempted,
                    "accepted_total": counters.accepted,
                    "hs_rejected_total": counters.hard_sphere_rejected,
                    "metropolis_rejected_total": counters.metropolis_rejected,
                    "acceptance_rate": counters.accepted / counters.attempted,
                    "s_current": current.scale,
                    "loss_mode": current.loss,
                    "loss_no_scale": common["loss_no_scale"],
                    "loss_fixed_initial": common["loss_fixed_initial"],
                    "loss_bounded_dynamic": common["loss_bounded_dynamic"],
                    "Rw_no_scale": common["Rw_no_scale"],
                    "Rw_fixed_initial": common["Rw_fixed_initial"],
                    "Rw_bounded_dynamic": common["Rw_bounded_dynamic"],
                    "Rw_shape_only": common["Rw_shape_only"],
                    "wall_s": time.perf_counter() - started,
                }
            )
            trace_stream.flush()

        if counters.attempted % config.validation_interval == 0:
            tick = time.perf_counter()
            errors = engine.validation_errors()
            direct_A, direct_B = objective.coefficients(engine.g_r_A2)
            errors.update(
                {
                    "attempted_step": counters.attempted,
                    "abs_A_drift": abs(direct_A - current.A),
                    "abs_B_drift": abs(direct_B - current.B),
                }
            )
            drift_records.append(errors)
            timings.validation_s += time.perf_counter() - tick
            if errors["max_abs_G_A-2"] > 1e-8 or errors["max_abs_hist_count"] != 0:
                raise RuntimeError(f"Incremental state drift at step {counters.attempted}: {errors}")

        if counters.attempted % config.snapshot_interval == 0:
            write(snapshots_dir / f"step_{counters.attempted:09d}.extxyz", engine.atoms, format="extxyz")

        if counters.attempted % config.checkpoint_interval == 0:
            tick = time.perf_counter()
            _save_checkpoint(
                checkpoints_dir / f"checkpoint_{counters.attempted:09d}.npz",
                engine,
                counters,
                current,
                best_positions,
                best_g,
                best_loss,
                best_scale,
                best_step,
                rngs,
            )
            timings.checkpoint_s += time.perf_counter() - tick

    trace_stream.close()
    sampling_wall = time.perf_counter() - loop_started
    tick = time.perf_counter()
    final_atoms = engine.atoms.copy()
    best_atoms = info.atoms.copy()
    best_atoms.positions[:] = best_positions
    _write_structure_pair(output_dir, "final_structure", final_atoms)
    _write_structure_pair(output_dir, "best_structure", best_atoms)
    final_metrics = _curve_table(
        output_dir / "final_calc_Gr.dat",
        final_atoms,
        config,
        q,
        experiment.r_A,
        experiment.g_r_A2,
        objective,
        selected_mode,
    )
    best_metrics = _curve_table(
        output_dir / "best_calc_Gr.dat",
        best_atoms,
        config,
        q,
        experiment.r_A,
        experiment.g_r_A2,
        objective,
        selected_mode,
    )
    timings.output_s += time.perf_counter() - tick
    total_wall = time.perf_counter() - started
    timing_output = {
        **asdict(timings),
        "sampling_wall_s": sampling_wall,
        "total_wall_s": total_wall,
        "attempts_per_s_sampling": counters.attempted / sampling_wall,
        "legal_pdf_evaluations_per_s_sampling": counters.legal_pdf_evaluations / sampling_wall,
        "accepts_per_s_sampling": counters.accepted / sampling_wall,
        "counters": asdict(counters),
    }
    (output_dir / "timings.json").write_text(
        json.dumps(timing_output, indent=2), encoding="utf-8"
    )
    summary = {
        "mode": selected_mode.value,
        "seed": seed,
        "s0": objective.s0,
        "scale_bounds": [objective.lower_scale, objective.upper_scale],
        "final_step": counters.attempted,
        "final_loss": current.loss,
        "final_scale": current.scale,
        "best_loss": best_loss,
        "best_scale": best_scale,
        "best_step": best_step,
        "final_common_metrics": final_metrics,
        "best_common_metrics": best_metrics,
        "drift_checks": drift_records,
        "timings": timing_output,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ChainResult(
        str(output_dir),
        selected_mode.value,
        seed,
        counters.attempted,
        counters.accepted,
        counters.hard_sphere_rejected,
        current.loss,
        current.scale,
        best_loss,
        best_scale,
        best_step,
        total_wall,
    )
