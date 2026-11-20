"""Fixed-cell ASE FIRE wrapper for validated DPA4 calculators."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.optimize import FIRE

from .dpa4_backend import DPA4Backend


@dataclass(frozen=True)
class FireResult:
    atoms: Atoms
    requested_steps: int
    executed_steps: int
    converged: bool
    initial_energy_eV: float
    final_energy_eV: float
    initial_force_rms_eV_A: float
    final_force_rms_eV_A: float
    initial_force_max_eV_A: float
    final_force_max_eV_A: float
    wall_s: float


def _force_metrics(forces: np.ndarray) -> tuple[float, float]:
    norms = np.linalg.norm(forces, axis=1)
    return float(np.sqrt(np.mean(norms**2))), float(np.max(norms))


def run_fixed_cell_fire(
    atoms: Atoms,
    backend: DPA4Backend,
    max_steps: int,
    *,
    fmax_eV_A: float = 0.05,
    maxstep_A: float = 0.05,
    dt_fs_like: float = 0.05,
    logfile: str | None = None,
) -> FireResult:
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    work = atoms.copy()
    cell = np.asarray(work.cell).copy()
    symbols = work.get_chemical_symbols()
    initial = backend.evaluate(work)
    initial_rms, initial_max = _force_metrics(initial.forces_eV_A)
    backend.attach(work)
    optimizer = FIRE(work, logfile=logfile, maxstep=maxstep_A, dt=dt_fs_like)
    started = time.perf_counter()
    converged = bool(optimizer.run(fmax=fmax_eV_A, steps=max_steps))
    wall_s = time.perf_counter() - started
    if symbols != work.get_chemical_symbols() or not np.array_equal(cell, np.asarray(work.cell)):
        raise RuntimeError("FIRE changed species order or cell")
    final = backend.evaluate(work)
    final_rms, final_max = _force_metrics(final.forces_eV_A)
    return FireResult(
        atoms=work,
        requested_steps=max_steps,
        executed_steps=int(optimizer.get_number_of_steps()),
        converged=converged,
        initial_energy_eV=initial.energy_eV,
        final_energy_eV=final.energy_eV,
        initial_force_rms_eV_A=initial_rms,
        final_force_rms_eV_A=final_rms,
        initial_force_max_eV_A=initial_max,
        final_force_max_eV_A=final_max,
        wall_s=wall_s,
    )
