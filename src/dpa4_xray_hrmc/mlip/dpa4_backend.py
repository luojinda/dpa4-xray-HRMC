"""One-time-loaded DeePMD DPA4 backend with explicit, uncached evaluations."""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.calculator import all_changes


@dataclass(frozen=True)
class EnergyForces:
    energy_eV: float
    forces_eV_A: np.ndarray
    virial_eV: np.ndarray
    wall_s: float


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DPA4Backend:
    """Load one .pt2 package once and evaluate fixed-cell periodic structures."""

    def __init__(self, model_path: str | Path, required_elements=()) -> None:
        from deepmd.calculator import DP

        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.model_sha256 = sha256_file(self.model_path)
        started = time.perf_counter()
        self.calculator = DP(model=self.model_path)
        self.load_wall_s = time.perf_counter() - started
        self.type_map = tuple(self.calculator.dp.get_type_map())
        self.type_dict = dict(self.calculator.type_dict)
        missing = sorted(set(required_elements) - set(self.type_map))
        if missing:
            raise ValueError(f"Model type_map lacks elements: {missing}")
        self.rcut_A = float(self.calculator.dp.get_rcut())
        self.metadata = self._read_package_metadata()
        self.evaluation_count = 0

    def _read_package_metadata(self) -> dict:
        try:
            with zipfile.ZipFile(self.model_path) as archive:
                return json.loads(archive.read("model/extra/metadata.json"))
        except (KeyError, ValueError, zipfile.BadZipFile):
            return {}

    @staticmethod
    def _synchronize_cuda() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass

    def validate_atoms(self, atoms: Atoms) -> None:
        if not np.all(atoms.pbc):
            raise ValueError("DPA4 PZS calculations require three-dimensional PBC")
        if abs(float(np.linalg.det(np.asarray(atoms.cell)))) < 1e-12:
            raise ValueError("Cell is singular")
        missing = sorted(set(atoms.get_chemical_symbols()) - set(self.type_map))
        if missing:
            raise ValueError(f"Atoms contain elements absent from model: {missing}")
        if not np.all(np.isfinite(atoms.positions)):
            raise ValueError("Non-finite coordinates")

    def evaluate(self, atoms: Atoms) -> EnergyForces:
        """Force an actual model call; do not use ASE's unchanged-Atoms cache."""
        self.validate_atoms(atoms)
        self._synchronize_cuda()
        started = time.perf_counter()
        self.calculator.calculate(atoms=atoms, system_changes=all_changes)
        self._synchronize_cuda()
        wall_s = time.perf_counter() - started
        self.evaluation_count += 1
        energy = float(self.calculator.results["energy"])
        forces = np.asarray(self.calculator.results["forces"], dtype=float).copy()
        virial = np.asarray(self.calculator.results["virial"], dtype=float).copy()
        if not np.isfinite(energy) or not np.all(np.isfinite(forces)) or not np.all(np.isfinite(virial)):
            raise FloatingPointError("DPA4 returned NaN/Inf")
        return EnergyForces(energy, forces, virial, wall_s)

    def get_energy(self, atoms: Atoms) -> float:
        return self.evaluate(atoms).energy_eV

    def get_energy_forces(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        result = self.evaluate(atoms)
        return result.energy_eV, result.forces_eV_A

    def attach(self, atoms: Atoms) -> None:
        atoms.calc = self.calculator

    def manifest(self) -> dict:
        return {
            "model_path": str(self.model_path),
            "model_size_bytes": self.model_path.stat().st_size,
            "model_sha256": self.model_sha256,
            "load_wall_s": self.load_wall_s,
            "type_map": list(self.type_map),
            "type_indices": {symbol: self.type_dict[symbol] for symbol in ("Cl", "Li", "O", "P", "Si", "Zr")},
            "rcut_A": self.rcut_A,
            "package_metadata": self.metadata,
        }
