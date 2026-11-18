"""Configuration loading and validation for the Phase 1 calculator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PDFConfig:
    sample: str
    structure: str
    radiation_type: str
    qmin_Ainv: float
    qmax_Ainv: float
    qstep_Ainv: float
    output_rmin_A: float
    output_rmax_A: float
    output_rstep_A: float
    biso_A2: float
    lorch_mod: bool
    compute_device: str
    rdf_dr_A: float
    rdf_rcut_A: float
    convergence_rcuts_A: tuple[float, ...]
    convergence_rdf_dr_A: float
    plot_ranges_A: tuple[tuple[float, float], ...]
    output_directory: str

    def validate(self) -> None:
        if self.radiation_type.lower() != "xray":
            raise ValueError("Phase 1 implements only radiation_type='xray'")
        if self.compute_device.lower() != "cpu":
            raise ValueError("Phase 1 implements only compute_device='cpu'")
        if not (0.0 <= self.qmin_Ainv < self.qmax_Ainv):
            raise ValueError("Require 0 <= qmin_Ainv < qmax_Ainv")
        for name in ("qstep_Ainv", "output_rstep_A", "rdf_dr_A"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.output_rmin_A < 0 or self.output_rmax_A <= self.output_rmin_A:
            raise ValueError("Invalid output r range")
        if self.biso_A2 < 0:
            raise ValueError("biso_A2 cannot be negative")
        if self.rdf_rcut_A <= 0:
            raise ValueError("rdf_rcut_A must be positive")
        if not self.convergence_rcuts_A:
            raise ValueError("At least one convergence cutoff is required")
        if max(self.convergence_rcuts_A) > self.rdf_rcut_A + 1e-12:
            raise ValueError("Convergence cutoff exceeds rdf_rcut_A")
        if self.convergence_rdf_dr_A < self.rdf_dr_A:
            raise ValueError("Convergence RDF bin width must be >= primary width")
        ratio = self.convergence_rdf_dr_A / self.rdf_dr_A
        if abs(ratio - round(ratio)) > 1e-10:
            raise ValueError("Convergence RDF bin width must be an integer multiple")
        for lo, hi in self.plot_ranges_A:
            if lo < self.output_rmin_A or hi > self.output_rmax_A or hi <= lo:
                raise ValueError(f"Invalid plot range {(lo, hi)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> PDFConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["convergence_rcuts_A"] = tuple(float(x) for x in raw["convergence_rcuts_A"])
    raw["plot_ranges_A"] = tuple(tuple(float(x) for x in pair) for pair in raw["plot_ranges_A"])
    config = PDFConfig(**raw)
    config.validate()
    return config
