"""Configuration model for the Phase 2 Pure-RMC benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Phase2Config:
    sample: str
    structure: str
    experimental_gr: str
    qmin_Ainv: float
    qmax_Ainv: float
    qstep_Ainv: float
    fit_rmin_A: float
    fit_rmax_A: float
    biso_A2: float
    lorch_mod: bool
    rdf_dr_A: float
    rdf_rcut_A: float
    hard_sphere_scale: float
    move_max_A: float
    sigma_rmc_A2: float
    trace_interval: int
    checkpoint_interval: int
    validation_interval: int
    snapshot_interval: int
    seeds: tuple[int, ...]
    output_root: str

    def validate(self) -> None:
        if not 0.0 <= self.qmin_Ainv < self.qmax_Ainv:
            raise ValueError("Invalid Q range")
        if not 0.0 <= self.fit_rmin_A < self.fit_rmax_A:
            raise ValueError("Invalid fit range")
        for name in ("qstep_Ainv", "rdf_dr_A", "rdf_rcut_A", "move_max_A", "sigma_rmc_A2"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.hard_sphere_scale <= 1.0:
            raise ValueError("hard_sphere_scale must be in (0, 1]")
        for name in ("trace_interval", "checkpoint_interval", "validation_interval", "snapshot_interval"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be non-empty and unique")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_phase2_config(path: str | Path) -> Phase2Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw["seeds"] = tuple(int(seed) for seed in raw["seeds"])
    config = Phase2Config(**raw)
    config.validate()
    return config
