#!/usr/bin/env python
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
from ase.io import read

from dpa4_xray_hrmc.mlip.dpa4_backend import DPA4Backend, sha256_file


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model = Path(os.environ["DPA4_MODEL_PATH"]).resolve()
    out = root / "results/PZS_phase3/phase3A"
    out.mkdir(parents=True, exist_ok=True)
    structure_path = root / "PDF/PZS_quench_222.cif"
    experiment_path = root / "PDF/PDF_ZXS_1_100keV_350mm_10s_eiger16m_slice_0_.gr"
    atoms = read(structure_path)
    backend = DPA4Backend(model, required_elements=set(atoms.get_chemical_symbols()))
    gpu = torch.cuda.get_device_properties(0)
    manifest = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in ("torch", "deepmd-kit", "ase", "numpy", "scipy", "pymatgen", "xraydb", "pytest")},
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": {"name": gpu.name, "capability": list(torch.cuda.get_device_capability(0)), "total_memory_bytes": gpu.total_memory},
        "model": backend.manifest(),
        "structure": {
            "path": str(structure_path), "sha256": sha256_file(structure_path), "n_atoms": len(atoms),
            "composition": dict(sorted(Counter(atoms.get_chemical_symbols()).items())), "pbc": atoms.pbc.tolist(),
            "cell_A": atoms.cell.array.tolist(), "volume_A3": atoms.get_volume(),
        },
        "experiment": {"path": str(experiment_path), "sha256": sha256_file(experiment_path)},
    }
    (out / "environment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "model_audit.json").write_text(json.dumps(backend.manifest(), indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
