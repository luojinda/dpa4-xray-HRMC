#!/usr/bin/env python
from __future__ import annotations

import json, os, statistics, time
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from dpa4_xray_hrmc.mlip.dpa4_backend import DPA4Backend


def main() -> None:
    root=Path(__file__).resolve().parents[1]; out=root/"results/PZS_phase3/phase3A"; out.mkdir(parents=True,exist_ok=True)
    atoms=read(root/"PDF/PZS_quench_222.cif"); backend=DPA4Backend(os.environ["DPA4_MODEL_PATH"], required_elements=set(atoms.get_chemical_symbols()))
    torch.cuda.reset_peak_memory_stats()
    warm=[backend.evaluate(atoms).wall_s for _ in range(10)]
    times=[]
    for _ in range(100): times.append(backend.evaluate(atoms).wall_s)
    data=np.asarray(times)
    report={"model_load_s":backend.load_wall_s,"warmup_calls":10,"warmup_s":warm,"measured_calls":100,"api_note":"DeePMD ASE DP.calculate always returns energy, forces and virial; energy-only and energy+forces are the same model call in this interface.","median_s":float(np.median(data)),"p95_s":float(np.quantile(data,.95)),"mean_s":float(data.mean()),"min_s":float(data.min()),"max_s":float(data.max()),"calls_per_s_median":float(1/np.median(data)),"gpu_peak_allocated_MiB":torch.cuda.max_memory_allocated()/2**20,"gpu_peak_reserved_MiB":torch.cuda.max_memory_reserved()/2**20}
    (out/"gpu_benchmark.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
