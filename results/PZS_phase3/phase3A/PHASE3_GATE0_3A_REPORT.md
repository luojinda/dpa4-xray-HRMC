# PZS Phase 3 Gate 0 / Phase 3A validation report

Date: 2026-11-20  
Branch: `codex/phase3-dpa4-hrmc`  
Gate decision: **PASS for Phase 3B/3C development and bounded pilot runs**

## Scope

This report records the mandatory pre-HRMC validation of the DPA4 model on the
original 896-atom PZS MQ structure.  No DPA4 term was inserted into the RMC
loop before this gate was completed.  Phase 2.5 PDF, coordination and
hard-sphere definitions remain unchanged; the Phase 3 evaluator always derives
hard-sphere cutoffs once from the original MQ structure.

## Gate 0: environment and artifacts

- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0, 32,668 MiB.
- Driver/system CUDA: 580.76.05 / 13.0.
- Runtime: Python 3.12.13, PyTorch 2.11.0+cu128, DeepMD-kit
  3.2.0b1.dev114+g0acd0e57d, ASE 3.29.0.
- Model: `/root/autodl-tmp/CODEX/DPA4_Neo_PZS_NPT_LoRA.pt2`.
- Model SHA-256: `166a5450f878e9e46238d71eca36ca87e7d07d30a4729cfbe7945c67479c2bcd`.
- Model size: 24,979,873 bytes; cutoff 6.0 A; mixed 118-element type map.
- Required type indices: Li 2, O 7, Si 13, P 14, Cl 16, Zr 39.
- Original MQ input SHA-256:
  `9d003a062e8e5f2c8579c095fcd020a093f0faf0604664383e997bc30d92736a`.
- Experimental ZXS_1 `.gr` SHA-256:
  `4a05e8cae01f1eece89e1fa99a7a6e9b76f35cfe050ac010438aa07ebc3c02183`.
- Structure: 896 atoms (`Cl512 Li96 O128 P32 Si64 Zr64`), periodic in all
  directions, cell 29.379999 x 24.742001 x 28.0 A.

The model, CIF and experimental data are intentionally not committed.  The
manifest records their paths and hashes.  DeepMD emits `Loading outdated pt2
file`; the package loads and evaluates correctly, but the warning is retained
as a reproducibility risk.  The runtime is Python 3.12 although the repository
metadata currently declares `<3.12`; all tests and calculations below passed.

## Phase 3A: single-point validation

The original MQ structure has total energy -4518.55160 eV
(-5.04302634 eV/atom), vector-force RMS 0.55422 eV/A and maximum force
2.93962 eV/A.  Repeating the same explicit uncached call three times gave an
energy range of 1.04e-5 eV and maximum force-component range of 4.16e-6 eV/A,
consistent with the float32 packaged model.

Gaussian coordinate perturbations gave smooth, physically ordered responses:

| displacement RMS (A) | delta E (eV) | force RMS (eV/A) | max force (eV/A) |
|---:|---:|---:|---:|
| 0.005 | +0.4801 | 0.5940 | 2.5217 |
| 0.010 | +1.9046 | 0.7094 | 3.4161 |
| 0.020 | +7.1447 | 1.0391 | 5.6448 |

The Phase 2.5 no-scale final structures are far above the MQ structure in the
DPA4 landscape: seeds 101/202/303 are respectively +274.80, +270.27 and
+281.18 eV, and have force RMS values 2.95--2.99 eV/A.  This is the expected
motivation for adding an energy term rather than evidence that a PDF-only
structure is energetically valid.

## Force finite-difference check

Central differences were evaluated for Li-x, Si-y, P-z and O-x.  At delta =
0.01 A, all relative errors are 0.013--0.143% (absolute errors
2.93e-5--2.29e-4 eV/A).  At delta = 0.001 A the relative errors are
0.13--4.99%; the degradation at the smaller displacement is consistent with
float32 energy noise.  The stable 0.01 A result verifies the energy/force sign,
units, atom ordering and type mapping.

## GPU benchmark

The DeepMD ASE interface returns energy, forces and virial together, so an
"energy-only" HRMC request has the same model-call cost.

- First warm call: 1.084 s; model load: 1.132 s.
- 100 hot calls: median 0.025752 s, mean 0.026169 s, p95 0.027615 s.
- Median throughput: 38.83 calls/s.
- Peak GPU memory: 2839 MiB allocated, 3298 MiB reserved.

The raw model-call lower bounds are approximately 4.3 min for 10k, 42.9 min
for 100k and 7.15 h for 1M attempts, before PDF/CN/RMC overhead.

## Fixed-cell FIRE validation

FIRE used the original cell, `dt=0.05`, `maxstep=0.05 A`, and target
`fmax=0.05 eV/A`.  Five, ten and twenty optimizer steps were deliberately
short validation runs and did not claim convergence.

| source / steps | delta E (eV) | force RMS | Rw no-scale | P/Si fourfold | P/Si angle RMS (deg) | P/Si CSM | HS violations |
|---|---:|---:|---:|---:|---:|---:|---:|
| MQ / 0 | 0.000 | 0.554 | 0.7131 | 1.0 / 1.0 | 3.67 / 2.95 | 0.359 / 0.601 | 0 |
| MQ / 5 | -2.866 | 0.302 | 0.7187 | 1.0 / 1.0 | 3.46 / 2.82 | 0.316 / 0.574 | 0 |
| MQ / 10 | -4.545 | 0.224 | 0.7240 | 1.0 / 1.0 | 3.31 / 2.67 | 0.293 / 0.551 | 0 |
| MQ / 20 | -6.855 | 0.134 | 0.7368 | 1.0 / 1.0 | 3.16 / 2.32 | 0.270 / 0.514 | 0 |
| Phase2.5 seed101 / 0 | 0.000 | 2.954 | 0.1221 | 1.0 / 1.0 | 14.49 / 13.77 | 4.504 / 4.745 | 0 |
| seed101 / 5 | -20.735 | 2.598 | 0.1305 | 1.0 / 1.0 | 14.01 / 13.55 | 4.175 / 4.545 | 1 |
| seed101 / 10 | -38.943 | 2.287 | 0.1485 | 1.0 / 1.0 | 13.54 / 13.33 | 3.875 / 4.349 | 1 |
| seed101 / 20 | -69.122 | 1.798 | 0.1966 | 1.0 / 1.0 | 12.59 / 12.83 | 3.342 / 3.965 | 1 |

Energy, forces and tetrahedral geometry improve consistently under FIRE.  The
no-scale PDF fit worsens, demonstrating the intended competition between the
experimental and energy objectives.  Short FIRE from the Phase 2.5 endpoint
also crosses one frozen MQ hard-sphere cutoff; therefore Phase 3B must rebuild
caches after FIRE and explicitly audit/reject unacceptable geometry instead of
assuming relaxation preserves the RMC constraints.

## Gate decision and limits

Gate A passes because the model loads on the target GPU, contains all required
elements, gives finite/repeatable E/F values, passes finite differences at a
numerically stable displacement, has usable throughput, and produces monotonic
energy/force improvement during fixed-cell FIRE.  Development may proceed to
Phase 3B, then Phase 3C, with bounded smoke/pilot runs.

This is not an independent DFT accuracy validation and does not establish that
the model extrapolates safely across every HRMC structure.  The obsolete PT2
warning, Python metadata mismatch, Phase2.5 high-energy endpoints and the one
FIRE hard-sphere crossing must remain visible in later reports.  No million-step
Phase 3 production run is authorized by this gate.

## Reproduction

From the repository root in the `dpa4_torch211` environment:

```bash
export PYTHONPATH=$PWD/src
export DPA4_MODEL_PATH=/root/autodl-tmp/CODEX/DPA4_Neo_PZS_NPT_LoRA.pt2
python scripts/audit_phase3_gpu.py
python scripts/validate_dpa4_pzs.py
python scripts/benchmark_dpa4_pzs.py
python scripts/run_phase3A_fire.py
pytest -q
```

Machine-readable results are in this directory, especially
`environment_manifest.json`, `single_point_energy_force.csv`,
`force_fd_check.csv`, `gpu_benchmark.json`, `fire_comparison.csv`, and
`fire_details.json`.
