# Phase 3B alternating HRMC pilot report

Date: 2026-11-20. Gate B status: **PASS for bounded Phase 3C development**.

The implementation in `src/dpa4_xray_hrmc/rmc/runner_phase3.py` starts from the
original MQ structure, uses the unchanged Phase 2.5 no-scale PDF+CN Metropolis
objective, stops each full block at 500 accepted moves, then performs at most
five fixed-cell DPA4/FIRE steps.  FIRE energy is not included in RMC acceptance.
After every FIRE call the incremental PDF and CN engines are reconstructed from
all coordinates.  The hard-sphere model remains the one generated from MQ.

Smoke command:

```bash
python scripts/run_phase3B_alternating.py --attempts 300 \
  --accepted-per-block 50 --tag smoke_300
```

It completed three cache-rebuild/FIRE cycles: 300 attempts, 166 accepted,
3 hard-sphere rejects, 131 Metropolis rejects, `Rw=0.6791`, and no P/Si
fourfold loss.

Pilot command:

```bash
python scripts/run_phase3B_alternating.py --attempts 10000 \
  --accepted-per-block 500 --seed 101 --tag pilot_10k_seed101
```

The actual result was 10,000 attempts, 4,384 accepted, 992 hard-sphere rejects,
4,624 Metropolis rejects, and eight complete 500-accepted FIRE blocks in
37.26 s.  The remaining 384 accepted moves form an incomplete final RMC block
and therefore have no following FIRE call.  `Rw(no_scale)` improved from
0.71309 to 0.18120 and Phase 2.5 total loss from 32377.39 to 2090.66.  P and Si
fourfold fractions remained 1.0.

Each successful FIRE block lowered its pre-FIRE energy by 10.55--17.04 eV.
Nevertheless the final total energy, -4397.55 eV, is 121.00 eV above the
original MQ energy because PDF-driven RMC moves raise energy between FIRE
blocks.  This is a non-equilibrium refinement/relaxation algorithm, not
sampling from a combined Boltzmann target.

Block 4 produced a frozen-MQ hard-sphere violation and was correctly rolled
back as a whole; coordinates and both incremental caches were rebuilt from the
pre-FIRE structure.  The later blocks continued successfully.  This observed
rollback is evidence that the failure path is active, not merely unit-tested.

Outputs are under `results/PZS_phase3/phase3B/`.  `trace.json` is block resolved
and `summary.json`, final CIF/extxyz and FIRE logs are retained.  This is a
single-seed 10k pilot, not a convergence claim; no 100k or 1M run was launched.
