# Phase 3C inline HRMC pilot report

Date: 2026-11-20. Gate C status: **PASS for bounded parameter studies**.

`run_inline` in `src/dpa4_xray_hrmc/rmc/runner_phase3.py` uses every legal
single-atom trial's full 896-atom DPA4 total energy.  The actual exponent is

`delta_L = delta_L_Phase2.5 + (E_trial - E_current)/(kB*T_eff)`

with `kB=8.617333262145e-5 eV/K`.  No per-atom energy, extra scale fit, FIRE or
duplicate sigma normalization is used.  Hard-sphere rejects make zero model
calls.  Rejected legal trials leave coordinates, PDF, CN and cached current
energy unchanged.  Full PDF/CN drift checks run every 100 attempts and a final
independent DPA4 call audits cached energy.

The 300-attempt temperature scan from the identical original MQ coordinates was:

| T (K) | accepted | HS reject | acceptance | final Rw | delta E from MQ (eV) | wall (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 138 | 3 | 0.460 | 0.65389 | +6.0270 | 8.98 |
| 175 | 153 | 4 | 0.510 | 0.65248 | +7.6159 | 9.07 |
| 300 | 161 | 4 | 0.537 | 0.65228 | +8.5216 | 8.82 |

The energy term does not freeze the chain.  As expected, lower T retains lower
energy and has lower acceptance, while all three short chains improve PDF.

The selected 175 K, seed-101, 1,000-attempt pilot used 974 model calls and took
32.09 s including ten full PDF/CN drift checks. It accepted 479 moves, rejected 28 by hard sphere and 493 by
Metropolis.  `Rw(no_scale)` changed 0.71309 to 0.54617, P/Si fourfold fractions
remained 1.0, and the total energy changed by +23.5483 eV.  The independent
final energy drift was +2.13e-6 eV, consistent with the packaged float32 model.

Auditable examples from this real chain:

- Accepted step 1: `delta_L_Phase2.5=-43.9741`, `delta_E=+0.072034 eV`,
  `delta_E/kBT=+4.77670`, total `delta_L=-39.1974`, probability 1,
  random number 0.791073.
- Rejected step 2: `delta_L_Phase2.5=+18.5307`, `delta_E=+0.0823485 eV`,
  `delta_E/kBT=+5.46066`, total `delta_L=+23.9914`, probability
  3.8078e-11, random number 0.104506.

Commands:

```bash
python scripts/run_phase3C_inline.py --attempts 300 --temperature 100 --tag scan_T100_300
python scripts/run_phase3C_inline.py --attempts 300 --temperature 175 --tag scan_T175_300
python scripts/run_phase3C_inline.py --attempts 300 --temperature 300 --tag scan_T300_300
python scripts/run_phase3C_inline.py --attempts 1000 --temperature 175 --tag pilot_T175_1k_seed101
```

Outputs are under `results/PZS_phase3/phase3C/`.  These runs establish correct
execution and short-range trends, not equilibrium or statistical convergence.
At the measured ~0.029 s/legal trial, 10k/100k/1M inline attempts are roughly
5 min/49 min/8.1 h plus variation in hard-sphere legal rate.  No 100k or 1M run
was launched.

