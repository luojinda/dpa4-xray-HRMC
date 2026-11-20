# Phase 3 first-round completion summary

Gate 0 and Phase 3A passed, followed sequentially by Phase 3B and Phase 3C.
Separate detailed reports and machine-readable results are in `phase3A/`,
`phase3B/`, and `phase3C/`.

The main physical result of this first round is the expected PDF/energy tradeoff.
The original MQ has poor `Rw=0.7131` but low DPA4 energy (-4518.55 eV).
The 3B 10k pilot reaches `Rw=0.1812`; five-step FIRE lowers energy inside each
successful block, but intervening PDF RMC moves leave the final energy 121.0 eV
above MQ.  The 3C 175 K, 1k pilot reaches `Rw=0.5462` while limiting the energy
increase to 23.55 eV.  Both retain 100% P/Si fourfold coordination in the runs
performed.

Implemented artifacts include a one-load DPA4 backend, force finite-difference
and GPU benchmarks, fixed-cell FIRE validation, alternating full-cache rebuild
and rollback, and inline total-energy Metropolis acceptance with auditable move
records.  The complete test suite passes.

This round intentionally did not run 100k or 1M chains.  Results are pilot
evidence only; they do not demonstrate convergence, detailed balance for the
alternating method, or independent DFT accuracy of the DPA4 model.
