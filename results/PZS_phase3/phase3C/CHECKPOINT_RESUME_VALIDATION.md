# Inline checkpoint/resume validation

Date: 2026-11-20

Before launching the million-attempt Inline chain, the runner was extended to
atomically save positions, counters, cached energy, RNG states, trace, elapsed
wall time, model hash and run identity.  Resume rejects a checkpoint whose
seed, temperature, move size, model hash or atom count differs.

An actual GPU integration test used seed 991 and Inline 100 K:

1. run attempts 1--200 and save checkpoints every 100 attempts;
2. restart Python/DeepMD and resume the same checkpoint to attempt 300;
3. independently run a continuous 300-attempt chain with the same seed;
4. compare counters, coordinates, PDF objective and DPA4 energy.

Both paths produced 137 accepted moves, one hard-sphere rejection and 162
Metropolis rejections.  The maximum coordinate difference was exactly 0 A and
the PDF loss difference was exactly zero.  The final DPA4 total energies
differed by 8.96e-6 eV, consistent with the packaged float32 model's measured
repeatability.  The checkpoint/resume path is therefore accepted for the
million-attempt production run.
