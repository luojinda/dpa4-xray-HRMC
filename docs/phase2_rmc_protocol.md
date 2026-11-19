# Phase 2 Pure-RMC protocol

## Fixed inputs

- Structure: `PZS_quench_222.cif`, fixed 896 atoms and fixed periodic cell.
- Experiment: ZXS_1 reduced X-ray PDF, fit over 1.5--10.0 A.
- Forward model: Q=0.9--16.0 A^-1, dQ=0.05 A^-1, Biso=0.3 A^2,
  no Lorch window, RDF dr=0.01 A and r_cut=12 A.
- Uniform residual point weights.
- Pair hard-sphere cutoffs are 0.95 times the initial MQ minimum for each
  element pair.
- Cartesian displacement components are uniform in [-0.10, 0.10] A.
- `sigma_RMC=0.03 A^-2` and the summed squared-residual Metropolis rule are
  used for every mode and seed.

The displacement and sigma values were frozen after the common-seed 10k pilot:
all three modes had approximately 40--42% acceptance.  They are not physical
temperature or experimental uncertainty estimates.

## Scale modes

- `no_scale`: sampling scale exactly 1.
- `fixed_initial`: the initial least-squares `s0` remains fixed.
- `bounded_dynamic`: every legal proposal profiles scale analytically inside
  the immutable `[0.8*s0, 1.2*s0]` bounds.

Atom-index, displacement, and Metropolis-uniform random streams are independent
and identically seeded across modes.  Different acceptance decisions cause the
trajectories to diverge, as intended.

## Stages and persistence

- 10k: one common seed for all modes, used to freeze proposal parameters.
- 100k: three modes by three seeds, with full checks at 50k and 100k.
- 1M: three modes by three seeds, 9 chains total.
- Trace every 1,000 attempts; structure/checkpoint/full incremental validation
  every 50,000 attempts.

Each checkpoint stores positions, histograms, current G, A/B coefficients,
scale, counters, best state, and all RNG states.  A real 100k-to-101k resume
test is performed before the formal runs.

No energy, DPA4, coordination, bond-angle, or rigid-unit term participates in
proposal scoring or acceptance.  Coordination and polyhedral statistics are
post-run diagnostics only.
