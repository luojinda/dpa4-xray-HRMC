#!/usr/bin/env python
"""Run or resume one Phase 2 Pure-RMC PZS chain."""

from __future__ import annotations

import argparse
from pathlib import Path

from dpa4_xray_hrmc.rmc.config import load_phase2_config
from dpa4_xray_hrmc.rmc.objectives import ScaleMode
from dpa4_xray_hrmc.rmc.runner import run_chain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pzs_phase2_common.yaml")
    parser.add_argument("--mode", required=True, choices=[mode.value for mode in ScaleMode])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--steps", required=True, type=int, help="Total attempted moves desired")
    parser.add_argument("--output-root", help="Override output root, e.g. AutoDL data disk")
    parser.add_argument("--resume", help="Checkpoint .npz to resume")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    result = run_chain(
        load_phase2_config(config_path),
        args.mode,
        args.seed,
        args.steps,
        config_path.parent.parent,
        args.output_root,
        args.resume,
    )
    print(f"Output: {result.output_directory}")
    print(
        f"Mode={result.mode}; seed={result.seed}; attempted={result.attempted}; "
        f"accepted={result.accepted}; HS rejected={result.hard_sphere_rejected}"
    )
    print(
        f"Final loss={result.final_loss:.10g}, scale={result.final_scale:.8g}; "
        f"best loss={result.best_loss:.10g} at step {result.best_step}"
    )
    print(f"Wall time: {result.wall_seconds:.3f} s")


if __name__ == "__main__":
    main()
