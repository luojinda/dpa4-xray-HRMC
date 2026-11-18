#!/usr/bin/env python
"""Run the Phase 1 periodic X-ray PDF calculation from YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from dpa4_xray_hrmc.xray_pdf import load_config, run_calculation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML configuration")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    project_root = config_path.parent.parent
    config = load_config(config_path)
    summary = run_calculation(config, project_root)
    print(f"Structure: {summary.structure.formula} ({summary.structure.n_atoms} atoms)")
    print(f"Elements: {summary.structure.element_counts}")
    print(f"Output: {summary.output_directory}")
    print(f"Wall time: {summary.elapsed_seconds:.3f} s")
    print(f"Sampled peak RSS: {summary.peak_rss_mb:.1f} MiB")


if __name__ == "__main__":
    main()
