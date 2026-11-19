"""End-to-end calculation, convergence diagnostics, and artifact writing."""

from __future__ import annotations

import json
import os
import platform
import threading
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import psutil
from scipy.signal import find_peaks

from .config import PDFConfig
from .form_factors import XrayWeights, q_dependent_weights
from .partial_rdf import PartialRDF, compute_partial_rdfs, rebin_rdf, truncate_rdf
from .plotting import plot_pdf
from .structure import StructureInfo, load_structure
from .thermal import biso_to_uiso
from .transform import PDFResult, forward_transform


@dataclass(frozen=True)
class RunSummary:
    output_directory: str
    structure: StructureInfo
    primary_result: PDFResult
    elapsed_seconds: float
    peak_rss_mb: float
    convergence: dict[str, dict[str, float]]


class _MemorySampler:
    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_MemorySampler":
        process = psutil.Process(os.getpid())

        def sample() -> None:
            while not self._stop.is_set():
                self.peak = max(self.peak, process.memory_info().rss)
                self._stop.wait(self.interval_seconds)
            self.peak = max(self.peak, process.memory_info().rss)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join()


def _inclusive_grid(start: float, stop: float, step: float) -> np.ndarray:
    n_steps = int(round((stop - start) / step))
    if not np.isclose(start + n_steps * step, stop, atol=1e-10):
        raise ValueError(f"Grid [{start}, {stop}] is not divisible by step {step}")
    return np.linspace(start, stop, n_steps + 1)


def _save_table(path: Path, columns: list[np.ndarray], names: list[str]) -> None:
    data = np.column_stack(columns)
    np.savetxt(path, data, fmt="%.10e", header=" ".join(names))


def _write_rdf(path: Path, rdf: PartialRDF) -> None:
    pairs = sorted(rdf.g)
    names = ["r_A"] + [f"g_{a}-{b}" for a, b in pairs]
    _save_table(path, [rdf.r_A] + [rdf.g[pair] for pair in pairs], names)


def _write_form_factors(path: Path, weights: XrayWeights) -> None:
    elements = sorted(weights.form_factors)
    _save_table(
        path,
        [weights.q_Ainv] + [weights.form_factors[element] for element in elements],
        ["Q_Ainv"] + [f"f0_{element}" for element in elements],
    )


def _write_pair_weights(path: Path, weights: XrayWeights) -> None:
    pairs = sorted(weights.pair_weights)
    _save_table(
        path,
        [weights.q_Ainv] + [weights.pair_weights[pair] for pair in pairs],
        ["Q_Ainv"] + [f"w_{a}-{b}" for a, b in pairs],
    )


def _difference_metrics(reference: PDFResult, trial: PDFResult, lo: float, hi: float) -> dict[str, float]:
    mask = (reference.r_A >= lo) & (reference.r_A <= hi)
    difference = trial.g_r_A2[mask] - reference.g_r_A2[mask]
    reference_values = reference.g_r_A2[mask]
    rms = float(np.sqrt(np.mean(difference**2)))
    scale = float(np.sqrt(np.mean(reference_values**2)))
    return {
        "rms_A-2": rms,
        "relative_rms": rms / scale if scale > 0 else 0.0,
        "max_abs_A-2": float(np.max(np.abs(difference))),
    }


def _peak_positions(result: PDFResult, lo: float, hi: float, count: int = 8) -> list[tuple[float, float]]:
    mask = (result.r_A >= lo) & (result.r_A <= hi)
    r = result.r_A[mask]
    values = result.g_r_A2[mask]
    prominence = max(0.03 * (float(np.max(values)) - float(np.min(values))), 1e-8)
    indices, properties = find_peaks(values, prominence=prominence)
    if len(indices) == 0:
        return []
    ranking = np.argsort(properties["prominences"])[::-1][:count]
    selected = sorted(indices[ranking], key=lambda index: r[index])
    return [(float(r[index]), float(values[index])) for index in selected]


def _shell_weighted_tail_means(rdf: PartialRDF, width_A: float = 5.0) -> dict[str, float]:
    lo = max(0.0, rdf.rcut_A - width_A)
    mask = rdf.r_A >= lo
    shell = (4.0 * np.pi / 3.0) * (rdf.edges_A[1:] ** 3 - rdf.edges_A[:-1] ** 3)
    return {
        f"{a}-{b}": float(np.sum(values[mask] * shell[mask]) / np.sum(shell[mask]))
        for (a, b), values in rdf.g.items()
    }


def _write_validation(
    path: Path,
    info: StructureInfo,
    config: PDFConfig,
    rdf: PartialRDF,
    weights: XrayWeights,
    convergence: dict[str, dict[str, float]],
    primary: PDFResult,
    elapsed: float,
    peak_rss_mb: float,
) -> None:
    peaks_short = _peak_positions(primary, 1.5, 4.5)
    peaks_long = _peak_positions(primary, 1.5, 10.0)
    weight_sum_error = float(
        np.max(np.abs(np.sum(np.stack(list(weights.pair_weights.values())), axis=0) - 1.0))
    )
    negative_s_count = int(np.count_nonzero(primary.s_q < 0.0))
    s_min_index = int(np.argmin(primary.s_q))
    lines = [
        f"# {config.sample} Phase 1 validation report",
        "",
        "## Structure",
        "",
        f"- ASE formula: `{info.formula}`; atoms: {info.n_atoms}; counts: `{info.element_counts}`.",
        f"- Cell volume: {info.volume_A3:.6f} Å³; number density: {info.number_density_A3:.8f} Å⁻³.",
        f"- Cell lengths: {[round(x, 6) for x in info.cell_lengths_A]} Å; angles: {[round(x, 6) for x in info.cell_angles_deg]}°.",
        f"- PBC: {info.pbc}; minimum periodic non-self separation: {info.minimum_periodic_separation_A:.6f} Å.",
        f"- Explicit occupancies checked: {len(info.occupancies)} sites, all equal to 1.0.",
        "",
        "## Numerical model",
        "",
        f"- RDF convention: ordered central-α/neighbour-β counts normalized by `Nα ρ0 cβ ΔV`; {len(rdf.g)} α≤β partials.",
        f"- Primary RDF: `dr={rdf.dr_A:g} Å`, bin centres, `r_cut={rdf.rcut_A:g} Å`; explicit image bounds {rdf.translation_bounds} ({rdf.n_translations} candidate translations).",
        f"- Q grid: {config.qmin_Ainv:g}–{config.qmax_Ainv:g} Å⁻¹ in {config.qstep_Ainv:g} Å⁻¹ steps; rectangular Q window (Lorch={config.lorch_mod}).",
        f"- `Biso={config.biso_A2:g} Å²`, hence `Uiso={float(biso_to_uiso(config.biso_A2)):.10f} Å²`. Distinct-pair partials are multiplied once by `exp[-Q²(Uα+Uβ)/2]`; no qdamp/qbroad.",
        "- Neutral elastic `f0(Q)` uses `xraydb.f0(element, Q/(4π))`; anomalous terms are not included.",
        "- `G(0)` is explicitly set to zero; no smoothing, scale adjustment, background subtraction, or r shift is applied.",
        "",
        "## Diagnostics",
        "",
        f"- Maximum `|sum(w_ab)-1|` over all Q points: {weight_sum_error:.3e}; all stored weights are finite and non-negative.",
        f"- Primary S(Q) range: {float(np.min(primary.s_q)):.6g} to {float(np.max(primary.s_q)):.6g}; {negative_s_count} of {len(primary.s_q)} samples are negative. The minimum occurs at Q={primary.q_Ainv[s_min_index]:g} Å⁻¹.",
        "- Negative S(Q) samples are finite-r-cut termination ringing around sharp ideal-crystal Bragg features, not a physical claim of negative scattering intensity. A smooth real-space window was deliberately not applied to the default result.",
        f"- Tail shell-volume-weighted g(r) means over the last 5 Å: `{_shell_weighted_tail_means(rdf)}`.",
        f"- Prominent peaks in 1.5–4.5 Å (r, G): `{[(round(r, 3), round(g, 4)) for r, g in peaks_short]}`.",
        f"- Prominent peaks in 1.5–10 Å (r, G): `{[(round(r, 3), round(g, 4)) for r, g in peaks_long]}`.",
        "",
        "### Convergence differences",
        "",
        "Each entry is measured against the primary 40 Å / 0.01 Å result. Finite-cutoff crystal PDFs need not converge monotonically because an ideal periodic crystal has long-range order.",
        "",
        "| variant/range | RMS (Å⁻²) | relative RMS | max abs (Å⁻²) |",
        "|---|---:|---:|---:|",
    ]
    for name, metrics in convergence.items():
        lines.append(
            f"| {name} | {metrics['rms_A-2']:.6g} | {metrics['relative_rms']:.6g} | {metrics['max_abs_A-2']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Performance and interpretation",
            "",
            f"- Wall time: {elapsed:.3f} s; sampled peak process RSS: {peak_rss_mb:.1f} MiB; platform: `{platform.platform()}`.",
            "- This is a finite-real-space-cutoff approximation to an ideal periodic crystal, not a claim of an exact infinite-crystal PDF.",
            "- The ideal optimized CIF lacks nanocrystal size effects, static/local disorder, and experimental qdamp/qbroad. It should not be expected to reproduce every experimental peak width or medium-range amplitude.",
            "- Experimental agreement metrics are outside this forward-calculation report; use the dedicated comparison output when experimental data are supplied.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_calculation(config: PDFConfig, project_root: str | Path = ".") -> RunSummary:
    config.validate()
    root = Path(project_root).resolve()
    structure_path = (root / config.structure).resolve()
    output_dir = (root / config.output_directory).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    q = _inclusive_grid(config.qmin_Ainv, config.qmax_Ainv, config.qstep_Ainv)
    r_out = _inclusive_grid(config.output_rmin_A, config.output_rmax_A, config.output_rstep_A)

    started = time.perf_counter()
    with _MemorySampler() as memory:
        info = load_structure(structure_path)
        rdf = compute_partial_rdfs(info.atoms, config.rdf_rcut_A, config.rdf_dr_A)
        weights = q_dependent_weights(rdf.element_counts, q)
        primary = forward_transform(rdf, weights, r_out, config.biso_A2, config.lorch_mod)

        variants: dict[str, PDFResult] = {}
        for cutoff in config.convergence_rcuts_A:
            if np.isclose(cutoff, config.rdf_rcut_A):
                continue
            variants[f"rcut_{cutoff:g}"] = forward_transform(
                truncate_rdf(rdf, cutoff), weights, r_out, config.biso_A2, config.lorch_mod
            )
        coarse_rdf = rebin_rdf(rdf, config.convergence_rdf_dr_A)
        variants[f"dr_{config.convergence_rdf_dr_A:g}"] = forward_transform(
            coarse_rdf, weights, r_out, config.biso_A2, config.lorch_mod
        )
    elapsed = time.perf_counter() - started
    peak_rss_mb = memory.peak / (1024.0**2)

    convergence: dict[str, dict[str, float]] = {}
    for name, result in variants.items():
        for lo, hi in config.plot_ranges_A:
            convergence[f"{name}, {lo:g}-{hi:g} A"] = _difference_metrics(primary, result, lo, hi)

    _write_rdf(output_dir / f"{config.sample}_partial_RDF.dat", rdf)
    _write_form_factors(output_dir / "form_factors.dat", weights)
    _write_pair_weights(output_dir / "pair_weights.dat", weights)
    _save_table(
        output_dir / f"{config.sample}_SQ.dat",
        [primary.q_Ainv, primary.s_q, primary.f_q_Ainv],
        ["Q_Ainv", "S_Q", "F_Q_Ainv"],
    )
    _save_table(
        output_dir / f"{config.sample}_xray_Gr.dat",
        [primary.r_A, primary.g_r_A2],
        ["r_A", "G_r_A-2"],
    )
    for lo, hi in config.plot_ranges_A:
        suffix = f"{lo:g}_{hi:g}".replace(".", "p")
        plot_pdf(
            primary.r_A,
            primary.g_r_A2,
            output_dir / f"{config.sample}_PDF_{suffix}.png",
            (lo, hi),
            (config.qmin_Ainv, config.qmax_Ainv),
            config.biso_A2,
            config.lorch_mod,
            config.rdf_rcut_A,
            config.sample,
        )

    (output_dir / "structure.json").write_text(
        json.dumps(info.serializable(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    parameters = {
        **config.to_dict(),
        "resolved_structure": str(structure_path),
        "q_points": len(q),
        "output_r_points": len(r_out),
        "uiso_A2": float(biso_to_uiso(config.biso_A2)),
        "thermal_pair_factor": "exp[-Q^2*(U_alpha+U_beta)/2]",
        "form_factor_argument": "Q_Ainv/(4*pi)",
        "extra_instrument_resolution": None,
        "rdf_translation_bounds": rdf.translation_bounds,
        "rdf_candidate_translations": rdf.n_translations,
        "wall_time_seconds": elapsed,
        "sampled_peak_rss_mb": peak_rss_mb,
        "software_versions": {
            "python": platform.python_version(),
            **{
                package: version(package)
                for package in ("numpy", "scipy", "ase", "matplotlib", "PyYAML", "xraydb", "psutil")
            },
        },
    }
    (output_dir / "parameters.json").write_text(
        json.dumps(parameters, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_validation(
        output_dir / "validation_report.md",
        info,
        config,
        rdf,
        weights,
        convergence,
        primary,
        elapsed,
        peak_rss_mb,
    )
    return RunSummary(str(output_dir), info, primary, elapsed, peak_rss_mb, convergence)
