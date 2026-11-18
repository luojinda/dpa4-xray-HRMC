"""Partial RDF -> Faber-Ziman S(Q) -> reduced PDF G(r)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .form_factors import XrayWeights
from .partial_rdf import Pair, PartialRDF
from .thermal import pair_debye_waller


@dataclass(frozen=True)
class PDFResult:
    q_Ainv: np.ndarray
    partial_sq_minus_one: dict[Pair, np.ndarray]
    s_q: np.ndarray
    f_q_Ainv: np.ndarray
    r_A: np.ndarray
    g_r_A2: np.ndarray


def _integration_weights(x: np.ndarray) -> np.ndarray:
    """Trapezoid weights for an arbitrary strictly increasing 1D grid."""
    if x.ndim != 1 or len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("Integration grid must be strictly increasing")
    weights = np.empty_like(x)
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    return weights


def sine_transform(q_Ainv: np.ndarray, f_q_Ainv: np.ndarray, r_A: np.ndarray) -> np.ndarray:
    """Return (2/pi) integral F(Q) sin(Qr) dQ using trapezoid weights."""
    q = np.asarray(q_Ainv, dtype=float)
    f_q = np.asarray(f_q_Ainv, dtype=float)
    r = np.asarray(r_A, dtype=float)
    if f_q.shape != q.shape:
        raise ValueError("F(Q) and Q must have the same shape")
    return (2.0 / np.pi) * (np.sin(np.outer(r, q)) @ (f_q * _integration_weights(q)))


def partial_sq_from_rdf(rdf: PartialRDF, q_Ainv: np.ndarray) -> dict[Pair, np.ndarray]:
    q = np.asarray(q_Ainv, dtype=float)
    r = rdf.r_A
    rho0 = sum(rdf.element_counts.values()) / rdf.volume_A3
    # np.sinc(x/pi) = sin(x)/x and is stable at x=0.
    kernel = np.sinc(np.outer(q, r) / np.pi)
    radial_factor = 4.0 * np.pi * rho0 * r**2 * rdf.dr_A
    return {pair: kernel @ ((values - 1.0) * radial_factor) for pair, values in rdf.g.items()}


def forward_transform(
    rdf: PartialRDF,
    weights: XrayWeights,
    output_r_A: np.ndarray,
    biso_A2: float,
    lorch_mod: bool = False,
) -> PDFResult:
    q = weights.q_Ainv
    partial = partial_sq_from_rdf(rdf, q)
    total_minus_one = np.zeros_like(q)
    for pair, values in partial.items():
        damping = pair_debye_waller(q, biso_A2, biso_A2)
        total_minus_one += weights.pair_weights[pair] * damping * values
    s_q = 1.0 + total_minus_one
    f_q = q * total_minus_one
    if lorch_mod:
        delta_q = q[-1] - q[0]
        window_argument = np.pi * (q - q[0]) / delta_q
        lorch = np.sinc(window_argument / np.pi)
        f_q = f_q * lorch
    r_out = np.asarray(output_r_A, dtype=float)
    g_r = sine_transform(q, f_q, r_out)
    if len(r_out) and np.isclose(r_out[0], 0.0):
        g_r[0] = 0.0
    if not all(np.all(np.isfinite(x)) for x in (s_q, f_q, g_r)):
        raise FloatingPointError("Non-finite values in PDF transform")
    return PDFResult(q, partial, s_q, f_q, r_out, g_r)
