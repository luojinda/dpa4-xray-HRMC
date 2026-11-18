"""Neutral-atom X-ray form factors and Q-dependent Faber-Ziman weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xraydb

from .partial_rdf import Pair, element_pairs


@dataclass(frozen=True)
class XrayWeights:
    q_Ainv: np.ndarray
    form_factors: dict[str, np.ndarray]
    concentrations: dict[str, float]
    pair_weights: dict[Pair, np.ndarray]


def neutral_form_factors(elements: list[str], q_Ainv: np.ndarray) -> dict[str, np.ndarray]:
    q = np.asarray(q_Ainv, dtype=float)
    if np.any(q < 0):
        raise ValueError("Q must be non-negative")
    # xraydb uses sin(theta)/lambda, whereas scattering Q=4*pi*sin(theta)/lambda.
    xraydb_q = q / (4.0 * np.pi)
    factors = {element: np.asarray(xraydb.f0(element, xraydb_q), dtype=float) for element in elements}
    for element, values in factors.items():
        if values.shape != q.shape or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"Invalid xraydb f0 values for {element}")
    return factors


def q_dependent_weights(element_counts: dict[str, int], q_Ainv: np.ndarray) -> XrayWeights:
    elements = sorted(element_counts)
    total = float(sum(element_counts.values()))
    concentrations = {element: element_counts[element] / total for element in elements}
    factors = neutral_form_factors(elements, q_Ainv)
    mean_f = sum(concentrations[element] * factors[element] for element in elements)
    weights: dict[Pair, np.ndarray] = {}
    for a, b in element_pairs(elements):
        multiplicity = 1.0 if a == b else 2.0
        weights[(a, b)] = (
            multiplicity
            * concentrations[a]
            * concentrations[b]
            * factors[a]
            * factors[b]
            / mean_f**2
        )
    weight_sum = np.sum(np.stack(list(weights.values())), axis=0)
    if not np.allclose(weight_sum, 1.0, rtol=1e-12, atol=1e-12):
        raise ValueError("Faber-Ziman pair weights do not sum to one")
    return XrayWeights(np.asarray(q_Ainv), factors, concentrations, weights)
