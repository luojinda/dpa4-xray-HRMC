"""Isotropic Debye-Waller conventions used by the forward model."""

from __future__ import annotations

import numpy as np


def biso_to_uiso(biso_A2: float | np.ndarray) -> float | np.ndarray:
    if np.any(np.asarray(biso_A2) < 0):
        raise ValueError("Biso cannot be negative")
    return np.asarray(biso_A2) / (8.0 * np.pi**2)


def pair_debye_waller(q_Ainv: np.ndarray, biso_a_A2: float, biso_b_A2: float) -> np.ndarray:
    """Distinct-pair damping exp[-Q^2(Ua+Ub)/2]."""
    ua = float(biso_to_uiso(biso_a_A2))
    ub = float(biso_to_uiso(biso_b_A2))
    return np.exp(-0.5 * np.asarray(q_Ainv, dtype=float) ** 2 * (ua + ub))
