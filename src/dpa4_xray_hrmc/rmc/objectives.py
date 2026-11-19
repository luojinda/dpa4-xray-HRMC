"""Pure functions for the three Phase 2 scale strategies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ScaleMode(str, Enum):
    NO_SCALE = "no_scale"
    FIXED_INITIAL = "fixed_initial"
    BOUNDED_DYNAMIC = "bounded_dynamic"


@dataclass(frozen=True)
class Score:
    loss: float
    scale: float
    A: float
    B: float


@dataclass(frozen=True)
class ScaleObjective:
    target_A2: np.ndarray
    weights: np.ndarray
    s0: float
    lower_scale: float
    upper_scale: float
    C: float

    @classmethod
    def from_initial(
        cls,
        initial_calculated_A2: np.ndarray,
        target_A2: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> "ScaleObjective":
        x = np.asarray(initial_calculated_A2, dtype=float)
        y = np.asarray(target_A2, dtype=float)
        w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
        if x.shape != y.shape or w.shape != y.shape:
            raise ValueError("Initial, target, and weight arrays must have equal shape")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Objective inputs must be finite")
        if np.any(w < 0.0) or not np.any(w > 0.0):
            raise ValueError("Weights must be non-negative with at least one positive value")
        A = float(np.dot(w * x, x))
        B = float(np.dot(w * x, y))
        if not np.isfinite(A) or A <= 0.0:
            raise ValueError("Initial A must be positive")
        s0 = B / A
        if not np.isfinite(s0) or s0 <= 0.0:
            raise ValueError("Initial optimal scale must be positive")
        return cls(y, w, s0, 0.8 * s0, 1.2 * s0, float(np.dot(w * y, y)))

    def coefficients(self, calculated_A2: np.ndarray) -> tuple[float, float]:
        x = np.asarray(calculated_A2, dtype=float)
        return float(np.dot(self.weights * x, x)), float(np.dot(self.weights * x, self.target_A2))

    def scale_for(self, mode: ScaleMode | str, A: float, B: float) -> float:
        selected = ScaleMode(mode)
        if selected is ScaleMode.NO_SCALE:
            return 1.0
        if selected is ScaleMode.FIXED_INITIAL:
            return self.s0
        if A <= 0.0 or not np.isfinite(A):
            raise FloatingPointError("Dynamic scale requires positive finite A")
        return float(np.clip(B / A, self.lower_scale, self.upper_scale))

    def score_from_coefficients(self, mode: ScaleMode | str, A: float, B: float) -> Score:
        scale = self.scale_for(mode, A, B)
        loss = self.C - 2.0 * scale * B + scale * scale * A
        tolerance = 1e-11 * max(self.C, scale * scale * A, 1.0)
        if loss < -tolerance or not np.isfinite(loss):
            raise FloatingPointError(f"Invalid residual loss {loss}")
        return Score(max(float(loss), 0.0), scale, float(A), float(B))

    def score(self, mode: ScaleMode | str, calculated_A2: np.ndarray) -> Score:
        return self.score_from_coefficients(mode, *self.coefficients(calculated_A2))

    def trial_score(
        self,
        mode: ScaleMode | str,
        current_calculated_A2: np.ndarray,
        current: Score,
        delta_A2: np.ndarray,
    ) -> Score:
        x = np.asarray(current_calculated_A2, dtype=float)
        delta = np.asarray(delta_A2, dtype=float)
        A = current.A + 2.0 * float(np.dot(self.weights * x, delta)) + float(
            np.dot(self.weights * delta, delta)
        )
        B = current.B + float(np.dot(self.weights * self.target_A2, delta))
        return self.score_from_coefficients(mode, A, B)

    def common_scores(self, A: float, B: float) -> dict[str, float]:
        result: dict[str, float] = {}
        for mode in ScaleMode:
            score = self.score_from_coefficients(mode, A, B)
            result[f"loss_{mode.value}"] = score.loss
            result[f"Rw_{mode.value}"] = float(np.sqrt(score.loss / self.C))
            result[f"scale_{mode.value}"] = score.scale
        positive_scale = max(B / A, 0.0)
        positive_loss = self.C - 2.0 * positive_scale * B + positive_scale**2 * A
        result["scale_unbounded_positive"] = positive_scale
        result["loss_unbounded_positive"] = max(float(positive_loss), 0.0)
        result["Rw_shape_only"] = float(np.sqrt(result["loss_unbounded_positive"] / self.C))
        return result
