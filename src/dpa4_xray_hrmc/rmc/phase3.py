"""Phase 3 alternating and inline kernels built on Phase 2.5 state engines."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from ase import Atoms

from .coordination import CoordinationComponents, IncrementalCoordination
from .hard_sphere import HardSphereModel
from .incremental_pdf import IncrementalXrayPDF
from .objectives import ScaleMode, ScaleObjective, Score

KB_EV_K = 8.617333262145e-5


def phase25_total(score: Score, cn: CoordinationComponents, weights, sigma: float) -> float:
    return float(score.loss / (2.0 * sigma * sigma) + cn.weighted(weights))


def metropolis_accept(delta_loss: float, random_u: float) -> tuple[bool, float]:
    """Stable Metropolis test and its probability, also returned for auditing."""
    if delta_loss <= 0.0:
        return True, 1.0
    probability = math.exp(-delta_loss) if delta_loss < 745.0 else 0.0
    return math.log(random_u) < -delta_loss, probability


def inline_delta(delta_phase25: float, delta_energy_eV: float, temperature_K: float) -> float:
    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    return float(delta_phase25 + delta_energy_eV / (KB_EV_K * temperature_K))


@dataclass
class Phase3State:
    pdf: IncrementalXrayPDF
    cn: IncrementalCoordination
    hard_sphere: HardSphereModel
    objective: ScaleObjective
    weights: dict[str, float]
    sigma: float
    score: Score
    total: float

    @classmethod
    def build(cls, atoms, q, r_fit, g_exp_fit, config, cutoffs, weights,
              hard_sphere=None, objective=None):
        pdf = IncrementalXrayPDF(atoms, q, r_fit, config.rdf_rcut_A,
                                 config.rdf_dr_A, config.biso_A2, config.lorch_mod)
        obj = objective or ScaleObjective.from_initial(pdf.g_r_A2, g_exp_fit)
        cn = IncrementalCoordination(atoms, cutoffs)
        hs = hard_sphere or HardSphereModel.from_atoms(atoms, config.hard_sphere_scale)
        score = obj.score(ScaleMode.NO_SCALE, pdf.g_r_A2)
        total = phase25_total(score, cn.components, weights, config.sigma_rmc_A2)
        return cls(pdf, cn, hs, obj, dict(weights), config.sigma_rmc_A2, score, total)

    @property
    def atoms(self) -> Atoms:
        return self.pdf.atoms

    def propose(self, atom_index: int, new_position_A):
        valid, margin, pair = self.hard_sphere.check_move(
            self.atoms, atom_index, new_position_A
        )
        if not valid:
            return None, {"hard_sphere_margin_A": margin, "limiting_pair": pair}
        pp = self.pdf.propose_move(atom_index, new_position_A)
        cp = self.cn.propose_move(atom_index, new_position_A)
        trial = self.objective.trial_score(
            ScaleMode.NO_SCALE, self.pdf.g_r_A2, self.score, pp.delta_g_A2
        )
        total = phase25_total(trial, cp.trial.components, self.weights, self.sigma)
        return (pp, cp, trial, total), {"delta_phase25": total - self.total}

    def commit(self, proposal) -> None:
        pp, cp, trial, total = proposal
        self.pdf.commit(pp)
        self.cn.commit(cp)
        self.score, self.total = trial, total

    def reject(self, proposal) -> None:
        pp, cp, _, _ = proposal
        self.pdf.reject(pp)
        self.cn.reject(cp)

    def rebuilt(self, atoms, q, r_fit, g_exp_fit, config, cutoffs):
        return self.build(
            atoms, q, r_fit, g_exp_fit, config, cutoffs, self.weights,
            hard_sphere=self.hard_sphere, objective=self.objective
        )

