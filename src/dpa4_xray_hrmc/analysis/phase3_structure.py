"""Common Phase 3 PDF, coordination, hard-sphere, and geometry evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

from dpa4_xray_hrmc.rmc.config import Phase2Config
from dpa4_xray_hrmc.rmc.coordination import CoordinationCutoffs, IncrementalCoordination
from dpa4_xray_hrmc.rmc.hard_sphere import HardSphereModel, minimum_image_vectors
from dpa4_xray_hrmc.rmc.objectives import ScaleMode, ScaleObjective
from dpa4_xray_hrmc.rmc.runner import inclusive_grid
from dpa4_xray_hrmc.xray_pdf.experimental import read_pdfgetx_gr
from dpa4_xray_hrmc.xray_pdf.form_factors import q_dependent_weights
from dpa4_xray_hrmc.xray_pdf.partial_rdf import compute_partial_rdfs
from dpa4_xray_hrmc.xray_pdf.transform import forward_transform

from .tetrahedra import tetrahedral_metrics


class Phase3StructureEvaluator:
    def __init__(self, root: Path, config: Phase2Config, cutoffs_path: Path) -> None:
        self.root=root; self.config=config
        self.experiment=read_pdfgetx_gr(root/config.experimental_gr)
        self.fit=(self.experiment.r_A>=config.fit_rmin_A)&(self.experiment.r_A<=config.fit_rmax_A)
        self.q=inclusive_grid(config.qmin_Ainv,config.qmax_Ainv,config.qstep_Ainv)
        initial_atoms = read(root / config.structure)
        initial = self._curve_only(initial_atoms)
        self.objective=ScaleObjective.from_initial(initial[self.fit],self.experiment.g_r_A2[self.fit])
        self.cutoffs=CoordinationCutoffs.from_mapping(json.loads(cutoffs_path.read_text()))
        # Phase 3 freezes the Phase 2.5 hard-sphere model.  The cutoffs are
        # derived once from the original MQ structure and must not be relaxed
        # implicitly when a trial/relaxed structure is evaluated.
        self.hard_sphere = HardSphereModel.from_atoms(
            initial_atoms, self.config.hard_sphere_scale
        )

    def _curve_only(self, atoms: Atoms) -> np.ndarray:
        rdf=compute_partial_rdfs(atoms,self.config.rdf_rcut_A,self.config.rdf_dr_A)
        weights=q_dependent_weights(rdf.element_counts,self.q)
        return forward_transform(rdf,weights,self.experiment.r_A,self.config.biso_A2,self.config.lorch_mod).g_r_A2

    def evaluate(self, atoms: Atoms) -> dict:
        curve=self._curve_only(atoms); score=self.objective.score(ScaleMode.NO_SCALE,curve[self.fit])
        cn=IncrementalCoordination(atoms,self.cutoffs).current_snapshot()
        hs = self.hard_sphere
        symbols=np.asarray(atoms.get_chemical_symbols()); minima={pair:np.inf for pair in hs.pairs}
        for index in range(len(atoms)-1):
            vectors=minimum_image_vectors(atoms.positions[index+1:]-atoms.positions[index],np.asarray(atoms.cell))
            for other,distance in enumerate(np.linalg.norm(vectors,axis=1),start=index+1):
                pair=tuple(sorted((str(symbols[index]),str(symbols[other]))))
                minima[pair]=min(minima[pair],float(distance))
        violations={f'{a}-{b}':int(minima[(a,b)]<hs.cutoffs_A[(a,b)]-1e-10) for a,b in hs.pairs}
        return {"pdf":{"loss_sum":score.loss,"Rw_no_scale":self.objective.common_scores(score.A,score.B)["Rw_no_scale"],"scale":score.scale},"coordination":{"components":cn.components.as_dict(),"p_fourfold_fraction":float(np.mean(cn.p_o==4)),"si_fourfold_fraction":float(np.mean(cn.si_total==4)),"zr_hist_total":cn.zr_hist_total.tolist(),"zr_hist_o":cn.zr_hist_o.tolist()},"hard_sphere":{"violation_total":sum(violations.values()),"violations_by_pair":violations},"tetrahedra":tetrahedral_metrics(atoms),"g_r_A2":curve}
