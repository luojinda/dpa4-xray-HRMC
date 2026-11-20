#!/usr/bin/env python
from __future__ import annotations

import csv,json,os
from pathlib import Path

import numpy as np
from ase.io import read,write

from dpa4_xray_hrmc.analysis.phase3_structure import Phase3StructureEvaluator
from dpa4_xray_hrmc.mlip.dpa4_backend import DPA4Backend
from dpa4_xray_hrmc.mlip.fire_relaxation import run_fixed_cell_fire
from dpa4_xray_hrmc.rmc.config import load_phase2_config
from dpa4_xray_hrmc.rmc.hard_sphere import minimum_image_vectors


def compact(label,source,fire,evaluation,initial_atoms):
    displacement=np.linalg.norm(minimum_image_vectors(fire.atoms.positions-initial_atoms.positions,np.asarray(initial_atoms.cell)),axis=1)
    p=evaluation['tetrahedra']['P']; si=evaluation['tetrahedra']['Si']
    return {"label":label,"source":source,"requested_steps":fire.requested_steps,"executed_steps":fire.executed_steps,"converged":fire.converged,"initial_energy_eV":fire.initial_energy_eV,"final_energy_eV":fire.final_energy_eV,"delta_energy_eV":fire.final_energy_eV-fire.initial_energy_eV,"final_energy_eV_atom":fire.final_energy_eV/len(fire.atoms),"initial_force_rms_eV_A":fire.initial_force_rms_eV_A,"final_force_rms_eV_A":fire.final_force_rms_eV_A,"initial_force_max_eV_A":fire.initial_force_max_eV_A,"final_force_max_eV_A":fire.final_force_max_eV_A,"wall_s":fire.wall_s,"Rw_no_scale":evaluation['pdf']['Rw_no_scale'],"pdf_loss_sum":evaluation['pdf']['loss_sum'],"P_fourfold":evaluation['coordination']['p_fourfold_fraction'],"Si_fourfold":evaluation['coordination']['si_fourfold_fraction'],"P_angle_rms_deg":p['angle_rms_from_ideal_deg']['mean'],"Si_angle_rms_deg":si['angle_rms_from_ideal_deg']['mean'],"P_fixed_center_CSM":p['fixed_center_csm_percent']['mean'],"Si_fixed_center_CSM":si['fixed_center_csm_percent']['mean'],"P_volume_A3":p['volume_A3']['mean'],"Si_volume_A3":si['volume_A3']['mean'],"hard_sphere_violations":evaluation['hard_sphere']['violation_total'],"rms_displacement_A":float(np.sqrt(np.mean(displacement**2))),"max_displacement_A":float(np.max(displacement))}


def main():
    root=Path(__file__).resolve().parents[1]; out=root/'results/PZS_phase3/phase3A'; structures=out/'structures'; plots=out/'plots'; structures.mkdir(parents=True,exist_ok=True);plots.mkdir(exist_ok=True)
    config=load_phase2_config(root/'configs/pzs_phase2_common.yaml'); evaluator=Phase3StructureEvaluator(root,config,root/'configs/pzs_phase2_5_coordination_cutoffs.json'); backend=DPA4Backend(os.environ['DPA4_MODEL_PATH'],required_elements=('Cl','Li','O','P','Si','Zr'))
    sources=[('MQ',root/'PDF/PZS_quench_222.cif'),('phase2_5_no_scale_seed_101',root/'results/PZS_phase2_5/formal_1M/no_scale/seed_101/final_structure.extxyz')]
    rows=[]; details={}
    for source_label,path in sources:
        initial=read(path); initial_eval=evaluator.evaluate(initial); initial_energy=backend.evaluate(initial); key=f'{source_label}_initial'
        fake=type('Initial',(),dict(atoms=initial,requested_steps=0,executed_steps=0,converged=False,initial_energy_eV=initial_energy.energy_eV,final_energy_eV=initial_energy.energy_eV,initial_force_rms_eV_A=float(np.sqrt(np.mean(np.sum(initial_energy.forces_eV_A**2,axis=1)))),final_force_rms_eV_A=float(np.sqrt(np.mean(np.sum(initial_energy.forces_eV_A**2,axis=1)))),initial_force_max_eV_A=float(np.linalg.norm(initial_energy.forces_eV_A,axis=1).max()),final_force_max_eV_A=float(np.linalg.norm(initial_energy.forces_eV_A,axis=1).max()),wall_s=0.0))
        rows.append(compact(key,str(path),fake,initial_eval,initial)); details[key]={k:v for k,v in initial_eval.items() if k!='g_r_A2'}
        for steps in (5,10,20):
            label=f'{source_label}_fire_{steps}'; fire=run_fixed_cell_fire(initial,backend,steps,fmax_eV_A=.05,maxstep_A=.05,dt_fs_like=.05,logfile=str(out/f'{label}.log')); evaluation=evaluator.evaluate(fire.atoms)
            write(structures/f'{label}.extxyz',fire.atoms); write(structures/f'{label}.cif',fire.atoms)
            np.savetxt(out/f'{label}_calc_Gr.dat',np.column_stack((evaluator.experiment.r_A,evaluation['g_r_A2'],evaluator.experiment.g_r_A2)),header='r_A Gcalc_A-2 Gexp_A-2')
            rows.append(compact(label,str(path),fire,evaluation,initial)); details[label]={k:v for k,v in evaluation.items() if k!='g_r_A2'}
    with (out/'fire_comparison.csv').open('w',newline='') as h: w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (out/'fire_details.json').write_text(json.dumps(details,indent=2),encoding='utf-8'); print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
