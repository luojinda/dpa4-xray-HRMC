"""Uniform analysis of four new-PZS Inline 100k model/scale combinations."""
import csv,json,os
from dataclasses import replace
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from dpa4_xray_hrmc.analysis.phase3_structure import Phase3StructureEvaluator
from dpa4_xray_hrmc.mlip.dpa4_backend import DPA4Backend
from dpa4_xray_hrmc.rmc.config import load_phase2_config
from dpa4_xray_hrmc.rmc.objectives import ScaleMode,ScaleObjective
from dpa4_xray_hrmc.rmc.runner import inclusive_grid
from dpa4_xray_hrmc.xray_pdf.experimental import read_pdfgetx_gr
from dpa4_xray_hrmc.xray_pdf.form_factors import q_dependent_weights
from dpa4_xray_hrmc.xray_pdf.partial_rdf import compute_partial_rdfs
from dpa4_xray_hrmc.xray_pdf.transform import forward_transform
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/"results/PZS_phase3/phase3C/PZS_new_20261121"
RUNS={k:BASE/f"validation_100k_seed101_T100_d0p1_r1p4_10p0A_{k}" for k in ("lora_no_scale","lora_bounded_dynamic","frozen_no_scale","frozen_bounded_dynamic")}
MODELS={"lora":ROOT/"models/PZS_new_20261121/DPA4_PZS_new_LoRA.pt2","frozen":ROOT/"models/PZS_new_20261121/DPA4-Neo-OMat24-v20260627_frozen.pt2"}
OUT=ROOT/"results/PZS_new_20261121/four_inline_100k_comparison"; OUT.mkdir(parents=True,exist_ok=True)
cfg=replace(load_phase2_config(ROOT/"configs/pzs_phase2_common.yaml"),structure="inputs/PZS_new_20261121/PZS_new_20261121.cif",fit_rmin_A=1.4,fit_rmax_A=10.0,move_max_A=.1)
exp=read_pdfgetx_gr(ROOT/cfg.experimental_gr); fit=(exp.r_A>=1.4)&(exp.r_A<=10); q=inclusive_grid(cfg.qmin_Ainv,cfg.qmax_Ainv,cfg.qstep_Ainv)
def curve(a):
 r=compute_partial_rdfs(a,cfg.rdf_rcut_A,cfg.rdf_dr_A); return forward_transform(r,q_dependent_weights(r.element_counts,q),exp.r_A,cfg.biso_A2,cfg.lorch_mod).g_r_A2
atoms={"new_MQ":read(ROOT/cfg.structure),**{k:read(v/"final_structure.extxyz") for k,v in RUNS.items()}}
curves={k:curve(v) for k,v in atoms.items()}; obj=ScaleObjective.from_initial(curves["new_MQ"][fit],exp.g_r_A2[fit])
modes={"new_MQ":ScaleMode.BOUNDED_DYNAMIC,**{k:ScaleMode.BOUNDED_DYNAMIC if "bounded" in k else ScaleMode.NO_SCALE for k in RUNS}}
scores={k:obj.score(modes[k],v[fit]) for k,v in curves.items()}; summaries={k:json.loads((v/"summary.json").read_text()) for k,v in RUNS.items()}
evaluator=Phase3StructureEvaluator(ROOT,cfg,ROOT/"configs/pzs_phase2_5_coordination_cutoffs.json")
diagnostics={k:evaluator.evaluate(v) for k,v in atoms.items()}
for v in diagnostics.values(): v.pop("g_r_A2",None)
metrics={"fit_range_A":[1.4,10.0],"scale_initial":obj.s0,"scale_bounds":[obj.lower_scale,obj.upper_scale],"runs":{},"diagnostics":diagnostics,"cross_model_energy":{}}
for k,s in scores.items():
 common=obj.common_scores(s.A,s.B); metrics["runs"][k]={**common,"Rw_active":float(np.sqrt(s.loss/obj.C)),"scale_active":s.scale,"scale_raw":s.B/s.A}
for k,s in summaries.items(): metrics["runs"][k].update({x:s[x] for x in ("acceptance_rate","energy_eV_atom","delta_energy_from_MQ_eV","P_fourfold","Si_fourfold","hard_sphere_rejected","wall_s","model_sha256")})
for model,path in MODELS.items():
 b=DPA4Backend(path,required_elements=set(atoms["new_MQ"].get_chemical_symbols())); e0=b.get_energy(atoms["new_MQ"]); metrics["cross_model_energy"][model]={k:{"energy_eV":(e:=b.get_energy(a)),"delta_vs_new_MQ_meV_atom":(e-e0)*1000/len(a)} for k,a in atoms.items()}
(OUT/"metrics.json").write_text(json.dumps(metrics,indent=2))
with (OUT/"pdf_curves.csv").open("w",newline="") as f:
 w=csv.writer(f); w.writerow(["r_A","G_experiment",*[f"G_{k}" for k in curves]]); w.writerows(zip(exp.r_A,exp.g_r_A2,*curves.values(),strict=True))
rows=[]
for k,p in RUNS.items():
 tr=json.loads((p/"trace.json").read_text()); s=summaries[k]
 if not tr or int(tr[-1]["attempted"])!=100000: tr.append(s)
 rows.append({"run":k,"attempts":0,"Rw_active":metrics["runs"]["new_MQ"]["Rw_active"] if "bounded" in k else obj.common_scores(obj.score(ScaleMode.NO_SCALE,curves["new_MQ"][fit]).A,obj.score(ScaleMode.NO_SCALE,curves["new_MQ"][fit]).B)["Rw_no_scale"],"scale":obj.s0 if "bounded" in k else 1.0,"delta_energy_meV_atom":0.0})
 for x in tr: rows.append({"run":k,"attempts":int(x["attempted"]),"Rw_active":x["Rw_active"],"scale":x["scale_current"],"delta_energy_meV_atom":(x["energy_eV"]-(s["energy_eV"]-s["delta_energy_from_MQ_eV"]))*1000/len(atoms[k])})
with (OUT/"trajectories.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
colors={"new_MQ":"#555555","lora_no_scale":"#377EB8","lora_bounded_dynamic":"#984EA3","frozen_no_scale":"#4DAF4A","frozen_bounded_dynamic":"#E41A1C"}
labels={"new_MQ":"New MQ","lora_no_scale":"LoRA no scale","lora_bounded_dynamic":"LoRA dynamic","frozen_no_scale":"Frozen no scale","frozen_bounded_dynamic":"Frozen dynamic"}
fig,axes=plt.subplots(2,1,figsize=(12,10),dpi=300,sharex=True)
for ax,active in zip(axes,(False,True),strict=True):
 ax.plot(exp.r_A,exp.g_r_A2,color="black",lw=2,label="Experiment")
 for k,g in curves.items():
  scale=scores[k].scale if active else 1.0; rw=metrics["runs"][k]["Rw_active" if active else "Rw_no_scale"]
  ax.plot(exp.r_A,scale*g,color=colors[k],lw=1.5,ls="--" if k=="new_MQ" else "-",label=f"{labels[k]} (Rw={rw:.3f}, w={scale:.3f})")
 ax.set(xlim=(1.4,10),ylabel=r"$G(r)$ ($\AA^{-2}$)",title="Active strategy scale" if active else "Common no-scale diagnostic"); ax.grid(True,ls=":",alpha=.45); ax.legend(fontsize=8,ncol=2)
axes[-1].set_xlabel(r"$r$ ($\AA$)"); fig.tight_layout(); fig.savefig(OUT/"pdf_four_runs_1p4_10A.png",dpi=300,bbox_inches="tight"); plt.close(fig)
fig,axes=plt.subplots(3,1,figsize=(12,12),dpi=300,sharex=True)
for k in RUNS:
 r=[x for x in rows if x["run"]==k]; a=[x["attempts"] for x in r]
 axes[0].plot(a,[x["Rw_active"] for x in r],color=colors[k],lw=1.5,label=labels[k]); axes[1].plot(a,[x["delta_energy_meV_atom"] for x in r],color=colors[k],lw=1.5); axes[2].plot(a,[x["scale"] for x in r],color=colors[k],lw=1.5)
axes[0].set_ylabel("Active Rw"); axes[1].set_ylabel("Delta E (meV/atom)"); axes[2].set_ylabel("Scale w"); axes[2].set_xlabel("MC attempts")
axes[2].axhspan(obj.lower_scale,obj.upper_scale,color="#999999",alpha=.12); axes[0].legend(fontsize=9,ncol=2)
for ax in axes: ax.grid(True,ls=":",alpha=.45)
fig.tight_layout(); fig.savefig(OUT/"trajectories_Rw_energy_scale.png",dpi=300,bbox_inches="tight"); plt.close(fig)
geometry=[]
for k,d in diagnostics.items(): geometry.append({"structure":k,"P_angle_RMS_deg":d["tetrahedra"]["P"]["angle_rms_from_ideal_deg"]["mean"],"Si_angle_RMS_deg":d["tetrahedra"]["Si"]["angle_rms_from_ideal_deg"]["mean"],"P_CSM_percent":d["tetrahedra"]["P"]["fixed_center_csm_percent"]["mean"],"Si_CSM_percent":d["tetrahedra"]["Si"]["fixed_center_csm_percent"]["mean"]})
with (OUT/"tetrahedral_geometry.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(geometry[0])); w.writeheader(); w.writerows(geometry)
fig,axes=plt.subplots(2,2,figsize=(12,9),dpi=300); keys=[("P_angle_RMS_deg","P angle RMS (deg)"),("Si_angle_RMS_deg","Si angle RMS (deg)"),("P_CSM_percent","P CSM (%)"),("Si_CSM_percent","Si CSM (%)")]
for ax,(key,title) in zip(axes.flat,keys,strict=True):
 bars=ax.bar([labels[x["structure"]] for x in geometry],[x[key] for x in geometry],color=[colors[x["structure"]] for x in geometry]); ax.set_title(title); ax.tick_params(axis="x",rotation=20); ax.bar_label(bars,fmt="%.3f",fontsize=8); ax.grid(True,axis="y",ls=":",alpha=.45)
fig.tight_layout(); fig.savefig(OUT/"tetrahedral_geometry.png",dpi=300,bbox_inches="tight"); plt.close(fig)
print(json.dumps(metrics,indent=2))
