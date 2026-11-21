"""Compare 1.4--10 A fixed and bounded-dynamic Inline HRMC runs."""
import csv, json
from dataclasses import replace
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from dpa4_xray_hrmc.analysis.tetrahedra import tetrahedral_metrics
from dpa4_xray_hrmc.rmc.config import load_phase2_config
from dpa4_xray_hrmc.rmc.objectives import ScaleMode, ScaleObjective
from dpa4_xray_hrmc.rmc.runner import inclusive_grid
from dpa4_xray_hrmc.xray_pdf.experimental import read_pdfgetx_gr
from dpa4_xray_hrmc.xray_pdf.form_factors import q_dependent_weights
from dpa4_xray_hrmc.xray_pdf.partial_rdf import compute_partial_rdfs
from dpa4_xray_hrmc.xray_pdf.transform import forward_transform
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/"results/PZS_phase3/phase3C"
RUNS={"fixed_initial":BASE/"validation_100k_seed101_T100_d0p1_r1p4_10p0A_fixed_initial","bounded_dynamic":BASE/"validation_100k_seed101_T100_d0p1_r1p4_10p0A_bounded_dynamic"}
OUT=BASE/"scale_strategy_1p4_10_comparison"; OUT.mkdir(parents=True,exist_ok=True)
cfg=replace(load_phase2_config(ROOT/"configs/pzs_phase2_common.yaml"),fit_rmin_A=1.4,fit_rmax_A=10.0,move_max_A=0.1)
exp=read_pdfgetx_gr(ROOT/cfg.experimental_gr); fit=(exp.r_A>=1.4)&(exp.r_A<=10.0); q=inclusive_grid(cfg.qmin_Ainv,cfg.qmax_Ainv,cfg.qstep_Ainv)
def curve(a):
    rdf=compute_partial_rdfs(a,cfg.rdf_rcut_A,cfg.rdf_dr_A)
    return forward_transform(rdf,q_dependent_weights(rdf.element_counts,q),exp.r_A,cfg.biso_A2,cfg.lorch_mod).g_r_A2
atoms={"MQ":read(ROOT/cfg.structure),**{k:read(v/"final_structure.extxyz") for k,v in RUNS.items()}}
curves={k:curve(v) for k,v in atoms.items()}; obj=ScaleObjective.from_initial(curves["MQ"][fit],exp.g_r_A2[fit])
modes={"MQ":ScaleMode.FIXED_INITIAL,"fixed_initial":ScaleMode.FIXED_INITIAL,"bounded_dynamic":ScaleMode.BOUNDED_DYNAMIC}
scores={k:obj.score(modes[k],v[fit]) for k,v in curves.items()}; summaries={k:json.loads((v/"summary.json").read_text()) for k,v in RUNS.items()}
metrics={"fit_range_A":[1.4,10.0],"scale_initial":obj.s0,"scale_bounds":[obj.lower_scale,obj.upper_scale],"structures":{}}
for k,s in scores.items(): metrics["structures"][k]={**obj.common_scores(s.A,s.B),"Rw_active":float(np.sqrt(s.loss/obj.C)),"scale_active":s.scale,"scale_raw":s.B/s.A}
for k,s in summaries.items(): metrics["structures"][k].update({"acceptance_rate":s["acceptance_rate"],"delta_energy_meV_atom":s["delta_energy_from_MQ_eV"]*1000/len(atoms[k]),"wall_s":s["wall_s"],"hard_sphere_rejected":s["hard_sphere_rejected"],"P_fourfold":s["P_fourfold"],"Si_fourfold":s["Si_fourfold"],"energy_drift_eV":s["energy_drift_eV"]})
geo=[]
for k,a in atoms.items():
    t=tetrahedral_metrics(a); geo.append({"structure":k,"P_angle_RMS_deg":t["P"]["angle_rms_from_ideal_deg"]["mean"],"Si_angle_RMS_deg":t["Si"]["angle_rms_from_ideal_deg"]["mean"],"P_CSM_percent":t["P"]["fixed_center_csm_percent"]["mean"],"Si_CSM_percent":t["Si"]["fixed_center_csm_percent"]["mean"]})
metrics["tetrahedral_geometry"]=geo; (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2))
with (OUT/"tetrahedral_geometry.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(geo[0])); w.writeheader(); w.writerows(geo)
with (OUT/"pdf_curves.csv").open("w",newline="") as f:
    w=csv.writer(f); w.writerow(["r_A","G_experiment","G_MQ_scaled","G_fixed_scaled","G_dynamic_scaled"])
    w.writerows(zip(exp.r_A,exp.g_r_A2,scores["MQ"].scale*curves["MQ"],scores["fixed_initial"].scale*curves["fixed_initial"],scores["bounded_dynamic"].scale*curves["bounded_dynamic"],strict=True))
rows=[]; mq_e=summaries["fixed_initial"]["energy_eV"]-summaries["fixed_initial"]["delta_energy_from_MQ_eV"]
for k,p in RUNS.items():
    rows.append({"strategy":k,"attempts":0,"Rw_active":metrics["structures"]["MQ"]["Rw_active"],"scale":obj.s0,"delta_energy_meV_atom":0.0})
    trace=json.loads((p/"trace.json").read_text())
    if int(trace[-1]["attempted"])!=100000: trace.append(summaries[k])
    for x in trace: rows.append({"strategy":k,"attempts":int(x["attempted"]),"Rw_active":x["Rw_active"],"scale":x["scale_current"],"delta_energy_meV_atom":(x["energy_eV"]-mq_e)*1000/len(atoms[k])})
with (OUT/"trajectories.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
colors={"MQ":"#666666","fixed_initial":"#15572C","bounded_dynamic":"#FF0025"}; labels={"MQ":"MQ","fixed_initial":"Fixed initial","bounded_dynamic":"Bounded dynamic"}
fig,ax=plt.subplots(figsize=(11,5),dpi=300); ax.plot(exp.r_A,exp.g_r_A2,color="black",lw=2,label="Experiment")
for k in ("MQ","fixed_initial","bounded_dynamic"):
    s=scores[k]; ax.plot(exp.r_A,s.scale*curves[k],color=colors[k],lw=1.8,ls="--" if k=="MQ" else "-",label=rf"{labels[k]} ($R_w={metrics['structures'][k]['Rw_active']:.3f}$, $w={s.scale:.3f}$)")
ax.set(xlim=(1.4,10),xlabel=r"$r$ ($\AA$)",ylabel=r"$G(r)$ ($\AA^{-2}$)",title="Inline HRMC scale strategies, 100k attempts (1.4--10 A)")
ax.tick_params(direction="in",labelsize=13); ax.grid(True,ls=":",alpha=.5); ax.legend(fontsize=10); fig.tight_layout(); fig.savefig(OUT/"pdf_comparison_1p4_10A.png",dpi=300,bbox_inches="tight"); plt.close(fig)
fig,(top,bottom)=plt.subplots(2,1,figsize=(11,8),dpi=300,sharex=True); right=top.twinx()
for k in RUNS:
    r=[x for x in rows if x["strategy"]==k]; a=np.asarray([x["attempts"] for x in r])
    top.plot(a,[x["Rw_active"] for x in r],color=colors[k],lw=1.8,label=f"{labels[k]} Rw")
    right.plot(a,[x["delta_energy_meV_atom"] for x in r],color=colors[k],lw=1.2,ls="--",label=f"{labels[k]} dE")
    bottom.plot(a,[x["scale"] for x in r],color=colors[k],lw=1.8,label=labels[k])
top.set_ylabel("Active Rw"); right.set_ylabel("Delta E (meV/atom)"); top.grid(True,ls=":",alpha=.5)
h1,l1=top.get_legend_handles_labels(); h2,l2=right.get_legend_handles_labels(); top.legend(h1+h2,l1+l2,ncol=2,fontsize=9)
bottom.axhspan(obj.lower_scale,obj.upper_scale,color="#999999",alpha=.12,label="Allowed range"); bottom.axhline(obj.s0,color="#555555",ls=":",label="Initial scale")
bottom.set(xlabel="MC attempts",ylabel="Scale w"); bottom.grid(True,ls=":",alpha=.5); bottom.legend(ncol=2,fontsize=9)
fig.tight_layout(); fig.savefig(OUT/"trajectory_Rw_energy_scale.png",dpi=300,bbox_inches="tight"); plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(11,9),dpi=300)
keys=[("P_angle_RMS_deg","P angle RMS (deg)"),("Si_angle_RMS_deg","Si angle RMS (deg)"),("P_CSM_percent","P CSM (%)"),("Si_CSM_percent","Si CSM (%)")]
for ax,(key,title) in zip(axes.flat,keys,strict=True):
    bars=ax.bar([labels[x["structure"]] for x in geo],[x[key] for x in geo],color=[colors[x["structure"]] for x in geo]); ax.set_title(title); ax.bar_label(bars,fmt="%.3f"); ax.grid(True,axis="y",ls=":",alpha=.5)
fig.tight_layout(); fig.savefig(OUT/"tetrahedral_geometry.png",dpi=300,bbox_inches="tight"); plt.close(fig)
print(json.dumps(metrics,indent=2))
