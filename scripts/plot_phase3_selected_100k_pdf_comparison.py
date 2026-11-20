"""Plot experiment, MQ, and selected final Phase 3 100k PDF curves."""
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from dpa4_xray_hrmc.rmc.config import load_phase2_config
from dpa4_xray_hrmc.rmc.objectives import ScaleMode,ScaleObjective
from dpa4_xray_hrmc.rmc.runner import inclusive_grid
from dpa4_xray_hrmc.xray_pdf.experimental import read_pdfgetx_gr
from dpa4_xray_hrmc.xray_pdf.form_factors import q_dependent_weights
from dpa4_xray_hrmc.xray_pdf.partial_rdf import compute_partial_rdfs
from dpa4_xray_hrmc.xray_pdf.transform import forward_transform
ROOT=Path(__file__).resolve().parents[1]
CFG=load_phase2_config(ROOT/"configs/pzs_phase2_common.yaml")
OUT=ROOT/"results/PZS_phase3/calibration"
ALT=ROOT/"results/PZS_phase3/phase3B/validation_100k_seed303_A500_F10/final_structure.extxyz"
INLINE=ROOT/"results/PZS_phase3/phase3C/validation_100k_seed303_T100/final_structure.extxyz"
EXP=read_pdfgetx_gr(ROOT/CFG.experimental_gr)
Q=inclusive_grid(CFG.qmin_Ainv,CFG.qmax_Ainv,CFG.qstep_Ainv)
FIT=(EXP.r_A>=CFG.fit_rmin_A)&(EXP.r_A<=CFG.fit_rmax_A)
def curve(atoms):
    rdf=compute_partial_rdfs(atoms,CFG.rdf_rcut_A,CFG.rdf_dr_A)
    w=q_dependent_weights(rdf.element_counts,Q)
    return forward_transform(rdf,w,EXP.r_A,CFG.biso_A2,CFG.lorch_mod).g_r_A2
curves={
    "MQ":curve(read(ROOT/CFG.structure)),
    "Alternating 100k, seed 303 final":curve(read(ALT)),
    "Inline 100k, seed 303 final":curve(read(INLINE)),
}
obj=ScaleObjective.from_initial(curves["MQ"][FIT],EXP.g_r_A2[FIT])
metrics={}
for label,g in curves.items():
    score=obj.score(ScaleMode.NO_SCALE,g[FIT])
    metrics[label]={"Rw_no_scale_1p5_10A":obj.common_scores(score.A,score.B)["Rw_no_scale"],"loss_sum_A-4":score.loss,"scale_applied":score.scale}
OUT.mkdir(parents=True,exist_ok=True)
with (OUT/"selected_final_100k_pdf_comparison.csv").open("w",newline="",encoding="utf-8") as h:
    w=csv.writer(h);w.writerow(["r_A","G_experiment_A-2","G_MQ_A-2","G_alternating_seed303_final_A-2","G_inline_seed303_final_A-2"])
    w.writerows(zip(EXP.r_A,EXP.g_r_A2,curves["MQ"],curves["Alternating 100k, seed 303 final"],curves["Inline 100k, seed 303 final"],strict=True))
(OUT/"selected_final_100k_pdf_comparison_metrics.json").write_text(json.dumps(metrics,indent=2))
fig,ax=plt.subplots(figsize=(11,5),dpi=300)
ax.plot(EXP.r_A,EXP.g_r_A2,color="black",lw=2,label="Experiment",zorder=4)
styles={"MQ":("#666666","--",1.5),"Alternating 100k, seed 303 final":("#15572C","-",1.8),"Inline 100k, seed 303 final":("#FF0025","-",1.8)}
for label,g in curves.items():
    color,ls,lw=styles[label];rw=metrics[label]["Rw_no_scale_1p5_10A"]
    ax.plot(EXP.r_A,g,color=color,ls=ls,lw=lw,label=rf"{label} ($R_w={rw:.3f}$)")
ax.set_xlim(CFG.fit_rmin_A,CFG.fit_rmax_A)
v=np.concatenate([EXP.g_r_A2[FIT],*(g[FIT] for g in curves.values())]);p=.06*(v.max()-v.min())
ax.set_ylim(v.min()-p,v.max()+p)
ax.set_xlabel(r"$r$ ($\AA$)",fontsize=14);ax.set_ylabel(r"$G(r)$ ($\AA^{-2}$)",fontsize=14)
ax.set_title("Experimental and selected final 100k PDFs (no scale)",fontsize=14,pad=10)
ax.tick_params(direction="in",labelsize=14);ax.grid(True,ls=":",alpha=.5);ax.legend(loc="upper right",frameon=True,fontsize=11)
fig.tight_layout();plot=OUT/"plots/selected_final_100k_pdf_comparison_1p5_10A.png";plot.parent.mkdir(exist_ok=True);fig.savefig(plot,dpi=300,bbox_inches="tight");plt.close(fig)
print(json.dumps({"plot":str(plot),"metrics":metrics},indent=2))
