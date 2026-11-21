"""Audit local environments in the original and 2026-11-21 PZS MQ cells."""
import csv,json
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from dpa4_xray_hrmc.analysis.tetrahedra import tetrahedral_metrics
from dpa4_xray_hrmc.rmc.coordination import CoordinationCutoffs,IncrementalCoordination
from dpa4_xray_hrmc.xray_pdf.partial_rdf import compute_partial_rdfs
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"results/PZS_new_20261121/mq_audit"; OUT.mkdir(parents=True,exist_ok=True)
PATHS={"Original MQ":ROOT/"PDF/PZS_quench_222.cif","New MQ":ROOT/"inputs/PZS_new_20261121/PZS_new_20261121.cif"}
PAIRS=(("Zr","Cl"),("Zr","O"),("P","O"),("Si","Cl"),("Si","O")); COLORS={"Original MQ":"#666666","New MQ":"#E41A1C"}
cut=CoordinationCutoffs.from_mapping(json.loads((ROOT/"configs/pzs_phase2_5_coordination_cutoffs.json").read_text()))
atoms={k:read(v) for k,v in PATHS.items()}; rdfs={k:compute_partial_rdfs(v,6.0,.02) for k,v in atoms.items()}; audit={"coordination_cutoffs_A":cut.as_mapping(),"structures":{}}
for name,a in atoms.items():
 cn=IncrementalCoordination(a,cut).current_snapshot(); t=tetrahedral_metrics(a)
 audit["structures"][name]={"path":str(PATHS[name].relative_to(ROOT)),"atoms":len(a),"composition":dict(Counter(a.get_chemical_symbols())),"cell_A":np.asarray(a.cell).tolist(),"volume_A3":a.get_volume(),"density_g_cm3":a.get_masses().sum()*1.66053906660/a.get_volume(),"P_CN_counts":dict(zip(*[x.tolist() for x in np.unique(cn.p_o,return_counts=True)])),"Si_CN_counts":dict(zip(*[x.tolist() for x in np.unique(cn.si_total,return_counts=True)])),"P_fourfold":float(np.mean(cn.p_o==4)),"Si_fourfold":float(np.mean(cn.si_total==4)),"Zr_total_hist":cn.zr_hist_total.tolist(),"Zr_O_hist":cn.zr_hist_o.tolist(),"tetrahedra":{e:{"angle_RMS_deg":t[e]["angle_rms_from_ideal_deg"]["mean"],"CSM_percent":t[e]["fixed_center_csm_percent"]["mean"],"volume_A3":t[e]["volume_A3"]["mean"],"bonds":{p:{q:v[q] for q in ("mean","std","min","max")} for p,v in t[e]["bond_lengths_A"].items()}} for e in ("P","Si")}}
old=audit["structures"]["Original MQ"]; new=audit["structures"]["New MQ"]
audit["changes_new_minus_original"]={"volume_percent":(new["volume_A3"]/old["volume_A3"]-1)*100,"density_percent":(new["density_g_cm3"]/old["density_g_cm3"]-1)*100,"P_fourfold_percentage_points":(new["P_fourfold"]-old["P_fourfold"])*100,"Si_fourfold_percentage_points":(new["Si_fourfold"]-old["Si_fourfold"])*100}
(OUT/"mq_local_environment_audit.json").write_text(json.dumps(audit,indent=2))
def pk(a,b): return tuple(sorted((a,b)))
fig,axes=plt.subplots(3,2,figsize=(13,14),dpi=300); axes=axes.flat
for ax,(central,neighbor) in zip(axes,PAIRS):
 pair=pk(central,neighbor); rows=[]
 for name,rdf in rdfs.items():
  g=rdf.g[pair]; n=np.cumsum(rdf.counts[pair])/rdf.element_counts[central]; rows.append((name,g,n)); ax.plot(rdf.r_A,g,color=COLORS[name],lw=1.7,ls="--" if name=="Original MQ" else "-",label=name)
 ins=inset_axes(ax,width="44%",height="43%",loc="upper right",borderpad=1.0)
 for name,g,n in rows: ins.plot(rdfs[name].r_A,n,color=COLORS[name],lw=1.2,ls="--" if name=="Original MQ" else "-")
 ax.set(xlim=(.8,5),xlabel=r"$r$ ($\AA$)",ylabel=rf"$g_{{{central}-{neighbor}}}(r)$",title=f"{central}-{neighbor}"); ax.grid(True,ls=":",alpha=.4); ax.legend(fontsize=9,loc="upper left")
 ins.set(xlim=(.8,5),xlabel=r"$r$",ylabel=rf"$N_{{{neighbor}\ around\ {central}}}$"); ins.tick_params(labelsize=7); ins.grid(True,ls=":",alpha=.4)
 table=OUT/f"{central}_{neighbor}_rdf_CN.csv"
 with table.open("w",newline="") as f:
  w=csv.writer(f); w.writerow(["r_A","g_original","N_original","g_new","N_new"]); w.writerows(zip(rdfs["Original MQ"].r_A,rows[0][1],rows[0][2],rows[1][1],rows[1][2],strict=True))
axes[-1].axis("off"); fig.suptitle("Original and new PZS MQ local pair environments",fontsize=16); fig.tight_layout(); fig.savefig(OUT/"partial_rdf_and_CN_comparison.png",dpi=300,bbox_inches="tight"); plt.close(fig)
print(json.dumps(audit,indent=2))
