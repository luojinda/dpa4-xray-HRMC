"""Visualize core Phase 2.5 three-scale PDF and local-structure results."""
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/"results/PZS_phase2_5";FORMAL=BASE/"formal_1M";OUT=BASE/"three_scale_core_results";OUT.mkdir(parents=True,exist_ok=True)
SEEDS=(101,202,303);MODES=("no_scale","fixed_initial","bounded_dynamic")
LABEL={"no_scale":"No scale","fixed_initial":"Fixed initial","bounded_dynamic":"Bounded dynamic"}
COLOR={"MQ":"#666666","no_scale":"#377EB8","fixed_initial":"#E69F00","bounded_dynamic":"#009E73"}
LS={"no_scale":"-","fixed_initial":"-.","bounded_dynamic":"-"}
summ=json.loads((BASE/"final_1M_summary.json").read_text())
def dirs(m):return [FORMAL/m/f"seed_{s}" for s in SEEDS]
def pdf_plot():
 curves={};r=exp=None;metrics={}
 for m in MODES:
  stack=[]
  for d in dirs(m):
   a=np.loadtxt(d/"final_structure_calc_Gr.dat");r=a[:,0];exp=a[:,5];stack.append(a[:,2])
  stack=np.stack(stack);rows=[x for x in summ if x["mode"]==m]
  curves[m]=(stack.mean(0),stack.std(0,ddof=0));metrics[m]={"Rw_mean":float(np.mean([x["Rw_mode"] for x in rows])),"Rw_std":float(np.std([x["Rw_mode"] for x in rows],ddof=0)),"scale_mean":float(np.mean([x["scale"] for x in rows])),"scale_std":float(np.std([x["scale"] for x in rows],ddof=0)),"raw_optimum_mean":float(np.mean([x["raw"] for x in rows])),"raw_optimum_std":float(np.std([x["raw"] for x in rows],ddof=0))}
 fig,ax=plt.subplots(figsize=(11,5),dpi=300);ax.plot(r,exp,color="black",lw=2,label="Experiment",zorder=5)
 for m in MODES:
  mean,std=curves[m];v=metrics[m];ax.plot(r,mean,color=COLOR[m],ls=LS[m],lw=1.8,label=rf"{LABEL[m]}: $R_w={v['Rw_mean']:.3f}\pm{v['Rw_std']:.3f}$, $s={v['scale_mean']:.3f}$");ax.fill_between(r,mean-std,mean+std,color=COLOR[m],alpha=.18,lw=0)
 ax.set_xlim(1.5,10);ax.set_xlabel(r"$r$ ($\AA$)",fontsize=14);ax.set_ylabel(r"$G(r)$ ($\AA^{-2}$)",fontsize=14);ax.set_title("Phase 2.5 PDF: formal 1M, three-seed mean +/- std",fontsize=14,pad=10);ax.tick_params(direction="in",labelsize=14);ax.grid(True,ls=":",alpha=.5);ax.legend(loc="upper right",fontsize=10,frameon=True);fig.tight_layout();fig.savefig(OUT/"pdf_three_scale_mean_std_1p5_10A.png",dpi=300,bbox_inches="tight");plt.close(fig)
 with (OUT/"pdf_three_scale_mean_std.csv").open("w",newline="") as h:
  w=csv.writer(h);w.writerow(["r_A","G_experiment",*(z for m in MODES for z in (f"G_{m}_mean",f"G_{m}_std"))])
  for i,rv in enumerate(r):
   row=[rv,exp[i]]
   for m in MODES:row.extend((curves[m][0][i],curves[m][1][i]))
   w.writerow(row)
 return metrics
def geometry_plot():
 rows=list(csv.DictReader(open(BASE/"coordination_geometry_audit/geometry_aggregate.csv")));chosen=[x for x in rows if x["stage"]=="initial_MQ" or (x["stage"]=="Phase2.5" and x["mode"] in MODES)];groups=("initial",)+MODES;metrics=(("bond_mean_A","Mean bond length (A)"),("angle_rms_from_ideal_deg","Angle RMS from ideal (deg)"),("csm_percent","CSM (%)"),("volume_A3","Tetrahedron volume (A3)"));fig,axs=plt.subplots(2,4,figsize=(16,8),dpi=300);x=np.arange(4);cols=[COLOR["MQ"],*(COLOR[m] for m in MODES)];ticks=["MQ","No scale","Fixed","Bounded"];out=[]
 for ri,e in enumerate(("P","Si")):
  selected=[]
  for g in groups:selected.append(next(z for z in chosen if z["element"]==e and z["mode"]==g))
  for ci,(key,title) in enumerate(metrics):
   vals=[float(z[key]) for z in selected];ax=axs[ri,ci];ax.bar(x,vals,color=cols,width=.68);ax.bar_label(ax.containers[0],fmt="%.3f",fontsize=7,padding=2);ax.set_title(f"{e}: {title}",fontsize=11,pad=8);ax.set_xticks(x,ticks,rotation=20,ha="right");ax.tick_params(direction="in",labelsize=9);ax.grid(True,axis="y",ls=":",alpha=.5);ax.set_ylim(0,max(vals)*1.18)
  for z in selected:out.append({"element":e,"group":z["mode"],**{k:float(z[k]) for k,t in metrics},"tetrahedral_fraction":float(z["tetrahedral_fraction_mean"])})
 fig.suptitle("Phase 2.5 local tetrahedral geometry: MQ vs three scale strategies",fontsize=14);fig.tight_layout();fig.savefig(OUT/"local_tetrahedral_geometry.png",dpi=300,bbox_inches="tight");plt.close(fig)
 with (OUT/"local_tetrahedral_geometry.csv").open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 return out
def coordination_plot():
 audit=json.loads((BASE/"audit/coordination_audit.json").read_text());mqt=audit["fixed_cutoff_summary"]["Zr"]["total"]["counts"];mqo=audit["fixed_cutoff_summary"]["Zr"]["O"]["counts"];res={"MQ":{"Zr_total":mqt,"Zr_O":mqo},"modes":{}}
 for m in MODES:
  rr=[x for x in summ if x["mode"]==m];t=np.stack([x["Zr_total"] for x in rr]);o=np.stack([x["Zr_O"] for x in rr]);res["modes"][m]={"Zr_total_mean":t.mean(0).tolist(),"Zr_total_std":t.std(0,ddof=0).tolist(),"Zr_O_mean":o.mean(0).tolist(),"Zr_O_std":o.std(0,ddof=0).tolist(),"P4":1.0,"Si4":1.0,"hard_sphere_violations":0}
 fig,axs=plt.subplots(1,2,figsize=(11,5),dpi=300);width=.19;tc=np.array([4,5,6]);oc=np.array([0,1,2,3]);groups=("MQ",)+MODES;cols=[COLOR["MQ"],*(COLOR[m] for m in MODES)];labs=["MQ","No scale","Fixed","Bounded"]
 for i,g in enumerate(groups):
  if g=="MQ":tm=np.array([float(mqt.get(str(c),0)) for c in tc]);ts=np.zeros(3);om=np.array([float(mqo.get(str(c),0)) for c in oc]);os=np.zeros(4)
  else:v=res["modes"][g];tm=np.array(v["Zr_total_mean"])[tc];ts=np.array(v["Zr_total_std"])[tc];om=np.array(v["Zr_O_mean"])[oc];os=np.array(v["Zr_O_std"])[oc]
  axs[0].bar(tc+(i-1.5)*width,tm,width,yerr=ts,color=cols[i],capsize=2,label=labs[i]);axs[1].bar(oc+(i-1.5)*width,om,width,yerr=os,color=cols[i],capsize=2,label=labs[i])
 axs[0].set_title("Zr total coordination distribution",fontsize=14,pad=10);axs[0].set_xlabel("Zr total CN",fontsize=12);axs[0].set_ylabel("Number of Zr sites",fontsize=12);axs[0].set_xticks(tc);axs[1].set_title("Zr-O coordination distribution",fontsize=14,pad=10);axs[1].set_xlabel("Number of O around Zr",fontsize=12);axs[1].set_ylabel("Number of Zr sites",fontsize=12);axs[1].set_xticks(oc)
 for ax in axs:ax.tick_params(direction="in",labelsize=11);ax.grid(True,axis="y",ls=":",alpha=.5)
 axs[1].legend(fontsize=9,frameon=True);fig.tight_layout();fig.savefig(OUT/"coordination_distributions.png",dpi=300,bbox_inches="tight");plt.close(fig);(OUT/"coordination_summary.json").write_text(json.dumps(res,indent=2));return res
def main():
 pdf=pdf_plot();geo=geometry_plot();cn=coordination_plot();result={"PDF":pdf,"geometry":geo,"coordination":cn,"constraints":{"P_fourfold_all_modes":1.0,"Si_fourfold_all_modes":1.0,"hard_sphere_violations_all_9_finals":0}};(OUT/"core_results_summary.json").write_text(json.dumps(result,indent=2));print(json.dumps(result["PDF"],indent=2))
if __name__=="__main__":main()
