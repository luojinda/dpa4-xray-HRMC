"""Compare MQ with three-seed mean Pure-RMC/Phase-3 partial RDF and CN curves."""
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from dpa4_xray_hrmc.xray_pdf.partial_rdf import compute_partial_rdfs
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"results/PZS_phase3/calibration/partial_rdf_four_structures"
RCUT=6.0;DR=0.02;XRANGE=(0.8,5.0);SEEDS=(101,202,303);MQ_PATH=ROOT/"PDF/PZS_quench_222.cif"
ENSEMBLES={
"Pure RMC + CN\n(3-seed mean)":tuple(ROOT/f"results/PZS_phase2_5/formal_1M/no_scale/seed_{s}/final_structure.extxyz" for s in SEEDS),
"Alternating 100k\n(3-seed mean)":tuple(ROOT/f"results/PZS_phase3/phase3B/validation_100k_seed{s}_A500_F10/final_structure.extxyz" for s in SEEDS),
"Inline 100k\n(3-seed mean)":tuple(ROOT/f"results/PZS_phase3/phase3C/validation_100k_seed{s}_T100/final_structure.extxyz" for s in SEEDS)}
PAIRS=(("Zr","Cl"),("Zr","O"),("P","O"),("Si","Cl"),("Si","O"))
STYLES={"MQ":{"color":"#666666","linestyle":"--","linewidth":1.6},"Pure RMC + CN\n(3-seed mean)":{"color":"#377EB8","linestyle":"-.","linewidth":1.7},"Alternating 100k\n(3-seed mean)":{"color":"#15572C","linestyle":"-","linewidth":1.8},"Inline 100k\n(3-seed mean)":{"color":"#FF0025","linestyle":"-","linewidth":1.8}}
def key(a,b):return tuple(sorted((a,b)))
def safe(s):return s.replace("\n"," ").replace(" ","_").replace("+","plus")
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 mq=compute_partial_rdfs(read(MQ_PATH),RCUT,DR)
 groups={label:[compute_partial_rdfs(read(path),RCUT,DR) for path in paths] for label,paths in ENSEMBLES.items()}
 manifest={"rdf_definition":"Periodic partial g_ab(r), normalized so dN_b=rho_b*g_ab(r)*4*pi*r^2*dr.","cumulative_definition":"N_neighbor_around_central(r)=cumulative raw pair count / N_central.","rdf_rcut_A":RCUT,"rdf_dr_A":DR,"plot_range_A":list(XRANGE),"seeds":list(SEEDS),"structures":{"MQ":str(MQ_PATH.relative_to(ROOT)),**{label.replace("\n"," "):[str(p.relative_to(ROOT)) for p in paths] for label,paths in ENSEMBLES.items()}},"statistics":"Pointwise population mean and standard deviation (ddof=0) over seeds 101, 202, 303. No smoothing or rebinning.","pairs":{}}
 for central,neighbor in PAIRS:
  pk=key(central,neighbor);name=f"{central}-{neighbor}";mqg=mq.g[pk];mqn=np.cumsum(mq.counts[pk])/mq.element_counts[central];data={}
  for label,rdfs in groups.items():
   gs=np.stack([x.g[pk] for x in rdfs]);ns=np.stack([np.cumsum(x.counts[pk])/x.element_counts[central] for x in rdfs])
   data[label]={"gm":gs.mean(0),"gs":gs.std(0,ddof=0),"nm":ns.mean(0),"ns":ns.std(0,ddof=0)}
  table=OUT/f"{name.replace('-','_')}_rdf_and_cumulative_CN.csv";fields=["r_A","g_MQ",f"N_{neighbor}_around_{central}_MQ"]
  for label in ENSEMBLES:
   p=safe(label);fields.extend((f"g_mean_{p}",f"g_std_{p}",f"N_mean_{neighbor}_around_{central}_{p}",f"N_std_{neighbor}_around_{central}_{p}"))
  with table.open("w",newline="",encoding="utf-8") as h:
   w=csv.writer(h);w.writerow(fields)
   for i,r in enumerate(mq.r_A):
    row=[r,mqg[i],mqn[i]]
    for label in ENSEMBLES:
     v=data[label];row.extend((v["gm"][i],v["gs"][i],v["nm"][i],v["ns"][i]))
    w.writerow(row)
  fig,ax=plt.subplots(figsize=(11,5),dpi=300);ax.plot(mq.r_A,mqg,label="MQ",**STYLES["MQ"])
  for label,v in data.items():
   ax.plot(mq.r_A,v["gm"],label=label.replace("\n"," "),**STYLES[label]);ax.fill_between(mq.r_A,np.maximum(v["gm"]-v["gs"],0),v["gm"]+v["gs"],color=STYLES[label]["color"],alpha=.18,lw=0)
  vis=(mq.r_A>=XRANGE[0])&(mq.r_A<=XRANGE[1]);ymax=max(float(mqg[vis].max()),*(float((v["gm"]+v["gs"])[vis].max()) for v in data.values()))
  ax.set_xlim(*XRANGE);ax.set_ylim(0,1.08*ymax);ax.set_xlabel(r"$r$ ($\AA$)",fontsize=14);ax.set_ylabel(rf"partial $g_{{\mathrm{{{central}-{neighbor}}}}}(r)$",fontsize=14);ax.set_title(f"{name} partial pair distribution function",fontsize=14,pad=10);ax.tick_params(direction="in",labelsize=14);ax.grid(True,ls=":",alpha=.5);ax.legend(loc="upper left",frameon=True,fontsize=10)
  ins=inset_axes(ax,width="44%",height="48%",loc="upper right",borderpad=1.1);ins.plot(mq.r_A,mqn,**STYLES["MQ"])
  for label,v in data.items():
   ins.plot(mq.r_A,v["nm"],**STYLES[label]);ins.fill_between(mq.r_A,np.maximum(v["nm"]-v["ns"],0),v["nm"]+v["ns"],color=STYLES[label]["color"],alpha=.18,lw=0)
  maxn=max(float(mqn[vis].max()),*(float((v["nm"]+v["ns"])[vis].max()) for v in data.values()));ins.set_xlim(*XRANGE);ins.set_ylim(0,1.08*maxn);ins.set_xlabel(r"$r$ ($\AA$)",fontsize=9);ins.set_ylabel(rf"$N_{{\mathrm{{{neighbor}\ around\ {central}}}}}(r)$",fontsize=9);ins.set_title("Cumulative bond number",fontsize=9,pad=4);ins.tick_params(direction="in",labelsize=8);ins.grid(True,ls=":",alpha=.5)
  fig.subplots_adjust(left=.09,right=.98,bottom=.14,top=.88);plot=OUT/f"{name.replace('-','_')}_partial_rdf_with_CN_inset.png";fig.savefig(plot,dpi=300,bbox_inches="tight");plt.close(fig)
  manifest["pairs"][name]={"plot":str(plot.relative_to(ROOT)),"data":str(table.relative_to(ROOT)),"max_partial_g_in_plot_range":{"MQ":float(mqg[vis].max()),**{label.replace("\n"," "):{"mean_curve_max":float(v["gm"][vis].max()),"largest_pointwise_std":float(v["gs"][vis].max())} for label,v in data.items()}},"cumulative_N_at_5A":{"MQ":float(mqn[vis][-1]),**{label.replace("\n"," "):{"mean":float(v["nm"][vis][-1]),"std":float(v["ns"][vis][-1])} for label,v in data.items()}}}
 (OUT/"partial_rdf_comparison_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8");print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
