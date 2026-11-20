"""Compare selected MQ/Pure-RMC/Phase-3 partial RDFs and cumulative CN curves."""
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from dpa4_xray_hrmc.xray_pdf.partial_rdf import compute_partial_rdfs
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/PZS_phase3/calibration/partial_rdf_four_structures"
RCUT=6.0;DR=0.02;XRANGE=(0.8,5.0)
STRUCTURES={
"MQ":ROOT/"PDF/PZS_quench_222.cif",
"Pure RMC + CN\n(no_scale, seed 101 final)":ROOT/"results/PZS_phase2_5/formal_1M/no_scale/seed_101/final_structure.extxyz",
"Alternating 100k\n(seed 303 final)":ROOT/"results/PZS_phase3/phase3B/validation_100k_seed303_A500_F10/final_structure.extxyz",
"Inline 100k\n(seed 303 final)":ROOT/"results/PZS_phase3/phase3C/validation_100k_seed303_T100/final_structure.extxyz"}
PAIRS=(("Zr","Cl"),("Zr","O"),("P","O"),("Si","Cl"),("Si","O"))
STYLES={
"MQ":{"color":"#666666","linestyle":"--","linewidth":1.6},
"Pure RMC + CN\n(no_scale, seed 101 final)":{"color":"#377EB8","linestyle":"-.","linewidth":1.7},
"Alternating 100k\n(seed 303 final)":{"color":"#15572C","linestyle":"-","linewidth":1.8},
"Inline 100k\n(seed 303 final)":{"color":"#FF0025","linestyle":"-","linewidth":1.8}}
def key(a,b):return tuple(sorted((a,b)))
def safe(s):return s.replace("\n"," ").replace(" ","_").replace("+","plus")
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 rdfs={label:compute_partial_rdfs(read(path),RCUT,DR) for label,path in STRUCTURES.items()}
 manifest={"rdf_definition":"Periodic partial g_ab(r), normalized so dN_b=rho_b*g_ab(r)*4*pi*r^2*dr.","cumulative_definition":"N_neighbor_around_central(r)=cumulative raw pair count / N_central.","rdf_rcut_A":RCUT,"rdf_dr_A":DR,"plot_range_A":list(XRANGE),"structures":{label.replace("\n"," "):str(path.relative_to(ROOT)) for label,path in STRUCTURES.items()},"selection_note":"Pure RMC uses the Phase 2.5 no_scale+coordination seed-101 final structure, which has the lowest final PDF/total loss among its three seeds. Phase 3 structures are the previously selected seed-303 final structures.","pairs":{}}
 for central,neighbor in PAIRS:
  pk=key(central,neighbor);name=f"{central}-{neighbor}";data={}
  for label,rdf in rdfs.items():
   data[label]=(rdf.g[pk],np.cumsum(rdf.counts[pk])/rdf.element_counts[central])
  table=OUT/f"{name.replace('-','_')}_rdf_and_cumulative_CN.csv"
  fields=["r_A"]
  for label in STRUCTURES:
   pre=safe(label);fields.extend((f"g_{pre}",f"N_{neighbor}_around_{central}_{pre}"))
  with table.open("w",newline="",encoding="utf-8") as h:
   w=csv.writer(h);w.writerow(fields)
   for i,r in enumerate(next(iter(rdfs.values())).r_A):
    row=[r]
    for label in STRUCTURES:row.extend((data[label][0][i],data[label][1][i]))
    w.writerow(row)
  fig,ax=plt.subplots(figsize=(11,5),dpi=300)
  for label in STRUCTURES:ax.plot(rdfs[label].r_A,data[label][0],label=label.replace("\n"," "),**STYLES[label])
  rr=next(iter(rdfs.values())).r_A;vis=(rr>=XRANGE[0])&(rr<=XRANGE[1]);ymax=max(float(v[0][vis].max()) for v in data.values())
  ax.set_xlim(*XRANGE);ax.set_ylim(0,1.08*ymax);ax.set_xlabel(r"$r$ ($\AA$)",fontsize=14);ax.set_ylabel(rf"partial $g_{{\mathrm{{{central}-{neighbor}}}}}(r)$",fontsize=14)
  ax.set_title(f"{name} partial pair distribution function",fontsize=14,pad=10);ax.tick_params(direction="in",labelsize=14);ax.grid(True,ls=":",alpha=.5);ax.legend(loc="upper left",frameon=True,fontsize=10)
  ins=inset_axes(ax,width="44%",height="48%",loc="upper right",borderpad=1.1)
  for label in STRUCTURES:ins.plot(rdfs[label].r_A,data[label][1],**STYLES[label])
  maxn=max(float(v[1][vis].max()) for v in data.values());ins.set_xlim(*XRANGE);ins.set_ylim(0,1.08*maxn);ins.set_xlabel(r"$r$ ($\AA$)",fontsize=9);ins.set_ylabel(rf"$N_{{\mathrm{{{neighbor}\ around\ {central}}}}}(r)$",fontsize=9);ins.set_title("Cumulative bond number",fontsize=9,pad=4);ins.tick_params(direction="in",labelsize=8);ins.grid(True,ls=":",alpha=.5)
  fig.tight_layout();plot=OUT/f"{name.replace('-','_')}_partial_rdf_with_CN_inset.png";fig.savefig(plot,dpi=300,bbox_inches="tight");plt.close(fig)
  manifest["pairs"][name]={"plot":str(plot.relative_to(ROOT)),"data":str(table.relative_to(ROOT)),"max_partial_g_in_plot_range":{label.replace("\n"," "):float(v[0][vis].max()) for label,v in data.items()},"cumulative_N_at_5A":{label.replace("\n"," "):float(v[1][vis][-1]) for label,v in data.items()}}
 path=OUT/"partial_rdf_comparison_manifest.json";path.write_text(json.dumps(manifest,indent=2),encoding="utf-8");print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
