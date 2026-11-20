"""Compare seed-101 Inline 100 K 10k displacement amplitudes."""
import csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from dpa4_xray_hrmc.analysis.phase3_structure import Phase3StructureEvaluator
from dpa4_xray_hrmc.rmc.config import load_phase2_config
ROOT=Path(__file__).resolve().parents[1];P3C=ROOT/"results/PZS_phase3/phase3C";OUT=ROOT/"results/PZS_phase3/calibration"
RUNS={.1:"calibration_seed101_T100_10k",.2:"calibration_seed101_T100_d0p2_10k",.3:"calibration_seed101_T100_d0p3_10k"};COL={.1:"#15572C",.2:"#E69F00",.3:"#FF0025"};N=896
def main():
 ev=json.loads((OUT/"calibration_evaluations.json").read_text());by={x["tag"]:x for x in ev};cfg=load_phase2_config(ROOT/"configs/pzs_phase2_common.yaml");evaluator=Phase3StructureEvaluator(ROOT,cfg,ROOT/"configs/pzs_phase2_5_coordination_cutoffs.json");mq_rw=evaluator.evaluate(read(ROOT/cfg.structure))["pdf"]["Rw_no_scale"];traces={};rec={}
 for d,tag in RUNS.items():
  x=by[tag];t=json.loads((P3C/tag/"trace.json").read_text());traces[d]=t;diag=x["diagnostics"];best=min(t,key=lambda z:z["Rw_no_scale"]);mqe=x["recomputed_energy_eV"]-x["delta_energy_from_MQ_eV"]
  rec[d]={"d_A":d,"final_Rw":x["Rw_no_scale"],"best_Rw":best["Rw_no_scale"],"best_Rw_attempt":best["attempted"],"final_delta_E_eV":x["delta_energy_from_MQ_eV"],"final_delta_E_meV_atom":x["delta_energy_from_MQ_eV"]*1000/N,"force_rms_eV_A":x["force_rms_eV_A"],"force_max_eV_A":x["force_max_eV_A"],"acceptance_rate":x["acceptance_rate"],"hard_sphere_reject_fraction":x["hard_sphere_rejected"]/x["attempted"],"wall_s":x["wall_s"],"attempts_per_s":x["attempted"]/x["wall_s"],"P_angle_RMS_deg":diag["tetrahedra"]["P"]["angle_rms_from_ideal_deg"]["mean"],"Si_angle_RMS_deg":diag["tetrahedra"]["Si"]["angle_rms_from_ideal_deg"]["mean"],"P_CSM_percent":diag["tetrahedra"]["P"]["fixed_center_csm_percent"]["mean"],"Si_CSM_percent":diag["tetrahedra"]["Si"]["fixed_center_csm_percent"]["mean"],"P_fourfold":diag["coordination"]["p_fourfold_fraction"],"Si_fourfold":diag["coordination"]["si_fourfold_fraction"],"hard_sphere_violations":diag["hard_sphere"]["violation_total"],"mq_energy_eV":mqe}
 common=min(x["wall_s"] for x in rec.values())
 for d,t in traces.items():
  z=max((x for x in t if x["wall_s"]<=common),key=lambda x:x["wall_s"]);rec[d].update({"common_wall_s":common,"wall_matched_attempt":z["attempted"],"wall_matched_Rw":z["Rw_no_scale"],"wall_matched_delta_E_meV_atom":(z["energy_eV"]-rec[d]["mq_energy_eV"])*1000/N})
 rows=[rec[d] for d in sorted(rec)]
 with (OUT/"inline_displacement_10k_summary.csv").open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 fig,ax=plt.subplots(2,2,figsize=(11,9),dpi=300)
 for d,t in traces.items():
  at=np.array([0,*[z["attempted"] for z in t]]);rw=np.array([mq_rw,*[z["Rw_no_scale"] for z in t]]);de=np.array([0,*[(z["energy_eV"]-rec[d]["mq_energy_eV"])*1000/N for z in t]]);ax[0,0].plot(at,rw,color=COL[d],lw=1.7,label=f"d={d:.1f} A");ax[0,1].plot(at,de,color=COL[d],lw=1.7,label=f"d={d:.1f} A")
 ax[0,0].set_title("PDF residual",fontsize=14,pad=10);ax[0,0].set_xlabel("MC attempts",fontsize=12);ax[0,0].set_ylabel(r"$R_w$ (no scale)",fontsize=12);ax[0,1].set_title("DPA4 energy relative to MQ",fontsize=14,pad=10);ax[0,1].set_xlabel("MC attempts",fontsize=12);ax[0,1].set_ylabel(r"$\Delta E$ (meV/atom)",fontsize=12)
 xx=np.arange(3);width=.36;ds=sorted(rec);colors=[COL[d] for d in ds]
 ax[1,0].bar(xx-width/2,[rec[d]["P_angle_RMS_deg"] for d in ds],width,color=colors,label="P");ax[1,0].bar(xx+width/2,[rec[d]["Si_angle_RMS_deg"] for d in ds],width,color=colors,alpha=.55,hatch="//",label="Si");ax[1,0].set_title("Final tetrahedral angle RMS",fontsize=14,pad=10);ax[1,0].set_ylabel("Angle RMS (deg)",fontsize=12);ax[1,0].set_xticks(xx,[f"d={d:.1f} A" for d in ds]);ax[1,0].legend()
 ax[1,1].bar(xx-width/2,[rec[d]["P_CSM_percent"] for d in ds],width,color=colors,label="P");ax[1,1].bar(xx+width/2,[rec[d]["Si_CSM_percent"] for d in ds],width,color=colors,alpha=.55,hatch="//",label="Si");ax[1,1].set_title("Final tetrahedral CSM",fontsize=14,pad=10);ax[1,1].set_ylabel("Fixed-center CSM (%)",fontsize=12);ax[1,1].set_xticks(xx,[f"d={d:.1f} A" for d in ds]);ax[1,1].legend()
 for a in ax.flat:a.tick_params(direction="in",labelsize=10);a.grid(True,ls=":",alpha=.5)
 ax[0,0].legend();ax[0,1].legend();fig.suptitle("Inline HRMC displacement scan: seed 101, 100 K, 10k attempts",fontsize=14);fig.tight_layout();fig.savefig(OUT/"inline_displacement_10k_comparison.png",dpi=300,bbox_inches="tight");plt.close(fig)
 recommendation="d=0.2 A is preferred: it improves final and wall-matched Rw relative to d=0.1 A, with modest energy and geometry penalties. d=0.3 A has lower acceptance, more hard-sphere rejection, higher energy, and worse Si distortion without matching the d=0.2 A PDF improvement."
 lines=["# Inline HRMC displacement scan","",f"Common wall time: {common:.3f} s.","","| d (A) | final Rw | best Rw (attempt) | wall-matched Rw | dE (meV/atom) | acceptance | HS reject | P angle RMS | Si angle RMS | P CSM | Si CSM |","|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
 for x in rows:lines.append(f"| {x['d_A']:.1f} | {x['final_Rw']:.5f} | {x['best_Rw']:.5f} ({x['best_Rw_attempt']}) | {x['wall_matched_Rw']:.5f} | {x['final_delta_E_meV_atom']:.2f} | {100*x['acceptance_rate']:.2f}% | {100*x['hard_sphere_reject_fraction']:.2f}% | {x['P_angle_RMS_deg']:.3f} | {x['Si_angle_RMS_deg']:.3f} | {x['P_CSM_percent']:.3f} | {x['Si_CSM_percent']:.3f} |")
 lines+=["","## Recommendation","",recommendation];(OUT/"INLINE_DISPLACEMENT_10K_REPORT.md").write_text("\n".join(lines)+"\n");print(json.dumps({"rows":rows,"recommendation":recommendation},indent=2))
if __name__=="__main__":main()
