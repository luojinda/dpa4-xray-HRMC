from __future__ import annotations
import csv,json,os,platform,sys,time
from dataclasses import asdict,dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any,Mapping
import numpy as np,yaml
from ase import Atoms
from ase.io import write
from dpa4_xray_hrmc.xray_pdf.experimental import read_pdfgetx_gr
from dpa4_xray_hrmc.xray_pdf.structure import load_structure
from .config import Phase2Config
from .coordination import CoordinationComponents,CoordinationCutoffs,IncrementalCoordination
from .diagnostics import coordination_cutoffs_from_initial,structure_diagnostics
from .hard_sphere import HardSphereModel
from .incremental_pdf import IncrementalXrayPDF
from .objectives import ScaleMode,ScaleObjective,Score
from .runner import _curve_table,_git_commit,_json_compatible,_write_structure_pair,file_sha256,inclusive_grid
@dataclass
class CNCounters:
 attempted:int=0;accepted:int=0;hard_sphere_rejected:int=0;metropolis_rejected:int=0;legal_evaluations:int=0;p_break_events:int=0;p_recovery_events:int=0;si_break_events:int=0;si_recovery_events:int=0
@dataclass
class CNTimings:
 initialize_s:float=0.;hard_sphere_s:float=0.;delta_pdf_s:float=0.;delta_cn_s:float=0.;score_s:float=0.;commit_s:float=0.;validation_s:float=0.;checkpoint_s:float=0.;output_s:float=0.
@dataclass(frozen=True)
class CNChainResult:
 output_directory:str;mode:str;seed:int;attempted:int;accepted:int;final_total_loss:float;final_pdf_loss:float;final_scale:float;best_total_step:int;best_pdf_step:int;wall_seconds:float
def _total(p,c,w,s):return float(p.loss/(2*s*s)+c.weighted(w))
def _payload(s):
 return {"p_o":s.p_o.tolist(),"si_o":s.si_o.tolist(),"si_cl":s.si_cl.tolist(),"si_total":s.si_total.tolist(),"zr_o":s.zr_o.tolist(),"zr_cl":s.zr_cl.tolist(),"zr_total":s.zr_total.tolist(),"zr_hist_total":s.zr_hist_total.tolist(),"zr_hist_o":s.zr_hist_o.tolist(),"components":s.components.as_dict(),"p_fourfold_fraction":float(np.mean(s.p_o==4)),"si_fourfold_fraction":float(np.mean(s.si_total==4))}
def _write_cn(d,stem,atoms,cn):
 s=cn.full_recompute(atoms);(d/f"{stem}_coordination.json").write_text(json.dumps(_payload(s),indent=2))
 with (d/f"{stem}_coordination.csv").open("w",newline="") as f:
  w=csv.writer(f);w.writerow(["atom_index","element","CN_O","CN_Cl","CN_total"])
  for e,o,cl in (("P",s.p_o,np.zeros_like(s.p_o)),("Si",s.si_o,s.si_cl),("Zr",s.zr_o,s.zr_cl)):
   for i,a,b in zip(cn.indices[e],o,cl,strict=True):w.writerow([int(i),e,int(a),int(b),int(a+b)])
def _save(path,pdf,counters,current_pdf,current_total,btp,bpp,bt,bpl,bts,bps,rngs):
 m={"counters":asdict(counters),"current_pdf":asdict(current_pdf),"current_total":current_total,"best_total":bt,"best_pdf_loss":bpl,"best_total_step":bts,"best_pdf_step":bps,"rng_states":[_json_compatible(r.bit_generator.state) for r in rngs]}
 np.savez_compressed(path,positions_A=pdf.positions_A,counts=pdf.counts,g_fit_A2=pdf.g_r_A2,best_total_positions_A=btp,best_pdf_positions_A=bpp,metadata_json=np.asarray(json.dumps(m)))
def _restore(path,pdf,cn,rngs):
 z=np.load(path,allow_pickle=False);pos=z["positions_A"];pdf.atoms.positions[:]=pos;pdf.counts[:]=z["counts"];pdf.g_r_A2[:]=z["g_fit_A2"];a=cn.atoms.copy();a.positions[:]=pos;cn.restore(pos,cn.full_recompute(a));m=json.loads(str(z["metadata_json"]))
 for r,s in zip(rngs,m["rng_states"],strict=True):r.bit_generator.state=s
 return CNCounters(**m["counters"]),Score(**m["current_pdf"]),float(m["current_total"]),z["best_total_positions_A"].copy(),z["best_pdf_positions_A"].copy(),float(m["best_total"]),float(m["best_pdf_loss"]),int(m["best_total_step"]),int(m["best_pdf_step"])
def run_cn_chain(config,cutoffs,cn_weights,mode,seed,target_attempts,project_root=".",output_root_override=None,resume_checkpoint=None,weight_label="frozen"):
 config.validate();mode=ScaleMode(mode);root=Path(project_root).resolve();sp=(root/config.structure).resolve();ep=(root/config.experimental_gr).resolve();outroot=Path(output_root_override or root/"results/PZS_phase2_5").resolve();out=outroot/weight_label/mode.value/f"seed_{seed}";ck=out/"checkpoints";sn=out/"snapshots";ck.mkdir(parents=True,exist_ok=True);sn.mkdir(exist_ok=True)
 start=time.perf_counter();tm=CNTimings();t=time.perf_counter();info=load_structure(sp);exp=read_pdfgetx_gr(ep);mask=(exp.r_A>=config.fit_rmin_A)&(exp.r_A<=config.fit_rmax_A);q=inclusive_grid(config.qmin_Ainv,config.qmax_Ainv,config.qstep_Ainv);pdf=IncrementalXrayPDF(info.atoms,q,exp.r_A[mask],config.rdf_rcut_A,config.rdf_dr_A,config.biso_A2,config.lorch_mod);obj=ScaleObjective.from_initial(pdf.g_r_A2,exp.g_r_A2[mask]);hs=HardSphereModel.from_atoms(info.atoms,config.hard_sphere_scale);cn=IncrementalCoordination(info.atoms,cutoffs);cur=obj.score(mode,pdf.g_r_A2);total=_total(cur,cn.components,cn_weights,config.sigma_rmc_A2);tm.initialize_s=time.perf_counter()-t
 rngs=tuple(np.random.default_rng(x) for x in np.random.SeedSequence(seed).spawn(3));ra,rd,rx=rngs;c=CNCounters();btp=pdf.positions_A.copy();bpp=pdf.positions_A.copy();bt=total;bpl=cur.loss;bts=bps=0
 if resume_checkpoint:c,cur,total,btp,bpp,bt,bpl,bts,bps=_restore(Path(resume_checkpoint),pdf,cn,rngs)
 resolved={**config.to_dict(),"mode":mode.value,"seed":seed,"target_attempts":target_attempts,"s0":obj.s0,"scale_bounds":[obj.lower_scale,obj.upper_scale],"coordination_cutoffs_A":cutoffs.as_mapping(),"coordination_weights":dict(cn_weights),"coordination_definition":cn.serializable_definition(),"total_loss":"X2/(2*sigma^2)+weighted_CN"}
 (out/"run_config_resolved.yaml").write_text(yaml.safe_dump(resolved,sort_keys=False));(out/"input_manifest.json").write_text(json.dumps({"structure":{"path":str(sp),"sha256":file_sha256(sp)},"experiment":{"path":str(ep),"sha256":file_sha256(ep)},"git_commit":_git_commit(root),"python":sys.version,"platform":platform.platform()},indent=2));(out/"hard_sphere.json").write_text(json.dumps(hs.serializable(),indent=2));_write_structure_pair(out,"initial_structure",info.atoms);_write_cn(out,"initial_structure",info.atoms,cn)
 fields=["attempted_step","accepted_total","hs_rejected_total","metropolis_rejected_total","acceptance_rate","s_current","s_raw","pdf_loss","pdf_dimensionless","cn_loss","total_loss","C_P","C_Si","C_Zr_total","C_Zr_O","weighted_P","weighted_Si","weighted_Zr_total","weighted_Zr_O","P_fourfold_fraction","Si_fourfold_fraction","Rw_no_scale","Rw_fixed_initial","Rw_bounded_dynamic","Rw_shape_only","wall_s"];tp=out/"trace.csv";append=tp.exists() and c.attempted>0;f=tp.open("a" if append else "w",newline="");w=csv.DictWriter(f,fieldnames=fields)
 if not append:w.writeheader()
 drift=[];loop=time.perf_counter()
 while c.attempted<target_attempts:
  c.attempted+=1;i=int(ra.integers(len(pdf.atoms)));new=pdf.positions_A[i]+rd.uniform(-config.move_max_A,config.move_max_A,3);t=time.perf_counter();valid,*_=hs.check_move(pdf.atoms,i,new);tm.hard_sphere_s+=time.perf_counter()-t
  if not valid:c.hard_sphere_rejected+=1
  else:
   t=time.perf_counter();pp=pdf.propose_move(i,new);tm.delta_pdf_s+=time.perf_counter()-t;t=time.perf_counter();cp=cn.propose_move(i,new);tm.delta_cn_s+=time.perf_counter()-t;c.legal_evaluations+=1;t=time.perf_counter();trial=obj.trial_score(mode,pdf.g_r_A2,cur,pp.delta_g_A2);newtotal=_total(trial,cp.trial.components,cn_weights,config.sigma_rmc_A2);delta=newtotal-total;accept=delta<=0 or np.log(rx.random()) < -delta;tm.score_s+=time.perf_counter()-t
   if accept:
    op=int(np.count_nonzero(cn.p_o!=4));osi=int(np.count_nonzero(cn.si_o+cn.si_cl!=4));npb=int(np.count_nonzero(cp.trial.p_o!=4));nsi=int(np.count_nonzero(cp.trial.si_total!=4));c.p_break_events+=max(npb-op,0);c.p_recovery_events+=max(op-npb,0);c.si_break_events+=max(nsi-osi,0);c.si_recovery_events+=max(osi-nsi,0);t=time.perf_counter();pdf.commit(pp);cn.commit(cp);tm.commit_s+=time.perf_counter()-t;cur,total=trial,newtotal;c.accepted+=1
    if total<bt:bt,bts,btp=total,c.attempted,pdf.positions_A.copy()
    if cur.loss<bpl:bpl,bps,bpp=cur.loss,c.attempted,pdf.positions_A.copy()
   else:pdf.reject(pp);cn.reject(cp);c.metropolis_rejected+=1
  if c.attempted%config.trace_interval==0 or c.attempted==target_attempts:
   common=obj.common_scores(cur.A,cur.B);cc=cn.components;row={"attempted_step":c.attempted,"accepted_total":c.accepted,"hs_rejected_total":c.hard_sphere_rejected,"metropolis_rejected_total":c.metropolis_rejected,"acceptance_rate":c.accepted/c.attempted,"s_current":cur.scale,"s_raw":cur.B/cur.A,"pdf_loss":cur.loss,"pdf_dimensionless":cur.loss/(2*config.sigma_rmc_A2**2),"cn_loss":cc.weighted(cn_weights),"total_loss":total,**cc.as_dict(),"weighted_P":cn_weights["P"]*cc.p,"weighted_Si":cn_weights["Si"]*cc.si,"weighted_Zr_total":cn_weights["Zr_total"]*cc.zr_total,"weighted_Zr_O":cn_weights["Zr_O"]*cc.zr_o,"P_fourfold_fraction":float(np.mean(cn.p_o==4)),"Si_fourfold_fraction":float(np.mean(cn.si_o+cn.si_cl==4)),**{k:common[k] for k in ("Rw_no_scale","Rw_fixed_initial","Rw_bounded_dynamic","Rw_shape_only")},"wall_s":time.perf_counter()-start};w.writerow(row);f.flush()
  if c.attempted%config.validation_interval==0:
   t=time.perf_counter();pe=pdf.validation_errors();ce=cn.validation_errors();A,B=obj.coefficients(pdf.g_r_A2);record={"attempted_step":c.attempted,"pdf":pe,"cn":ce,"abs_A_drift":abs(A-cur.A),"abs_B_drift":abs(B-cur.B)};drift.append(record);tm.validation_s+=time.perf_counter()-t
   if pe["max_abs_hist_count"] or pe["max_abs_G_A-2"]>1e-8 or ce["max_abs_integer_error"] or ce["max_abs_component_error"]>1e-12:raise RuntimeError(f"drift {record}")
  if c.attempted%config.snapshot_interval==0:write(sn/f"step_{c.attempted:09d}.extxyz",pdf.atoms,format="extxyz")
  if c.attempted%config.checkpoint_interval==0:
   t=time.perf_counter();_save(ck/f"checkpoint_{c.attempted:09d}.npz",pdf,c,cur,total,btp,bpp,bt,bpl,bts,bps,rngs);tm.checkpoint_s+=time.perf_counter()-t
 f.close();sampling=time.perf_counter()-loop;t=time.perf_counter();structures={"final_structure":pdf.atoms.copy()}
 for stem,pos in (("best_total_structure",btp),("best_pdf_structure",bpp)):
  a=info.atoms.copy();a.positions[:]=pos;structures[stem]=a
 metrics={};diag={};generic=coordination_cutoffs_from_initial(info.atoms)
 for stem,a in structures.items():
  _write_structure_pair(out,stem,a);_write_cn(out,stem,a,cn);metrics[stem]=_curve_table(out/f"{stem}_calc_Gr.dat",a,config,q,exp.r_A,exp.g_r_A2,obj,mode);diag[stem]=structure_diagnostics(info.atoms,a,hs,generic)
 (out/"diagnostics.json").write_text(json.dumps(diag,indent=2));tm.output_s+=time.perf_counter()-t;wall=time.perf_counter()-start;timing={**asdict(tm),"sampling_wall_s":sampling,"total_wall_s":wall,"attempts_per_s_sampling":c.attempted/sampling,"legal_evaluations_per_s_sampling":c.legal_evaluations/sampling,"counters":asdict(c)};(out/"timings.json").write_text(json.dumps(timing,indent=2));fc=cn.current_snapshot();summary={"mode":mode.value,"seed":seed,"s0":obj.s0,"scale_bounds":[obj.lower_scale,obj.upper_scale],"final_step":c.attempted,"final_pdf_loss":cur.loss,"final_pdf_dimensionless":cur.loss/(2*config.sigma_rmc_A2**2),"final_cn_loss":fc.components.weighted(cn_weights),"final_total_loss":total,"final_scale":cur.scale,"final_scale_raw":cur.B/cur.A,"best_total":bt,"best_total_step":bts,"best_pdf_loss":bpl,"best_pdf_step":bps,"final_coordination":_payload(fc),"structure_metrics":metrics,"drift_checks":drift,"timings":timing};(out/"summary.json").write_text(json.dumps(summary,indent=2))
 return CNChainResult(str(out),mode.value,seed,c.attempted,c.accepted,total,cur.loss,cur.scale,bts,bps,wall)
