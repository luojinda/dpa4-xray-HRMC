import argparse, json, os
from dataclasses import replace
from pathlib import Path
from dpa4_xray_hrmc.mlip.dpa4_backend import DPA4Backend
from dpa4_xray_hrmc.rmc.config import load_phase2_config
from dpa4_xray_hrmc.rmc.coordination import CoordinationCutoffs
from dpa4_xray_hrmc.rmc.runner_phase3 import run_inline

p=argparse.ArgumentParser(); p.add_argument('--attempts',type=int,default=1000); p.add_argument('--temperature',type=float,default=300.0); p.add_argument('--seed',type=int,default=101); p.add_argument('--tag',default='pilot'); p.add_argument('--move-max',type=float,default=None,help='Override maximum Cartesian displacement d in A'); p.add_argument('--checkpoint-interval',type=int,default=10000); p.add_argument('--resume',action='store_true')
p.add_argument('--fit-rmin',type=float,default=None,help='Override PDF fit-range lower bound in A')
p.add_argument('--fit-rmax',type=float,default=None,help='Override PDF fit-range upper bound in A')
p.add_argument('--scale-mode',choices=['no_scale','fixed_initial','bounded_dynamic'],default='no_scale')
p.add_argument('--structure',default=None,help='Override input structure path relative to repository root')
p.add_argument('--p-coordination-mode',choices=['P-O','P-total'],default='P-O')
p.add_argument('--p-cl-cutoff',type=float,default=None,help='P-Cl cutoff in A; required for P-total')
p.add_argument('--weight-p',type=float,default=800.0,help='P coordination penalty weight')
p.add_argument('--weight-si',type=float,default=2500.0,help='Si coordination penalty weight')
p.add_argument('--weight-zr-total',type=float,default=1.0,help='Zr total-CN distribution penalty weight')
p.add_argument('--weight-zr-o',type=float,default=0.2,help='Zr-O CN distribution penalty weight')
a=p.parse_args(); root=Path(__file__).resolve().parents[1]
c=load_phase2_config(root/'configs/pzs_phase2_common.yaml')
overrides={}
if a.move_max is not None:
    if a.move_max <= 0.0:
        p.error('--move-max must be positive')
    overrides['move_max_A']=a.move_max
if a.fit_rmin is not None:
    overrides['fit_rmin_A']=a.fit_rmin
if a.fit_rmax is not None:
    overrides['fit_rmax_A']=a.fit_rmax
if a.structure is not None:
    if not (root/a.structure).is_file():
        p.error(f'--structure does not exist: {a.structure}')
    overrides['structure']=a.structure
if overrides:
    c=replace(c,**overrides)
    try:
        c.validate()
    except ValueError as exc:
        p.error(str(exc))
weights=(a.weight_p,a.weight_si,a.weight_zr_total,a.weight_zr_o)
if any(value < 0.0 for value in weights):
    p.error('coordination penalty weights must be non-negative')
cut=CoordinationCutoffs.from_mapping(json.loads((root/'configs/pzs_phase2_5_coordination_cutoffs.json').read_text()))
w={'P':a.weight_p,'Si':a.weight_si,'Zr_total':a.weight_zr_total,'Zr_O':a.weight_zr_o}
b=DPA4Backend(os.environ['DPA4_MODEL_PATH'])
print(json.dumps(run_inline(
    root,c,cut,w,b,root/'results/PZS_phase3/phase3C'/a.tag,
    seed=a.seed,
    target_attempts=a.attempts,
    temperature_K=a.temperature,
    checkpoint_interval=a.checkpoint_interval,
    resume=a.resume,
    scale_mode=a.scale_mode,
    p_coordination_mode=a.p_coordination_mode,
    p_cl_cutoff_A=a.p_cl_cutoff,
),indent=2))
