import argparse, json, os
from pathlib import Path
from dpa4_xray_hrmc.mlip.dpa4_backend import DPA4Backend
from dpa4_xray_hrmc.rmc.config import load_phase2_config
from dpa4_xray_hrmc.rmc.coordination import CoordinationCutoffs
from dpa4_xray_hrmc.rmc.runner_phase3 import run_alternating

p=argparse.ArgumentParser(); p.add_argument('--attempts',type=int,default=10000); p.add_argument('--accepted-per-block',type=int,default=500); p.add_argument('--fire-steps',type=int,default=5); p.add_argument('--seed',type=int,default=101); p.add_argument('--tag',default='pilot_10k')
a=p.parse_args(); root=Path(__file__).resolve().parents[1]
c=load_phase2_config(root/'configs/pzs_phase2_common.yaml')
cut=CoordinationCutoffs.from_mapping(json.loads((root/'configs/pzs_phase2_5_coordination_cutoffs.json').read_text()))
w={'P':800.0,'Si':2500.0,'Zr_total':1.0,'Zr_O':0.2}
b=DPA4Backend(os.environ['DPA4_MODEL_PATH'])
print(json.dumps(run_alternating(root,c,cut,w,b,root/'results/PZS_phase3/phase3B'/a.tag,a.seed,a.attempts,a.accepted_per_block,a.fire_steps),indent=2))

