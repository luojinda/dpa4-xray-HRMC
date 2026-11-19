#!/usr/bin/env python
from __future__ import annotations
import argparse,json
from dataclasses import asdict
from pathlib import Path
import yaml
from dpa4_xray_hrmc.rmc.config import load_phase2_config
from dpa4_xray_hrmc.rmc.coordination import CoordinationCutoffs
from dpa4_xray_hrmc.rmc.objectives import ScaleMode
from dpa4_xray_hrmc.rmc.runner_cn import run_cn_chain
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/pzs_phase2_5_common.yaml");p.add_argument("--mode",required=True,choices=[x.value for x in ScaleMode]);p.add_argument("--seed",required=True,type=int);p.add_argument("--steps",required=True,type=int);p.add_argument("--output-root");p.add_argument("--resume");p.add_argument("--weight-label");a=p.parse_args()
 cp=Path(a.config).resolve();raw=yaml.safe_load(cp.read_text());root=cp.parent.parent
 result=run_cn_chain(load_phase2_config(root/raw["phase2_config"]),CoordinationCutoffs.from_mapping(json.loads((root/raw["coordination_cutoffs"]).read_text())),{k:float(v) for k,v in raw["coordination_weights"].items()},a.mode,a.seed,a.steps,root,a.output_root,a.resume,a.weight_label or raw.get("weight_label","frozen"))
 print(json.dumps(asdict(result),indent=2))
if __name__=="__main__":main()
