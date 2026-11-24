#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/dpa4-xray-HRMC

export DPA4_MODEL_PATH=models/PZS_new_20261121/DPA4_PZS_new_LoRA.pt2
export PYTHONPATH=src

PYTHON=/root/miniconda3/envs/dpa4_torch211/bin/python
TAG=PZS_MQ_3_relaxed/validation_100k_seed101_T100_d0p1_r1p4_10p0A_lora_bounded_dynamic_p_total
OUT=results/PZS_phase3/phase3C/$TAG
mkdir -p "$OUT"

if [[ -f "$OUT/summary.json" ]]; then
  echo "SKIP completed relaxed-MQ3 baseline"
  exit 0
fi

RESUME_ARGS=()
if [[ -f "$OUT/checkpoint_latest.npz" ]]; then
  RESUME_ARGS=(--resume)
fi

"$PYTHON" scripts/run_phase3C_inline.py \
  --attempts 100000 \
  --temperature 100 \
  --seed 101 \
  --move-max 0.1 \
  --fit-rmin 1.4 \
  --fit-rmax 10.0 \
  --structure results/PZS_phase3/relaxations/PZS_MQ_3_lora_fixed_cell/relaxed_structure.extxyz \
  --checkpoint-interval 10000 \
  --p-coordination-mode P-total \
  --p-cl-cutoff 2.3 \
  --scale-mode bounded_dynamic \
  --weight-p 800 \
  --weight-si 2500 \
  --weight-zr-total 1.0 \
  --weight-zr-o 0.2 \
  "${RESUME_ARGS[@]}" \
  > "$OUT/run.log" 2>&1

echo "DONE relaxed-MQ3 baseline 100k"
