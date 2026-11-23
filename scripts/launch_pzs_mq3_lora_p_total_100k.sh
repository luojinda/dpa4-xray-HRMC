#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/dpa4-xray-HRMC

export DPA4_MODEL_PATH=models/PZS_new_20261121/DPA4_PZS_new_LoRA.pt2
export PYTHONPATH=src

PYTHON=/root/miniconda3/envs/dpa4_torch211/bin/python
BASE=results/PZS_phase3/phase3C/PZS_MQ_3
COMMON=(
  --attempts 100000
  --temperature 100
  --seed 101
  --move-max 0.1
  --fit-rmin 1.4
  --fit-rmax 10.0
  --structure inputs/PZS_MQ_3/PZS_equilibrated.cif
  --checkpoint-interval 10000
  --p-coordination-mode P-total
  --p-cl-cutoff 2.3
)

NO_SCALE_TAG=PZS_MQ_3/validation_100k_seed101_T100_d0p1_r1p4_10p0A_lora_no_scale_p_total
NO_SCALE_OUT=$BASE/validation_100k_seed101_T100_d0p1_r1p4_10p0A_lora_no_scale_p_total
mkdir -p "$NO_SCALE_OUT"
"$PYTHON" scripts/run_phase3C_inline.py \
  "${COMMON[@]}" \
  --tag "$NO_SCALE_TAG" \
  --scale-mode no_scale \
  > "$NO_SCALE_OUT/run.log" 2>&1

DYNAMIC_TAG=PZS_MQ_3/validation_100k_seed101_T100_d0p1_r1p4_10p0A_lora_bounded_dynamic_p_total
DYNAMIC_OUT=$BASE/validation_100k_seed101_T100_d0p1_r1p4_10p0A_lora_bounded_dynamic_p_total
mkdir -p "$DYNAMIC_OUT"
"$PYTHON" scripts/run_phase3C_inline.py \
  "${COMMON[@]}" \
  --tag "$DYNAMIC_TAG" \
  --scale-mode bounded_dynamic \
  > "$DYNAMIC_OUT/run.log" 2>&1
