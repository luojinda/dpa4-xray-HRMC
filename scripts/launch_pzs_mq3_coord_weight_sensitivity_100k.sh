#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/dpa4-xray-HRMC

export DPA4_MODEL_PATH=models/PZS_new_20261121/DPA4_PZS_new_LoRA.pt2
export PYTHONPATH=src

PYTHON=/root/miniconda3/envs/dpa4_torch211/bin/python
BASE=results/PZS_phase3/phase3C/PZS_MQ_3/coord_weight_sensitivity_r1p4_10p0A
mkdir -p "$BASE"

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
  --scale-mode bounded_dynamic
)

run_case() {
  local label=$1
  local weight_p=$2
  local weight_si=$3
  local weight_zr_total=$4
  local weight_zr_o=$5
  local out="$BASE/$label"
  local tag="PZS_MQ_3/coord_weight_sensitivity_r1p4_10p0A/$label"

  if [[ -f "$out/summary.json" ]]; then
    echo "SKIP completed $label"
    return
  fi

  mkdir -p "$out"
  resume_args=()
  if [[ -f "$out/checkpoint_latest.npz" ]]; then
    resume_args=(--resume)
  fi

  echo "START $label P=$weight_p Si=$weight_si Zr_total=$weight_zr_total Zr_O=$weight_zr_o"
  "$PYTHON" scripts/run_phase3C_inline.py \
    "${COMMON[@]}" \
    "${resume_args[@]}" \
    --tag "$tag" \
    --weight-p "$weight_p" \
    --weight-si "$weight_si" \
    --weight-zr-total "$weight_zr_total" \
    --weight-zr-o "$weight_zr_o" \
    > "$out/run.log" 2>&1
  echo "DONE $label"
}

# Baseline is intentionally not rerun. Each case changes exactly one weight.
run_case p_low          80    2500  1.0  0.2
run_case p_high         8000  2500  1.0  0.2
run_case si_low         800   250   1.0  0.2
run_case si_high        800   25000 1.0  0.2
run_case zr_total_low   800   2500  0.1  0.2
run_case zr_total_high  800   2500  10.0 0.2
run_case zr_o_low       800   2500  1.0  0.02
run_case zr_o_high      800   2500  1.0  2.0

echo "ALL 8 COORDINATION-WEIGHT SENSITIVITY RUNS COMPLETED"
