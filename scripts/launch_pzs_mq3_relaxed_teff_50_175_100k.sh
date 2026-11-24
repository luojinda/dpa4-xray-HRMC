#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/dpa4-xray-HRMC

export DPA4_MODEL_PATH=models/PZS_new_20261121/DPA4_PZS_new_LoRA.pt2
export PYTHONPATH=src

PYTHON=/root/miniconda3/envs/dpa4_torch211/bin/python
BASE_TAG=PZS_MQ_3_relaxed
BASE_OUT=results/PZS_phase3/phase3C/${BASE_TAG}

run_temperature() {
  local temperature="$1"
  local tag="${BASE_TAG}/validation_100k_seed101_T${temperature}_d0p1_r1p4_10p0A_lora_bounded_dynamic_p_total"
  local out="results/PZS_phase3/phase3C/${tag}"
  mkdir -p "$out"

  if [[ -f "$out/summary.json" ]]; then
    echo "SKIP completed T=${temperature} K"
    return
  fi

  local resume_args=()
  if [[ -f "$out/checkpoint_latest.npz" ]]; then
    resume_args=(--resume)
  fi

  "$PYTHON" scripts/run_phase3C_inline.py \
    --attempts 100000 \
    --temperature "$temperature" \
    --seed 101 \
    --tag "$tag" \
    --move-max 0.1 \
    --fit-rmin 1.4 \
    --fit-rmax 10.0 \
    --structure results/PZS_phase3/relaxations/PZS_MQ_3_lora_fixed_cell/relaxed_structure.cif \
    --checkpoint-interval 10000 \
    --p-coordination-mode P-total \
    --p-cl-cutoff 2.3 \
    --scale-mode bounded_dynamic \
    --weight-p 800 \
    --weight-si 2500 \
    --weight-zr-total 1.0 \
    --weight-zr-o 0.2 \
    "${resume_args[@]}" \
    > "$out/run.log" 2>&1

  echo "DONE T=${temperature} K"
}

mkdir -p "$BASE_OUT"
run_temperature 50
run_temperature 175
