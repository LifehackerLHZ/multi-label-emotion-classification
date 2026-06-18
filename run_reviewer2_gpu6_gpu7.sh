#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/ubuntu/data/emotion"
PY="/home/ubuntu/data/exp-code/tango/bin/python3"
export PYTHONPATH="${ROOT}/.deps:${PYTHONPATH:-}"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="${ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TMPDIR="${ROOT}/.tmp"
export TOKENIZERS_PARALLELISM="false"
mkdir -p "${ROOT}/outputs" "${ROOT}/logs" "${TMPDIR}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}"
ts="$(date +%Y%m%d_%H%M%S)"

run_augmented() {
  export CUDA_VISIBLE_DEVICES="6"
  out="${ROOT}/outputs/augmented_balanced_distilroberta_${ts}"
  log="${ROOT}/logs/augmented_gpu6_${ts}.log"
  echo "starting augmented baseline on GPU 6: ${out}" | tee -a "${log}"
  exec "${PY}" "${ROOT}/scripts/transformer_baseline_train.py" \
    --data_path "${ROOT}/data/balanced_emotion_dataset.csv" \
    --output_dir "${out}" \
    --model_name "distilroberta-base" \
    --max_length 64 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 128 \
    --learning_rate 2e-5 \
    --seed 42 \
    --optimize_thresholds \
    --fp16 >> "${log}" 2>&1
}

run_train_only() {
  export CUDA_VISIBLE_DEVICES="7"
  out="${ROOT}/outputs/train_only_augmented_distilroberta_${ts}"
  log="${ROOT}/logs/train_only_gpu7_${ts}.log"
  echo "starting train-only protocol on GPU 7: ${out}" | tee -a "${log}"
  exec "${PY}" "${ROOT}/scripts/train_distilroberta_from_splits.py" \
    --train_path "${ROOT}/data/train_only_splits/train.csv" \
    --val_path "${ROOT}/data/train_only_splits/val.csv" \
    --test_path "${ROOT}/data/train_only_splits/test.csv" \
    --output_dir "${out}" \
    --model_name "distilroberta-base" \
    --max_length 64 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 128 \
    --learning_rate 2e-5 \
    --seed 42 \
    --optimize_thresholds \
    --fp16 >> "${log}" 2>&1
}

(run_augmented) & echo $! > "${ROOT}/logs/augmented_gpu6.pid"
(run_train_only) & echo $! > "${ROOT}/logs/train_only_gpu7.pid"

echo "augmented_pid=$(cat "${ROOT}/logs/augmented_gpu6.pid")"
echo "train_only_pid=$(cat "${ROOT}/logs/train_only_gpu7.pid")"
echo "timestamp=${ts}"
