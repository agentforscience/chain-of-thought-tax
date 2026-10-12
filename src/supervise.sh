#!/bin/bash
# Run one model to completion on shared GPUs: wait until some GPU has enough free memory, run the
# (resumable) inference script there, and retry if it dies.
# Usage: bash src/supervise.sh <model> <need_free_MiB> <batch_size>
MODEL=$1; NEED=$2; BS=$3; NAME=$(basename $MODEL)
attempt() { # extra args for run_inference.py
  while true; do
    read GPU FREE < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr -d ',' | sort -k2 -n -r | head -1)
    if [ "$FREE" -ge "$NEED" ]; then
      echo "$(date +%T) $NAME: launching on GPU $GPU ($FREE MiB free): $*" >> logs/supervise.log
      CUDA_VISIBLE_DEVICES=$GPU python src/run_inference.py --model $MODEL --batch-size $BS "$@" >> logs/run_${NAME}.log 2>&1 && return 0
      echo "$(date +%T) $NAME: run failed, retrying" >> logs/supervise.log
    fi
    sleep 45
  done
}
attempt --conditions direct cot
attempt --conditions cot2 --limit 100
echo "$(date +%T) $NAME: ALL DONE" >> logs/supervise.log
