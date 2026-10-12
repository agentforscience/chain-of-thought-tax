#!/bin/bash
# Launch the full grid. Usage: bash src/run_all.sh <gpu_for_14B> <gpu_for_smaller_models>
# Main grid: direct + cot on all items; robustness: cot2 on the first 100 items of each task.
G_BIG=${1:-3}; G_SMALL=${2:-2}
run() { # gpu model batch_size
  name=$(basename $2)
  CUDA_VISIBLE_DEVICES=$1 python src/run_inference.py --model $2 --conditions direct cot --batch-size $3 > logs/run_${name}.log 2>&1 \
  && CUDA_VISIBLE_DEVICES=$1 python src/run_inference.py --model $2 --conditions cot2 --limit 100 --batch-size $3 > logs/run_${name}_cot2.log 2>&1
}
run $G_BIG Qwen/Qwen2.5-14B-Instruct 96 &
run $G_SMALL Qwen/Qwen2.5-7B-Instruct 96 &
run $G_SMALL Qwen/Qwen2.5-3B-Instruct 96 &
run $G_SMALL Qwen/Qwen2.5-1.5B-Instruct 96 &
run $G_SMALL Qwen/Qwen2.5-0.5B-Instruct 96 &
wait
