#!/bin/bash
# Configure environment variables first
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.sh"

CONFIG_NAME=$1
CKPT=$2

###### MODIFY THESE ######
CONFIG=$CONFIG_NAME
##########################

ENV="VLABENCH"
CHECKPOINT="policy:checkpoint"
BASE_PORT=8009
NUM_GPUS=1

echo "Launching policy servers on ${NUM_GPUS} GPUs..."
echo "Using Checkpoints ${CKPT}"
# 启动每个GPU上的服务
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    port=$((BASE_PORT + gpu_id))
    
    echo "Starting policy server on GPU ${gpu_id}, port ${port}..."
    
    XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=2 uv run scripts/serve_policy.py \
        --port ${port} \
        --env ${ENV} ${CHECKPOINT} \
        --policy.config=${CONFIG} \
        --policy.dir=${CKPT} &
    
    sleep 2
done

echo "All policy servers started!"
echo "Ports: ${BASE_PORT}-$((BASE_PORT + NUM_GPUS - 1))"

wait