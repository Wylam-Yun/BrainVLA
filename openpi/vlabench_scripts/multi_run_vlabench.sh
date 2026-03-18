#!/bin/bash

export HF_HOME=~/.cache/huggingface
export MUJOCO_GL=egl
export VLABENCH_ROOT=/data1/haodong2/weilin/red_bird/VLABench/VLABench
export PYTHONPATH="${PYTHONPATH}:/data1/haodong2/weilin/red_bird/VLABench"
NUM_GPUS=1
NUM_TRIALS=50
MAX_PROCS_PER_GPU=1
MAX_PROCS=$((NUM_GPUS * MAX_PROCS_PER_GPU))
TRACK_CONFIG_DIR="${VLABENCH_ROOT}/configs/evaluation/tracks"

# --------------- 解析参数 ---------------
SAVE_DIR=""
TRACK_OPT=""
TASK_OPT=""

if [ "$#" -lt 1 ]; then
    usage
fi
SAVE_DIR=$1
shift 1

# 解析可选参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --track)
            TRACK_OPT="$2"
            shift 2
            ;;
        --task)
            TASK_OPT="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            usage
            ;;
    esac
done

# --------------- 初始化全量数据 ---------------
ALL_TRACKS=(track_1_in_distribution track_2_cross_category track_3_common_sense track_4_semantic_instruction track_6_unseen_texture)
ALL_TASKS=(add_condiment insert_flower select_book select_drink select_chemistry_tube select_mahjong select_toy select_fruit select_painting select_poker select_nth_largest_poker select_unique_type_mahjong)

# --------------- 处理track与task ---------------
if [[ -n "$TRACK_OPT" ]]; then
    IFS=',' read -ra TRACKS <<< "$TRACK_OPT"
else
    TRACKS=("${ALL_TRACKS[@]}")
fi

job_idx=0
BASE_PORT=8009


for TRACK in "${TRACKS[@]}"; do
    if [[ -n "$TASK_OPT" ]]; then
        IFS=',' read -ra TASKS <<< "$TASK_OPT"
    else
        TRACK_FILE="${TRACK_CONFIG_DIR}/${TRACK}.json"
        if [[ -f "$TRACK_FILE" ]]; then
            mapfile -t TASKS < <(
                python - <<'PY' "$TRACK_FILE"
import json
import sys

with open(sys.argv[1], "r") as f:
    data = json.load(f)
for k in data.keys():
    print(k)
PY
            )
        else
            TASKS=("${ALL_TASKS[@]}")
        fi
    fi

    for TASK in "${TASKS[@]}"; do
        GPU_ID=2
        NOTE="${CKPT_BASENAME}"
        port=8009
        echo "[INFO] Submit JOB: ckpt=$CKPT, track=$TRACK, task=$TASK, gpu=$GPU_ID"

        CUDA_VISIBLE_DEVICES=$GPU_ID MUJOCO_EGL_DEVICE_ID=$GPU_ID \
            uv run examples/vlabench/eval.py \
            --args.port $port \
            --args.eval_track $TRACK \
            --args.tasks $TASK \
            --args.n-episode $NUM_TRIALS \
            --args.save_dir $SAVE_DIR &

        job_idx=$((job_idx+1))
        while [ $(jobs -rp | wc -l) -ge $MAX_PROCS ]; do
            sleep 2
            wait -n
        done
    done
done

wait

# python examples/vlabench/summarize.py
# python scripts/gpu_runner.py

