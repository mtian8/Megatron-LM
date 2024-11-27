export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "Hello world!"

PBSNODEFILE=$HOME/hostname.txt 

#!/bin/bash

# Initialize an empty array
nodes_array=()

# Read the file line by line
while IFS= read -r line; do
    # Skip empty lines
    if [[ -n "$line" ]]; then
        nodes_array+=("$line")
    fi
done < "$PBSNODEFILE"

# Print the array elements
# echo "Detected nodes:"
# for line in "${nodes_array[@]}"; do
#     echo "$line"
# done

local_hostname=$(hostname)
# echo "Local hostname: $local_hostname"

for i in "${!nodes_array[@]}"; do
    if [[ "${nodes_array[$i]}" == *"$local_hostname"* ]]; then
        NODEID=$i
        break
    # else: print two lines to debug
    # else
    #     echo "nodes_array[$i]: ${nodes_array[$i]}"
    #     echo "local_hostname: $local_hostname"
    #     echo "No match"
    fi
done

if [ -z "$NODEID" ]; then
    echo "Error: NODEID not found"
    echo $NODEID
    exit 1
fi


export HEAD_NODE_IP=${nodes_array[0]}
export MASTER_PORT=29500

export NNODES=`wc -l < $PBSNODEFILE`
export GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
# unify interface
export SLURM_GPUS_PER_NODE=$GPUS_PER_NODE
export SLURM_NNODES=$NNODES

# echo "Head Node IP: ${HEAD_NODE_IP}"
# echo "GPUS per node: ${SLURM_GPUS_PER_NODE}"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MEGATRON_DIR="$SCRIPT_DIR/.."

echo "In script: run_sft_distributed for rank $NODEID"
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real

export NCCL_CROSS_NIC=1
export NCCL_NET="AWS Libfabric"
export NVTE_DEBUG_LEVEL=0
# export NCCL_DEBUG=debug
export NCCL_SOCKET_IFNAME=hsn
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH


# llama2-7B 32k 4k sliding window
# 8M batch size
BATCH_TOKEN_SIZE_4M=$((32768*4*32))
BATCH_TOKEN_SIZE_8M=$((32768*4*32*2))
SEQ_LENGTH=$((32768*4*4))
LEN_ABBR=512k
MBS=1
GBS=$(( BATCH_TOKEN_SIZE_8M / ( MBS * SEQ_LENGTH ) ))
# Rotary Base
THETA=80000000
# Total Steps
TOT_STEPS=1000
# Data Path
SLIDING_WINDOW_SIZE=-1
SAVE_INTERVAL=100


# Base data paths
DATADIR=/projects/bdao/mtian8
WORKDIR=/work/nvme/bbvf/mtian8
MODELDIR=/u/mtian8/LLM/model

DATA=$DATADIR/slimpajama-per-source-length-upsample_${LEN_ABBR}
# DATA=/work/nvme/bbvf/mtian8/slimpajama-per-source-length-upsample_${LEN_ABBR}
# DATA=/projects/bccu/mtian8/mistral_slimpajama-per-source-length-upsample-32k
MODELDIR_TMP=$WORKDIR/LLM/tmp; mkdir -p $MODELDIR_TMP
WORK_DATADIR=$WORKDIR/LLM/data; mkdir -p $WORK_DATADIR
WORK_TENSORBOARD=$WORKDIR/LLM/Megatron-LM/log; mkdir -p $WORK_TENSORBOARD
WORK_WANDB=$WORKDIR/LLM/wandb; mkdir -p $WORK_WANDB
CHECKPOINT_DIR=${WORK_DATADIR}/checkpoint_long_${LEN_ABBR}_${SLIDING_WINDOW_SIZE}_fuse; mkdir -p $CHECKPOINT_DIR
# W&B setting
WANDB_PROJECT=llama2_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}
WANDB_EXP=llama2_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}_${GBS}_${THETA}
# WANDB_PROJECT=Mistral01_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}
# WANDB_EXP=Mistral01_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}_${GBS}_${THETA}

## Summary of changes:
# - Change --nproc_per_node to $SLURM_GPUS_PER_NODE
# - Change --nnodes to $SLURM_NNODES
# - Remove --node_rank $NODEID
# - Remove --master_addr $MASTER_ADDR --master_port $MASTER_PORT


# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=/work/nvme/bbvf/mtian8/LLM/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
MODEL=${MODELDIR}/MG_model/llama2_7B_tp_4_pp_1; echo "Model: $MODEL"
# MODEL=/work/nvme/bcbw/mtian8/model/MG_model/mistral01_7B_tp_4_pp_1; echo "Model: $MODEL" 
TOKENIZER_LLAMA2=/work/nvme/bcbw/mtian8/model/HF_model/Llama-2-7b-hf/tokenizer.model
TOKENIZER_MISTRAL01=/work/nvme/bcbw/mtian8/model/HF_model/Mistral-7B-v0.1
TOKENIZER_TYPE=Llama2Tokenizer
CONTAINER_HOME=/sw/user/NGC_containers
SIF_FILE=pytorch_24.09-py3.sif
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_DEVICE_MAX_CONNECTIONS=1 
export TMPDIR=$MODELDIR_TMP
# export NVTE_FLASH_ATTN=1
# export NVTE_FUSED_ATTN=0

    # --env NVTE_FLASH_ATTN=${NVTE_FLASH_ATTN} \
    # --env NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN} \
apptainer exec --nv \
    --env HEAD_NODE_IP=${HEAD_NODE_IP} \
    --env SLURM_NNODES=${SLURM_NNODES} \
    --env MASTER_PORT=${MASTER_PORT} \
    --env SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE} \
    --env SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR} \
    --bind /work\
    --bind /projects \
    ${CONTAINER_HOME}/${SIF_FILE} \
    /usr/local/bin/torchrun \
        --nproc_per_node ${SLURM_GPUS_PER_NODE} \
        --nnodes ${SLURM_NNODES} \
        --node_rank $NODEID --master_addr $HEAD_NODE_IP --master_port $MASTER_PORT \
        /u/mtian8/LLM/Megatron-LM/sft_gpt.py \
            --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
            --sequence-parallel \
            --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
            --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
            --seq-length $SEQ_LENGTH  --max-position-embeddings $SEQ_LENGTH --rotary-base $THETA \
            --micro-batch-size $MBS --global-batch-size $GBS --split 100,0,0 \
            --lr 2.0e-5 --train-iters $TOT_STEPS --lr-decay-style constant \
            --clip-grad 0.0 --bf16 --use-flash-attn \
            --norm-epsilon 1e-5 --no-masked-softmax-fusion \
            --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
            --exit-on-missing-checkpoint --dataloader-type single \
            --data-path $DATA \
            --save-interval $SAVE_INTERVAL --eval-interval 500 \
            --window-size $SLIDING_WINDOW_SIZE 0 \
            --eval-iters 10000 --tokenizer-type $TOKENIZER_TYPE --attention-softmax-in-fp32 \
            --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
            --tokenizer-model $TOKENIZER_LLAMA2 \
            --load $MODEL \
            --save ${CHECKPOINT_DIR}  \
            --tensorboard-dir "${WORK_TENSORBOARD}" --tensorboard-queue-size 5 --wandb-project $WANDB_PROJECT \
            --wandb-exp-name $WANDB_EXP --wandb-save-dir "${WORK_WANDB}"

            # --weight-decay 0.1 --min-lr 1.0e-6 \