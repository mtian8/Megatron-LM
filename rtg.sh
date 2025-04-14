#!/bin/bash
# This example will start serving the Llama3-8B model
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_APPLY_QK_LAYER_SCALING=0
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real
export TORCH_NCCL_ENABLE_MONITORING=0
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DEVICE=$6
if [ -z "$6" ]; then
 DEVICE=0
fi
export CUDA_VISIBLE_DEVICES=$DEVICE

DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr 0.0.0.0 \
                  --master_port 600$DEVICE"

# Ensure CHECKPOINT and TOKENIZER_MODEL are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: You must provide CHECKPOINT and TOKENIZER_MODEL as command-line arguments."
  echo "Usage: $0 /path/to/checkpoint /path/to/tokenizer_model"
  exit 1
fi

echo "List of arguments: \"$1\" \"$2\" \"$3\" \"$4\" \"$5\" \"$6\""

# Assign command-line arguments to variables
CHECKPOINT=$1
TOKENIZER_MODEL=$2
ROPE_BASE=500000
if [ -z "$3" ]; then
  echo "Using default ROPE_BASE: $ROPE_BASE"
else
  ROPE_BASE=$3
fi
MODEL_TYPE=$4
SPARSE=$5
if [ -z "$5" ]; then
  SPARSE="te"
fi
PORT=$7
if [ -z "$7" ]; then
  PORT=500$DEVICE
fi
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
export TORCHINDUCTOR_CACHE_DIR=/projects/bdtq/yufengd4/tmp
#     --window-size 1024 0 \
# pip install flask-restful
if [[ $MODEL_TYPE == "llama3" ]]; then
echo "Notice: This script runs the llama3 model with ROPE scaling!"
SCRIPT="tools/run_llama3_text_generation_server.py"
else
echo "Normal model"
SCRIPT="tools/run_text_generation_server.py"
fi
if [[ $SPARSE == "te" ]]; then
TIMPL="transformer_engine"
else
TIMPL="dkernel"
fi
echo "Running model with checkpoint: ${CHECKPOINT} and tokenizer model: ${TOKENIZER_MODEL}"
torchrun $DISTRIBUTED_ARGS $SCRIPT   \
      --port $PORT \
      --distributed-timeout-minutes 1440 \
      --use-checkpoint-args \
      --disable-bias-linear \
      --tokenizer-type HuggingFaceTokenizer \
      --tokenizer-model ${TOKENIZER_MODEL} \
      --transformer-impl $TIMPL \
      --normalization RMSNorm \
      --group-query-attention \
      --no-masked-softmax-fusion \
      --attention-softmax-in-fp32 \
      --attention-dropout 0.0 \
      --hidden-dropout 0.0 \
      --untie-embeddings-and-output-weights \
      --position-embedding-type rope \
      --rotary-percent 1.0 \
      --rotary-base $ROPE_BASE \
      --use-rotary-position-embeddings \
      --swiglu \
      --load ${CHECKPOINT}  \
      --max-position-embeddings 131072  \
      --bf16  \
      --micro-batch-size 1  \
      --seq-length 131072
