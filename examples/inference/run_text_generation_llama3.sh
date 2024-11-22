#!/bin/bash
# This example will start serving the Llama3-8B model
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_APPLY_QK_LAYER_SCALING=0
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real
DISTRIBUTED_ARGS="--nproc_per_node 4 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr 0.0.0.0 \
                  --master_port 6000"

# Ensure CHECKPOINT and TOKENIZER_MODEL are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: You must provide CHECKPOINT and TOKENIZER_MODEL as command-line arguments."
  echo "Usage: $0 /path/to/checkpoint /path/to/tokenizer_model"
  exit 1
fi

# Assign command-line arguments to variables
CHECKPOINT=$1
TOKENIZER_MODEL=$2
ROPE_BASE=5000000
if [ -z "$3" ]; then
  echo "Using default ROPE_BASE: $ROPE_BASE"
else
  ROPE_BASE=$3
fi
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
export TORCHINDUCTOR_CACHE_DIR=/projects/bbzy/yufengd4/tmp
#     --window-size 1024 0 \
# pip install flask-restful
echo "Running llama3-8b model with checkpoint: ${CHECKPOINT} and tokenizer model: ${TOKENIZER_MODEL}"
torchrun $DISTRIBUTED_ARGS tools/run_text_generation_server.py   \
      --use-checkpoint-args \
      --disable-bias-linear \
      --tokenizer-type HuggingFaceTokenizer \
      --tokenizer-model ${TOKENIZER_MODEL} \
      --transformer-impl transformer_engine \
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
      --fp16  \
      --micro-batch-size 1  \
      --seq-length 131072
