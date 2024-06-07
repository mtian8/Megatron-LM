#!/bin/bash

# Run SFT on a pretrained multimodal model.

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

DATETIME=`date +'%y-%m-%d-%H-%M-%S'`
MODEL_NAME="mcore-llava-sft-${DATETIME}"

# Check that the user has set an output path for model checkpoints.
if [[ -z $WORKSPACE ]]; then
    echo "Please set WORKSPACE for storing your model checkpoints."
    exit 1
fi

SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"

if [[ -z $LOAD_NAME ]]; then
    echo "Please set LOAD_NAME for input model name."
    exit 1
fi

if [[ -z $TOKENIZER_MODEL ]]; then
    echo "Please set TOKENIZER_MODEL for tokenizer model name."
    exit 1
fi

CHECKPOINT_DIR="${WORKSPACE}/output/${LOAD_NAME}/checkpoints"

DATA_TRAIN="${SOURCE}/examples/multimodal/sft_dataset.yaml"
DATA_VALID="${SOURCE}/examples/multimodal/sft_dataset.yaml"

DEBUG=0
if [[ $DEBUG -eq 1 ]]; then
    BZ=8
    NW=1
    LI=1
    HD=0.0
    EXTRA_ARGS=""
else
    BZ=128
    NW=1
    LI=10
    HD=0.1
    EXTRA_ARGS=""
fi

OPTIONS=" \
    --num-workers ${NW} \
    --use-flash-attn \
    --apply-layernorm-1p \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --rotary-percent 0.5 \
    --squared-relu \
    --attention-dropout 0.0 \
    --hidden-dropout ${HD} \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 1024 \
    --max-position-embeddings 4096 \
    --train-samples 665000 \
    --micro-batch-size 1 \
    --global-batch-size ${BZ} \
    --lr-decay-samples 25600000 \
    --lr-warmup-samples 83200 \
    --lr 1e-6 \
    --min-lr 1e-7 \
    --lr-decay-style cosine \
    --log-interval ${LI} \
    --eval-iters 10 \
    --eval-interval 1000 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model ${WORKSPACE}/${TOKENIZER_MODEL} \
    --data-path ${DATA_TRAIN} \
    --valid-path ${DATA_VALID} \
    --prompt-path ${SOURCE}/examples/multimodal/manual_prompts.json \
    --save-interval 1000 \
    --exit-duration-in-mins 230 \
    --save ${FINETUNE_DIR} \
    --load ${CHECKPOINT_DIR} \
    --split 100,0,0 \
    --clip-grad 0.5 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --eod-mask-loss \
    --finetune \
    --freeze-ViT \
    --patch-dim 14 \
    --img-h 336 \
    --img-w 336 \
    --dataloader-type external \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --language-model-type=8b \
    --disable-vision-class-token \
    ${EXTRA_ARGS} \
    --distributed-timeout-minutes 60 \
"

export NVTE_APPLY_QK_LAYER_SCALING=1

# MULTI GPU
torchrun --nproc_per_node 8 examples/multimodal/train.py ${OPTIONS}
