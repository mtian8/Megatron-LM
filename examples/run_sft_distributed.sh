# llama3 cnt pretrain long context (sliding window)
set -ex

export CUDA_DEVICE_MAX_CONNECTIONS=1

export GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# Change for multinode config
PBSNODEFILE=$HOME/hostname.txt
export MASTER_ADDR=$(head -n 1 $PBSNODEFILE)
export MASTER_PORT=6000
NNODES=`wc -l < $PBSNODEFILE`
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
NODEID=$1 #RANDOM
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MEGATRON_DIR="$SCRIPT_DIR/.."

# torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$PALS_NODEID --master_addr=$MASTER_ADDR \
#        --master_port=$MASTER_PORT test_multinode.py --backend=nccl
echo "In script: run_sft_distributed for rank $NODEID"

CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
$MEGATRON_DIR/sft_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 4 \
--tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 \
--use-checkpoint-args --no-load-optim --no-load-rng \
--seq-length 32 --max-position-embeddings 32 --rotary-base 8000000 \
--micro-batch-size 1 --global-batch-size 128 --split 100,0,0 \
--lr 2.0e-5 --train-iters 506 --lr-decay-style constant \
--lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
--norm-epsilon 1e-5 --no-masked-softmax-fusion \
--optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
--exit-on-missing-checkpoint --dataloader-type cyclic \
--data-path $MEGATRON_DIR/../datasets/slimpajama-per-source-length-upsample \
--save-interval 1 --eval-interval 500  --window-size 4096 0 \
--eval-iters 10 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
--untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
--tokenizer-model $MEGATRON_DIR/../models/Llama-2-7b-hf/tokenizer.model \
--load $MEGATRON_DIR/../models/MG_llama2_7B_tp_4_pp_2 \
--save $MEGATRON_DIR/../models/checkpoint_long_sliding_window_4k \
--tensorboard-dir "$MEGATRON_DIR/log" --tensorboard-queue-size 5 --wandb-project 'llama2_7B_sliding_window_4k ' \
--wandb-exp-name 'llama2_7B_sliding_window_4k ' --wandb-save-dir "$MEGATRON_DIR/../wandb" \
2>&1 | tee out.out 


echo "HELLO"

