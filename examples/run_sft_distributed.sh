# llama3 cnt pretrain long context (sliding window)
set -ex

export CUDA_DEVICE_MAX_CONNECTIONS=1

export GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# Change for multinode config
PBSNODEFILE=$HOME/hostname.txt
export MASTER_ADDR=$(head -n 1 $PBSNODEFILE)
export MASTER_PORT=10086
NNODES=`wc -l < $PBSNODEFILE`
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
NODEID=$1 #RANDOM
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MEGATRON_DIR="$SCRIPT_DIR/.."

# torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$PALS_NODEID --master_addr=$MASTER_ADDR \
#        --master_port=$MASTER_PORT test_multinode.py --backend=nccl
echo "In script: run_sft_distributed for rank $NODEID"
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real

# llama2-7B 32k
TMPDIR=/projects/bcdz/mtian8/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
/u/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1  --recompute-granularity full --recompute-method uniform --recompute-num-layers 4 \
--tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 \
--use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
--seq-length 32768 --max-position-embeddings 32768 --rotary-base 1000000 \
--micro-batch-size 2 --global-batch-size 256 --split 100,0,0 \
--lr 2.0e-5 --train-iters 506 --lr-decay-style constant \
--lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
--norm-epsilon 1e-5 --no-masked-softmax-fusion \
--optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
--exit-on-missing-checkpoint --dataloader-type cyclic \
--data-path /projects/bdao/mtian8/slimpajama-per-source-length-upsample_32k \
--save-interval 10 --eval-interval 500 \
--eval-iters 10 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
--untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
--tokenizer-model /u/mtian8/LLM/model/HF_model/Llama-2-7b-hf/tokenizer.model \
--load /u/mtian8/LLM/model/MG_model/llama2_7B_tp_4_pp_1 \
--save /u/mtian8/LLM/data/checkpoint_long_32k \
--tensorboard-dir "/u/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'llama2_7B_32k ' \
--wandb-exp-name 'llama2_7B_32k ' --wandb-save-dir "/u/mtian8/LLM/wandb" \
2>&1 | tee out.out 

# llama2-7B 32k sliding window 
# TMPDIR=/projects/bcdz/mtian8/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /u/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1  --recompute-granularity full --recompute-method uniform --recompute-num-layers 4 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length 32768 --max-position-embeddings 32768 --rotary-base 1000000 \
# --micro-batch-size 1 --global-batch-size 256 --split 100,0,0 \
# --lr 2.0e-5 --train-iters 506 --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path /projects/bdao/mtian8/slimpajama-per-source-length-upsample_32k \
# --save-interval 10 --eval-interval 500  --window-size 4096 0 \
# --eval-iters 10 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /u/mtian8/LLM/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /u/mtian8/LLM/model/MG_model/llama2_7B_tp_4_pp_1 \
# --save /u/mtian8/LLM/data/checkpoint_long_sliding_window_4k \
# --tensorboard-dir "/u/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'llama2_7B_sliding_window ' \
# --wandb-exp-name 'llama2_7B_sliding_window ' --wandb-save-dir "/u/mtian8/LLM/wandb" \
# 2>&1 | tee out.out  

# llama2-70B 32k 
# TMPDIR=/projects/bcdz/mtian8/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /u/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1  --recompute-granularity full --recompute-method block --recompute-num-layers 8 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 8 \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length 32768 --max-position-embeddings 32768 --rotary-base 1000000 \
# --micro-batch-size 1 --global-batch-size 8 --split 100,0,0 \
# --lr 2.0e-5 --train-iters 506 --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path /projects/bdao/mtian8/slimpajama-per-source-length-upsample_32k \
# --save-interval 10 --eval-interval 500 \
# --eval-iters 10 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /u/mtian8/LLM/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /u/mtian8/LLM/model/MG_model/llama2_70B_tp_4_pp_8 \
# --save /u/mtian8/LLM/data/checkpoint_long_70B \
# --tensorboard-dir "/u/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'llama2_70B_trial' \
# --wandb-exp-name 'llama2_70B ' --wandb-save-dir "/u/mtian8/LLM/wandb" \
# 2>&1 | tee out.out  


