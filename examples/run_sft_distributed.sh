# llama3 cnt pretrain long context (sliding window)
set -ex

export CUDA_DEVICE_MAX_CONNECTIONS=1

export GPUS_PER_NODE=8
# Change for multinode config
PBSNODEFILE=/home/mtian8/pbs_nodefile
export MASTER_ADDR=$(head -n 1 $PBSNODEFILE)
export MASTER_PORT=6000
NNODES=`wc -l < $PBSNODEFILE`
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
NODEID=$1 #RANDOM

# torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$PALS_NODEID --master_addr=$MASTER_ADDR \
#        --master_port=$MASTER_PORT test_multinode.py --backend=nccl
{(CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
Megatron-LM/sft_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 4 \
--tensor-model-parallel-size 8 --pipeline-model-parallel-size 1 \
--use-checkpoint-args --no-load-optim --no-load-rng \
--seq-length 32 --max-position-embeddings 32 --rotary-base 8000000 \
--micro-batch-size 1 --global-batch-size 128 --split 100,0,0 \
--lr 2.0e-5 --train-iters 506 --lr-decay-style constant \
--lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
--norm-epsilon 1e-5 --no-masked-softmax-fusion \
--optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
--exit-on-missing-checkpoint --dataloader-type cyclic \
--data-path /home/mtian8/mtian8/tpc/mtian8/data/slimpajama_packed_64k_5b_per_source_down_sample_0.1 \
--save-interval 1 --eval-interval 500  --window-size 4096 0 \
--eval-iters 10 --tokenizer-type Llama3Tokenizer --attention-softmax-in-fp32 \
--untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
--tokenizer-model /home/mtian8/mtian8/tpc/mtian8/model/HF_model/Meta-Llama-3-8B/original/tokenizer.model \
--load /home/mtian8/mtian8/tpc/mtian8/model/MG_model/llama3_8B_tp_8_pp_1 \
--save /home/mtian8/mtian8/tpc/mtian8/data/checkpoint_long_sliding_window_4k \
--tensorboard-dir 'Megatron-LM/log' --tensorboard-queue-size 5 --wandb-project 'llama3_8B_sliding_window_4k ' \
--wandb-exp-name 'llama3_8B_sliding_window_4k ' --wandb-save-dir '/home/mtian8/mtian8/tpc/mtian8/wandb' \
2>&1 | tee out.out )&}


echo "HELLO"

