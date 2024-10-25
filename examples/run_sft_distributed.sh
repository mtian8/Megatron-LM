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

# rdma
# torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$PALS_NODEID --master_addr=$MASTER_ADDR \
#        --master_port=$MASTER_PORT test_multinode.py --backend=nccl
echo "In script: run_sft_distributed for rank $NODEID"
export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real
# export NCCL_NET_GDR_LEVEL=PHB
# export NCCL_CROSS_NIC=1
# export NCCL_COLLNET_ENABLE=1
# export NCCL_NET="AWS Libfabric"
# export FI_CXI_DISABLE_HOST_REGISTER=1
# export FI_MR_CACHE_MONITOR=userfaultfd
# export FI_CXI_DEFAULT_CQ_SIZE=131072

# export PATH="/work/nvme/bcbw/mtian8/libfabric/1.15.2.0/bin:$PATH"
# export LD_LIBRARY_PATH="/work/nvme/bcbw/mtian8/libfabric/1.15.2.0/lib64:$LD_LIBRARY_PATH"
# export MANPATH="/work/nvme/bcbw/mtian8/libfabric/1.15.2.0/share/man:$MANPATH"
# export PKG_CONFIG_PATH="/work/nvme/bcbw/mtian8/libfabric/1.15.2.0/lib64/pkgconfig:$PKG_CONFIG_PATH"
# export LD_LIBRARY_PATH="/work/nvme/bcbw/mtian8/nccl/lib:$LD_LIBRARY_PATH"
# export LD_LIBRARY_PATH="/work/nvme/bcbw/mtian8/aws-ofi.1.6.0/lib:$LD_LIBRARY_PATH"
# export FI_MR_CACHE_MONITOR="userfaultfd"
# export FI_CXI_DISABLE_HOST_REGISTER="1"
# export FI_CXI_DEFAULT_CQ_SIZE="131072"
# export NCCL_CROSS_NIC="1"
# export NCCL_SOCKET_IFNAME="hsn"
# export NCCL_NET="AWS Libfabric"
# export NCCL_DEBUG="info"
# export C_INCLUDE_PATH="/work/nvme/bcbw/mtian8/nccl/include:$C_INCLUDE_PATH"
# export CPLUS_INCLUDE_PATH="/work/nvme/bcbw/mtian8/nccl/include:$CPLUS_INCLUDE_PATH"
# export INCLUDE="/work/nvme/bcbw/mtian8/nccl/include:$INCLUDE"
# export LIBRARY_PATH="/work/nvme/bcbw/mtian8/nccl/lib:$LIBRARY_PATH"
# export CMAKE_PREFIX="/work/nvme/bcbw/mtian8/nccl/."
# export CMAKE_PREFIX_PATH="/work/nvme/bcbw/mtian8/nccl/.:$CMAKE_PREFIX_PATH"
# export PKG_CONFIG_PATH="/work/nvme/bcbw/mtian8/nccl/lib/pkgconfig:$PKG_CONFIG_PATH"
# export NCCL_HOME="/work/nvme/bcbw/mtian8/nccl/"

# export NCCL_CROSS_NIC=1
# export NCCL_NET="AWS Libfabric"
# export NCCL_DEBUG=info
# export NCCL_SOCKET_IFNAME=hsn
# export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

# Codestral 16k resume
# TMPDIR=/projects/bcdz/mtian8/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /u/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 4 --sequence-parallel \
# --use-checkpoint-args --logging-level 30 --log-throughput --log-progress \
# --seq-length 16384 --max-position-embeddings 16384 --rotary-base 1000000 \
# --micro-batch-size 4 --global-batch-size 512 --split 100,0,0 \
# --lr 2.0e-5 --train-iters 600 --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path /projects/bbtf/mtian8/slimpajama-per-source-length-upsample_16k \
# --save-interval 5 --eval-interval 500 \
# --eval-iters 10 --tokenizer-type MistralTokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /work/nvme/bcbw/mtian8/model/HF_model/Codestral-22B-v0.1/tokenizer.model \
# --load /u/mtian8/LLM/data/checkpoint_codestral_16k \
# --save /u/mtian8/LLM/data/checkpoint_codestral_16k \
# --tensorboard-dir "/u/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'codestral_22B_16k' \
# --wandb-exp-name 'codestral_22B_16k' --wandb-save-dir "/u/mtian8/LLM/wandb" \

# Codestral 32k
# TMPDIR=/projects/bcdz/mtian8/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /u/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 8  --sequence-parallel \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length 32768 --max-position-embeddings 32768 --rotary-base 1000000 \
# --micro-batch-size 1 --global-batch-size 8 --split 100,0,0 \
# --lr 2.0e-5 --train-iters 600 --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path /projects/bdao/mtian8/slimpajama-per-source-length-upsample_32k \
# --save-interval 5 --eval-interval 500 \
# --eval-iters 10 --tokenizer-type MistralTokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /work/nvme/bcbw/mtian8/model/HF_model/Codestral-22B-v0.1/tokenizer.model \
# --load /u/mtian8/LLM/model/MG_model/codestral_22B_tp_4_pp_8 \
# --save /u/mtian8/LLM/data/checkpoint_codestral_16k \
# --tensorboard-dir "/u/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'codestral_22B_16k' \
# --wandb-exp-name 'codestral_22B_16k' --wandb-save-dir "/u/mtian8/LLM/wandb" \

# llama2-7B 64k resume
# TMPDIR=/projects/bcdz/mtian8/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /u/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1  --recompute-granularity full --recompute-method uniform --recompute-num-layers 8 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 4 --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --logging-level 30 --log-throughput --log-progress \
# --seq-length 65536 --max-position-embeddings 65536 --rotary-base 1000000 \
# --micro-batch-size 1 --global-batch-size 1024 --split 100,0,0 \
# --lr 2.0e-5 --train-iters 75 --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path /projects/bdao/mtian8/slimpajama-per-source-length-upsample_64k \
# --save-interval 1 --eval-interval 500 \
# --eval-iters 10 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model  /work/nvme/bcbw/mtian8/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /u/mtian8/LLM/data/checkpoint_long_64k \
# --save /u/mtian8/LLM/data/checkpoint_long_64k \
# --tensorboard-dir "/u/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'llama2_7B_64k ' \
# --wandb-exp-name 'llama2_7B_64k ' --wandb-save-dir "/u/mtian8/LLM/wandb" \

# llama2-7B 64k
# TMPDIR=/projects/bcdz/mtian8/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /u/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1  --recompute-granularity full --recompute-method uniform --recompute-num-layers 8 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 4 --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length 65536 --max-position-embeddings 65536 --rotary-base 1000000 \
# --micro-batch-size 1 --global-batch-size 8 --split 100,0,0 \
# --lr 2.0e-5 --train-iters 75 --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path /projects/bdao/mtian8/slimpajama-per-source-length-upsample_64k \
# --save-interval 1 --eval-interval 500 \
# --eval-iters 10 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model  /work/nvme/bcbw/mtian8/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load  /work/nvme/bcbw/mtian8/model/MG_model/llama2_7B_tp_4_pp_4 \
# --save /u/mtian8/LLM/data/checkpoint_long_64k \
# --tensorboard-dir "/u/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'llama2_7B_64k ' \
# --wandb-exp-name 'llama2_7B_64k ' --wandb-save-dir "/u/mtian8/LLM/wandb" \

# # llama2-7B 128k
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=/work/nvme/bbvf/mtian8/LLM/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /work/nvme/bbvf/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length 131072 --max-position-embeddings 131072 --rotary-base 100000000 \
# --micro-batch-size 2 --global-batch-size 64 --split 100,0,0 \
# --lr 2.0e-5 --train-iters 625 --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path /projects/bdao/mtian8/slimpajama-per-source-length-upsample_128k \
# --save-interval 20 --eval-interval 500 \
# --eval-iters 1000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /work/nvme/bcbw/mtian8/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /work/nvme/bcbw/mtian8/model/MG_model/llama2_7B_tp_4_pp_1 \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_128k \
# --tensorboard-dir "/work/nvme/bbvf/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'llama2_7B_128k_nosw' \
# --wandb-exp-name 'llama2_7B_128k' --wandb-save-dir "/work/nvme/bbvf/mtian8/LLM/wandb" \

# llama2-7B 128k resume
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=/work/nvme/bbvf/mtian8/LLM/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /work/nvme/bbvf/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --logging-level 30 --log-throughput --log-progress \
# --seq-length 131072 --max-position-embeddings 131072 --rotary-base 100000000 \
# --micro-batch-size 2 --global-batch-size 64 --split 100,0,0 \
# --lr 2.0e-5 --train-iters 625 --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path /projects/bdao/mtian8/slimpajama-per-source-length-upsample_128k \
# --save-interval 20 --eval-interval 500 \
# --eval-iters 1000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /work/nvme/bcbw/mtian8/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_128k \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_128k \
# --tensorboard-dir "/work/nvme/bbvf/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'llama2_7B_128k_nosw' \
# --wandb-exp-name 'llama2_7B_128k' --wandb-save-dir "/work/nvme/bbvf/mtian8/LLM/wandb" \

# bash launch_multinode_sft.sh 2>&1 | tee out.out
# bash multinode_kill.sh
# 8M batch size
# SEQ_LENGTH=32768
# LEN_ABBR=32k
# MBS=1
# GBS=256
# # Rotary Base
# THETA=1000000
# # Total Steps
# TOT_STEPS=506
# # Data Path
# DATA=/projects/bdao/mtian8/slimpajama-per-source-length-upsample_${LEN_ABBR}
# SLIDING_WINDOW_SIZE=4096
# SAVE_INTERVAL=100
# # W&B setting
# WANDB_PROJECT=llama2_7B_${LEN_ABBR}_SW
# WANDB_EXP=llama2_7B_${LEN_ABBR}_SW_${GBS}_${THETA}

# # llama2-7B 32k_sliding_window
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=/work/nvme/bbvf/mtian8/LLM/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /work/nvme/bbvf/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length $SEQ_LENGTH  --max-position-embeddings $SEQ_LENGTH --rotary-base $THETA \
# --micro-batch-size $MBS --global-batch-size $GBS --split 100,0,0 \
# --lr 2.0e-5 --train-iters $TOT_STEPS --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path $DATA \
# --save-interval $SAVE_INTERVAL --eval-interval 500 \
# --window-size $SLIDING_WINDOW_SIZE 0 \
# --eval-iters 1000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /work/nvme/bcbw/mtian8/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /work/nvme/bcbw/mtian8/model/MG_model/llama2_7B_tp_4_pp_1 \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_${LEN_ABBR} \
# --tensorboard-dir "/work/nvme/bbvf/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project $WANDB_PROJECT \
# --wandb-exp-name $WANDB_EXP --wandb-save-dir "/work/nvme/bbvf/mtian8/LLM/wandb" \

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
# /u/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1  --recompute-granularity full --recompute-method block --recompute-num-layers 6 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 8 --sequence-parallel \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --seq-length 16384 --max-position-embeddings 16384 --rotary-base 1000000 \
# --micro-batch-size 1 --global-batch-size 1024 --split 100,0,0 \
# --lr 2.0e-5 --train-iters 300 --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path /projects/bbtf/mtian8/slimpajama-per-source-length-upsample_16k \
# --save-interval 10 --eval-interval 500 \
# --eval-iters 10 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /work/nvme/bcbw/mtian8/model/HF_model/Llama-2-70b-hf/tokenizer.model \
# --load /work/nvme/bcbw/mtian8/model/MG_model/llama2_70B_tp_4_pp_8 \
# --save /u/mtian8/LLM/data/checkpoint_long_70B \
# --tensorboard-dir "/u/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'llama2_70B_trial' \
# --wandb-exp-name 'llama2_70B ' --wandb-save-dir "/u/mtian8/LLM/wandb" \
# 2>&1 | tee out1.out  

# llama2-7B 32k
# TMPDIR=/projects/bcdz/mtian8/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /u/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1  --recompute-granularity full --recompute-method uniform --recompute-num-layers 8 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length 32768 --max-position-embeddings 32768 --rotary-base 1000000 \
# --micro-batch-size 2 --global-batch-size 64 --split 100,0,0 \
# --lr 2.0e-5 --train-iters 75 --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path /projects/bdao/mtian8/slimpajama-per-source-length-upsample_32k \
# --save-interval 10 --eval-interval 500 \
# --eval-iters 10 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model  /work/nvme/bcbw/mtian8/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load  /work/nvme/bcbw/mtian8/model/MG_model/llama2_7B_tp_4_pp_1 \
# --save /u/mtian8/LLM/data/checkpoint_long_32k \
# --tensorboard-dir "/u/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'llama2_7B_32k ' \
# --wandb-exp-name 'llama2_7B_32k ' --wandb-save-dir "/u/mtian8/LLM/wandb" \

# llama2-7B 128k
# TMPDIR=/projects/bcdz/mtian8/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /u/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 \
# --context-parallel-size 32 \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length 131072 --max-position-embeddings 131072 --rotary-base 1000000 \
# --micro-batch-size 1 --global-batch-size 1 --split 100,0,0 \
# --lr 2.0e-5 --train-iters 75 --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path /projects/bdao/mtian8/slimpajama-per-source-length-upsample_128k \
# --save-interval 10 --eval-interval 500 \
# --eval-iters 10 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /work/nvme/bcbw/mtian8/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /work/nvme/bcbw/mtian8/model/MG_model/llama2_7B_tp_4_pp_1 \
# --save /u/mtian8/LLM/data/checkpoint_long_32k \
# --tensorboard-dir "/u/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'llama2_7B_128k ' \
# --wandb-exp-name 'llama2_7B_128k ' --wandb-save-dir "/u/mtian8/LLM/wandb" \

# Mistral 7B 16k
# TMPDIR=/projects/bcdz/mtian8/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /u/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length 32768 --max-position-embeddings 32768 --rotary-base 1000000 \
# --micro-batch-size 1 --global-batch-size 32 --split 100,0,0 \
# --lr 2.0e-5 --train-iters 75 --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path /projects/bdao/mtian8/slimpajama-per-source-length-upsample_32k \
# --save-interval 10 --eval-interval 500 \
# --eval-iters 10 --tokenizer-type MistralTokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /work/nvme/bcbw/mtian8/model/HF_model/Mistral-7B-v0.1/tokenizer.model \
# --load /work/nvme/bcbw/mtian8/model/MG_model/mistral01_7B_tp_4_pp_1 \
# --save /u/mtian8/LLM/data/checkpoint_long_32k \
# --tensorboard-dir "/u/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'llama2_7B_32k ' \
# --wandb-exp-name 'llama2_7B_32k ' --wandb-save-dir "/u/mtian8/LLM/wandb" 

# agpt-7B sft
# TMPDIR=/projects/bcdz/mtian8/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /u/mtian8/LLM/Megatron-LM/sft_gpt_loss_mask.py --log-interval 1 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length 4096 --max-position-embeddings 4096 \
# --micro-batch-size 4 --global-batch-size 1024 --split 100,0,0 \
# --lr 2.0e-5 --train-iters 140 --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path /u/mtian8/LLM/Megatron-LM/processed/evol_sharegpt/train.jsonl \
# --save-interval 10 --eval-interval 500 \
# --eval-iters 10 --tokenizer-type Llama3Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model  /work/nvme/bcbw/mtian8/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load  /work/nvme/bcbw/mtian8/model/MG_model/agpt17000_7B_tp_4_pp_1 \
# --save /u/mtian8/LLM/data/agpt \
# --tensorboard-dir "/u/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'agpt ' \
# --wandb-exp-name 'agpt ' --wandb-save-dir "/u/mtian8/LLM/wandb" 

# llama2 7B sft
# TMPDIR=/projects/bcdz/mtian8/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /u/mtian8/LLM/Megatron-LM/sft_gpt_loss_mask.py --log-interval 1 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length 8192 --max-position-embeddings 8192 \
# --micro-batch-size 2 --global-batch-size 256 --split 100,0,0 \
# --lr 1.37e-5 --train-iters 1600 --lr-decay-style cosine --min-lr 1.37e-6 --weight-decay 0.1 \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion --attention-dropout 0 --hidden-dropout 0 \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path /u/mtian8/LLM/Megatron-LM/processed/merged_shuffle_train_llama2.jsonl \
# --save-interval 100 --eval-interval 500  --distributed-backend nccl --disable-bias-linear --normalization RMSNorm \
# --eval-iters 2000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /u/mtian8/LLM/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /u/mtian8/LLM/model/MG_model/llama2_7B_tp_4_pp_1 \
# --save /u/mtian8/LLM/data/checkpoint_llama2_7B_sft/ \
# --tensorboard-dir "/u/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'llama2 sft' \
# --wandb-exp-name 'llama2 sft' --wandb-save-dir "/u/mtian8/LLM/wandb" 

# llama3 8B sft
# TMPDIR=/projects/bcdz/mtian8/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /u/mtian8/LLM/Megatron-LM/sft_gpt_loss_mask.py --log-interval 1 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length 8192 --max-position-embeddings 8192 \
# --micro-batch-size 2 --global-batch-size 256 --split 100,0,0 \
# --lr 1.37e-5 --train-iters 1600 --lr-decay-style cosine --min-lr 1.37e-6 --weight-decay 0.1 \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion --attention-dropout 0 --hidden-dropout 0 \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path /u/mtian8/LLM/Megatron-LM/processed/merged_shuffle_train.jsonl \
# --save-interval 100 --eval-interval 500  --distributed-backend nccl --disable-bias-linear --normalization RMSNorm \
# --eval-iters 2000 --tokenizer-type Llama3Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /u/mtian8/LLM/model/HF_model/Meta-Llama-3-8B/tokenizer.model \
# --load /u/mtian8/LLM/model/MG_model/llama_8B_tp_4_pp_1 \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_llama3_8B_sft/ \
# --tensorboard-dir "/u/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'llama3 sft 0' \
# --wandb-exp-name 'llama3 sft' --wandb-save-dir "/u/mtian8/LLM/wandb" 

# llama3.1 8B sft resume
# TMPDIR=/projects/bcdz/mtian8/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /u/mtian8/LLM/Megatron-LM/sft_gpt_loss_mask.py --log-interval 1 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 \
# --use-checkpoint-args --logging-level 30 --log-throughput --log-progress \
# --seq-length 8192 --max-position-embeddings 8192 \
# --micro-batch-size 1 --global-batch-size 256 --split 100,0,0 \
# --lr 1.37e-5 --train-iters 1600 --lr-decay-style cosine --min-lr 1.37e-6 --weight-decay 0.1 \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion --attention-dropout 0 --hidden-dropout 0 \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path /u/mtian8/LLM/Megatron-LM/processed/merged_shuffle_train.jsonl \
# --save-interval 100 --eval-interval 500  --distributed-backend nccl --disable-bias-linear --normalization RMSNorm \
# --eval-iters 2000 --tokenizer-type Llama3Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /u/mtian8/LLM/model/HF_model/Meta-Llama-3-8B/tokenizer.model \
# --load /work/nvme/bbvf/mtian8/LLM/data/checkpoint_llama3_8B_sft \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_llama3_8B_sft/ \
# --tensorboard-dir "/u/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'llama3 sft' \
# --wandb-exp-name 'llama3 sft' --wandb-save-dir "/u/mtian8/LLM/wandb" 

# llama2-7B SciCoder 32k
# TMPDIR=/projects/bcdz/mtian8/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /u/mtian8/LLM/Megatron-LM/pretrain_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length 32768 --max-position-embeddings 32768 --rotary-base 1000000 \
# --micro-batch-size 4 --global-batch-size 256 --split 100,0,0 \
# --lr 2.0e-5 --train-iters 569 --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path /u/mtian8/LLM/data/pub_ocr/pub_ocr_text_document \
# --save-interval 10 --eval-interval 500 \
# --eval-iters 1000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model  /work/nvme/bcbw/mtian8/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /work/nvme/bcbw/mtian8/model/MG_model/llama2_7B_tp_4_pp_1 \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_scicoder_32k \
# --tensorboard-dir "/u/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'llama2_7B_scicoder ' \
# --wandb-exp-name 'llama2_7B_scicoder ' --wandb-save-dir "/u/mtian8/LLM/wandb" \

# llama2-7B SciCoder 32k resume
# TMPDIR=/projects/bcdz/mtian8/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /u/mtian8/LLM/Megatron-LM/pretrain_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --logging-level 30 --log-throughput --log-progress \
# --seq-length 32768 --max-position-embeddings 32768 --rotary-base 1000000 \
# --micro-batch-size 4 --global-batch-size 256 --split 100,0,0 \
# --lr 2.0e-5 --train-iters 569 --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path /u/mtian8/LLM/data/pub_ocr/pub_ocr_text_document \
# --save-interval 10 --eval-interval 500 \
# --eval-iters 1000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model  /work/nvme/bcbw/mtian8/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /work/nvme/bbvf/mtian8/LLM/data/checkpoint_scicoder_32k \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_scicoder_32k \
# --tensorboard-dir "/u/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'llama2_7B_scicoder ' \
# --wandb-exp-name 'llama2_7B_scicoder ' --wandb-save-dir "/u/mtian8/LLM/wandb" \

# llama2-7B SciCoder 32k
# TMPDIR=/projects/bcdz/mtian8/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /u/mtian8/LLM/Megatron-LM/pretrain_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length 32768 --max-position-embeddings 32768 --rotary-base 1000000 \
# --micro-batch-size 4 --global-batch-size 256 --split 100,0,0 \
# --lr 2.0e-5 --train-iters 569 --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path /u/mtian8/LLM/data/software/software_text_document \
# --save-interval 20 --eval-interval 500 \
# --eval-iters 1000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model  /work/nvme/bcbw/mtian8/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /work/nvme/bcbw/mtian8/model/MG_model/llama2_7B_tp_4_pp_1 \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_scicoder_code_32k \
# --tensorboard-dir "/u/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'llama2_7B_scicoder_code ' \
# --wandb-exp-name 'llama2_7B_scicoder_code ' --wandb-save-dir "/u/mtian8/LLM/wandb" \


# llama3 prolong 8B sft
# TMPDIR=/work/nvme/bbvf/mtian8/LLM/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /u/mtian8/LLM/Megatron-LM/sft_gpt_loss_mask.py --log-interval 1 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length 16384 --max-position-embeddings 16384 \
# --micro-batch-size 1 --global-batch-size 512 --split 100,0,0 \
# --lr 1.37e-5 --train-iters 482 --lr-decay-style cosine --min-lr 1.37e-6 --weight-decay 0.1 \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion --attention-dropout 0 --hidden-dropout 0 \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path /work/nvme/bcbw/mtian8/train.jsonl \
# --save-interval 100 --eval-interval 500  --distributed-backend nccl --disable-bias-linear --normalization RMSNorm \
# --eval-iters 2000 --tokenizer-type Llama3Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /work/nvme/bbvf/mtian8/LLM/model/HF_model/Meta-Llama-3-8B-Instruct/tokenizer.model \
# --load /work/nvme/bbvf/mtian8/LLM/model/MG_model/llama_8B_ProLong_tp_4_pp_1 \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_llama3_8B_ProLong_scicode/ \
# --tensorboard-dir "/work/nvme/bbvf/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project 'llama3 prolong SicCoder' \
# --wandb-exp-name 'llama3 ProLong SciCode' --wandb-save-dir "/work/nvme/bbvf/mtian8/LLM/wandb" 


# --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \

# bash launch_multinode_sft.sh 2>&1 | tee out.out
# bash multinode_kill.sh
# 8M batch size
# SEQ_LENGTH=32768
# LEN_ABBR=32k
# MBS=1
# GBS=256
# # Rotary Base
# THETA=1000000
# # Total Steps
# TOT_STEPS=506
# # Data Path
# DATA=/projects/bdao/mtian8/slimpajama-per-source-length-upsample_${LEN_ABBR}
# SLIDING_WINDOW_SIZE=4096
# SAVE_INTERVAL=20
# # W&B setting
# WANDB_PROJECT=llama2_7B_${LEN_ABBR}_SW
# WANDB_EXP=llama2_7B_${LEN_ABBR}_SW_${GBS}_${THETA}

# # # llama2-7B 32k_sliding_window
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=/work/nvme/bbvf/mtian8/LLM/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /u/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1  --sequence-parallel \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length $SEQ_LENGTH  --max-position-embeddings $SEQ_LENGTH --rotary-base $THETA \
# --micro-batch-size $MBS --global-batch-size $GBS --split 100,0,0 \
# --lr 2.0e-5 --train-iters $TOT_STEPS --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path $DATA \
# --save-interval $SAVE_INTERVAL --eval-interval 500 \
# --window-size $SLIDING_WINDOW_SIZE 0 \
# --eval-iters 1000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /u/mtian8/LLM/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /u/mtian8/LLM/model/MG_model/llama2_7B_tp_4_pp_1 \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_${LEN_ABBR}_ori \
# --tensorboard-dir "/work/nvme/bbvf/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project $WANDB_PROJECT \
# --wandb-exp-name $WANDB_EXP --wandb-save-dir "/work/nvme/bbvf/mtian8/LLM/wandb" \


# llama2-7B 64k
# 4M batch size
# SEQ_LENGTH=$((32768*2))
# LEN_ABBR=128k
# MBS=1
# GBS=$((256/4))
# # Rotary Base
# THETA=1000000
# # Total Steps
# TOT_STEPS=2000
# # Data Path
# DATA=/projects/bdao/mtian8/slimpajama-per-source-length-upsample_${LEN_ABBR}
# SAVE_INTERVAL=100
# # W&B setting
# WANDB_PROJECT=llama2_7B_${LEN_ABBR}
# WANDB_EXP=llama2_7B_${LEN_ABBR}_${GBS}_${THETA}

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=/work/nvme/bbvf/mtian8/LLM/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /work/nvme/bbvf/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length $SEQ_LENGTH  --max-position-embeddings $SEQ_LENGTH --rotary-base $THETA \
# --micro-batch-size $MBS --global-batch-size $GBS --split 100,0,0 \
# --lr 2.0e-5 --train-iters $TOT_STEPS --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path $DATA \
# --save-interval $SAVE_INTERVAL --eval-interval 5000 \
# --eval-iters 5000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /u/mtian8/LLM/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /u/mtian8/LLM/model/MG_model/llama2_7B_tp_4_pp_1 \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_${LEN_ABBR}_divide2 \
# --tensorboard-dir "/work/nvme/bbvf/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project $WANDB_PROJECT \
# --wandb-exp-name $WANDB_EXP --wandb-save-dir "/work/nvme/bbvf/mtian8/LLM/wandb" \


# llama2-7B 64k resume
# 4M batch size
SEQ_LENGTH=$((32768*2))
LEN_ABBR=128k
MBS=1
GBS=$((256/4))
# Rotary Base
THETA=1000000
# Total Steps
TOT_STEPS=2000
# Data Path
DATA=/projects/bdao/mtian8/slimpajama-per-source-length-upsample_${LEN_ABBR}
SAVE_INTERVAL=100
# W&B setting
WANDB_PROJECT=llama2_7B_${LEN_ABBR}
WANDB_EXP=llama2_7B_${LEN_ABBR}_${GBS}_${THETA}

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=/work/nvme/bbvf/mtian8/LLM/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
/work/nvme/bbvf/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
--tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel \
--use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
--use-checkpoint-args --logging-level 30 --log-throughput --log-progress \
--seq-length $SEQ_LENGTH  --max-position-embeddings $SEQ_LENGTH --rotary-base $THETA \
--micro-batch-size $MBS --global-batch-size $GBS --split 100,0,0 \
--lr 2.0e-5 --train-iters $TOT_STEPS --lr-decay-style constant \
--lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
--norm-epsilon 1e-5 --no-masked-softmax-fusion \
--optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
--exit-on-missing-checkpoint --dataloader-type cyclic \
--data-path $DATA \
--save-interval $SAVE_INTERVAL --eval-interval 5000 \
--eval-iters 5000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
--untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
--tokenizer-model /u/mtian8/LLM/model/HF_model/Llama-2-7b-hf/tokenizer.model \
--load /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_${LEN_ABBR}_divide2 \
--save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_${LEN_ABBR}_divide2 \
--tensorboard-dir "/work/nvme/bbvf/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project $WANDB_PROJECT \
--wandb-exp-name $WANDB_EXP --wandb-save-dir "/work/nvme/bbvf/mtian8/LLM/wandb" \



# llama2-7B 128k
# 8M batch size

# SEQ_LENGTH=$((32768*4))
# LEN_ABBR=128k
# MBS=1
# GBS=$((256/8))
# # Rotary Base
# THETA=5000000
# # Total Steps
# TOT_STEPS=2000
# # Data Path
# DATA=/projects/bdao/mtian8/slimpajama-per-source-length-upsample_${LEN_ABBR}
# SAVE_INTERVAL=100
# # W&B setting
# WANDB_PROJECT=llama2_7B_${LEN_ABBR}
# WANDB_EXP=llama2_7B_${LEN_ABBR}_${GBS}_${THETA}

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=/work/nvme/bbvf/mtian8/LLM/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /work/nvme/bbvf/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length $SEQ_LENGTH  --max-position-embeddings $SEQ_LENGTH --rotary-base $THETA \
# --micro-batch-size $MBS --global-batch-size $GBS --split 100,0,0 \
# --lr 2.0e-5 --train-iters $TOT_STEPS --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path $DATA \
# --save-interval $SAVE_INTERVAL --eval-interval 5000 \
# --eval-iters 1000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /u/mtian8/LLM/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /u/mtian8/LLM/model/MG_model/llama2_7B_tp_4_pp_1 \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_${LEN_ABBR} \
# --tensorboard-dir "/work/nvme/bbvf/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project $WANDB_PROJECT \
# --wandb-exp-name $WANDB_EXP --wandb-save-dir "/work/nvme/bbvf/mtian8/LLM/wandb" \

# llama2-7B 128k continue
# 8M batch size
# SEQ_LENGTH=$((32768*4))
# LEN_ABBR=128k
# MBS=1
# GBS=$((256/8))
# # Rotary Base
# THETA=5000000
# # Total Steps
# TOT_STEPS=2000
# # Data Path
# DATA=/projects/bdao/mtian8/slimpajama-per-source-length-upsample_${LEN_ABBR}
# SLIDING_WINDOW_SIZE=-1
# SAVE_INTERVAL=100
# # W&B setting
# WANDB_PROJECT=llama2_7B_${LEN_ABBR}
# WANDB_EXP=llama2_7B_${LEN_ABBR}_${GBS}_${THETA}

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=/work/nvme/bbvf/mtian8/LLM/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /work/nvme/bbvf/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --logging-level 30 --log-throughput --log-progress \
# --seq-length $SEQ_LENGTH  --max-position-embeddings $SEQ_LENGTH --rotary-base $THETA \
# --micro-batch-size $MBS --global-batch-size $GBS --split 100,0,0 \
# --lr 2.0e-5 --train-iters $TOT_STEPS --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path $DATA \
# --save-interval $SAVE_INTERVAL --eval-interval 500 \
# --window-size $SLIDING_WINDOW_SIZE 0 \
# --eval-iters 1000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /u/mtian8/LLM/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_${LEN_ABBR} \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_${LEN_ABBR} \
# --tensorboard-dir "/work/nvme/bbvf/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project $WANDB_PROJECT \
# --wandb-exp-name $WANDB_EXP --wandb-save-dir "/work/nvme/bbvf/mtian8/LLM/wandb" \


# llama2-7B 128k 16k sliding window
# 8M batch size
# SEQ_LENGTH=$((32768*4))
# LEN_ABBR=128k
# MBS=1
# GBS=$((256/4))
# # Rotary Base
# THETA=8000000
# # Total Steps
# TOT_STEPS=506
# # Data Path
# DATA=/projects/bdao/mtian8/slimpajama-per-source-length-upsample_${LEN_ABBR}
# SLIDING_WINDOW_SIZE=16384
# SAVE_INTERVAL=50
# # W&B setting
# WANDB_PROJECT=llama2_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}
# WANDB_EXP=llama2_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}_${GBS}_${THETA}

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=/work/nvme/bbvf/mtian8/LLM/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /work/nvme/bbvf/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length $SEQ_LENGTH  --max-position-embeddings $SEQ_LENGTH --rotary-base $THETA \
# --micro-batch-size $MBS --global-batch-size $GBS --split 100,0,0 \
# --lr 2.0e-5 --train-iters $TOT_STEPS --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path $DATA \
# --save-interval $SAVE_INTERVAL --eval-interval 500 \
# --window-size $SLIDING_WINDOW_SIZE 0 \
# --eval-iters 1000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /u/mtian8/LLM/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /u/mtian8/LLM/model/MG_model/llama2_7B_tp_4_pp_1 \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_${LEN_ABBR}_${SLIDING_WINDOW_SIZE} \
# --tensorboard-dir "/work/nvme/bbvf/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project $WANDB_PROJECT \
# --wandb-exp-name $WANDB_EXP --wandb-save-dir "/work/nvme/bbvf/mtian8/LLM/wandb" \


# llama2-7B 128k 16k sliding window continue
# 8M batch size
# SEQ_LENGTH=$((32768*4))
# LEN_ABBR=128k
# MBS=1
# GBS=$((256/4))
# # Rotary Base
# THETA=8000000
# # Total Steps
# TOT_STEPS=506
# # Data Path
# DATA=/projects/bdao/mtian8/slimpajama-per-source-length-upsample_${LEN_ABBR}
# SLIDING_WINDOW_SIZE=16384
# SAVE_INTERVAL=50
# # W&B setting
# WANDB_PROJECT=llama2_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}
# WANDB_EXP=llama2_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}_${GBS}_${THETA}

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=/work/nvme/bbvf/mtian8/LLM/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /work/nvme/bbvf/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --logging-level 30 --log-throughput --log-progress \
# --seq-length $SEQ_LENGTH  --max-position-embeddings $SEQ_LENGTH --rotary-base $THETA \
# --micro-batch-size $MBS --global-batch-size $GBS --split 100,0,0 \
# --lr 2.0e-5 --train-iters $TOT_STEPS --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path $DATA \
# --save-interval $SAVE_INTERVAL --eval-interval 500 \
# --window-size $SLIDING_WINDOW_SIZE 0 \
# --eval-iters 1000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /u/mtian8/LLM/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_${LEN_ABBR}_${SLIDING_WINDOW_SIZE} \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_${LEN_ABBR}_${SLIDING_WINDOW_SIZE} \
# --tensorboard-dir "/work/nvme/bbvf/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project $WANDB_PROJECT \
# --wandb-exp-name $WANDB_EXP --wandb-save-dir "/work/nvme/bbvf/mtian8/LLM/wandb" \



# llama2-7B 128k 4k sliding window
# 8M batch size
# SEQ_LENGTH=$((32768*4))
# LEN_ABBR=128k
# MBS=1
# GBS=$((256/4))
# # Rotary Base
# THETA=8000000
# # Total Steps
# TOT_STEPS=506
# # Data Path
# DATA=/projects/bdao/mtian8/slimpajama-per-source-length-upsample_${LEN_ABBR}
# SLIDING_WINDOW_SIZE=4096
# SAVE_INTERVAL=2
# # W&B setting
# WANDB_PROJECT=llama2_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}_1.9
# WANDB_EXP=llama2_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}_${GBS}_${THETA}_1.9

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=/work/nvme/bbvf/mtian8/LLM/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /work/nvme/bbvf/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length $SEQ_LENGTH  --max-position-embeddings $SEQ_LENGTH --rotary-base $THETA \
# --micro-batch-size $MBS --global-batch-size $GBS --split 100,0,0 \
# --lr 2.0e-5 --train-iters $TOT_STEPS --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path $DATA \
# --save-interval $SAVE_INTERVAL --eval-interval 500 \
# --window-size $SLIDING_WINDOW_SIZE 0 \
# --eval-iters 1000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /u/mtian8/LLM/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /u/mtian8/LLM/model/MG_model/llama2_7B_tp_4_pp_1 \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_${LEN_ABBR}_${SLIDING_WINDOW_SIZE}_1.9 \
# --tensorboard-dir "/work/nvme/bbvf/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project $WANDB_PROJECT \
# --wandb-exp-name $WANDB_EXP --wandb-save-dir "/work/nvme/bbvf/mtian8/LLM/wandb" \



# llama2-7B 128k 2k sliding window
# 8M batch size
# SEQ_LENGTH=$((32768*4))
# LEN_ABBR=128k
# MBS=1
# GBS=$((256/4))
# # Rotary Base
# THETA=8000000
# # Total Steps
# TOT_STEPS=506
# # Data Path
# DATA=/projects/bdao/mtian8/slimpajama-per-source-length-upsample_${LEN_ABBR}
# SLIDING_WINDOW_SIZE=2048
# SAVE_INTERVAL=100
# # W&B setting
# WANDB_PROJECT=llama2_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}
# WANDB_EXP=llama2_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}_${GBS}_${THETA}

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=/work/nvme/bbvf/mtian8/LLM/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /work/nvme/bbvf/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length $SEQ_LENGTH  --max-position-embeddings $SEQ_LENGTH --rotary-base $THETA \
# --micro-batch-size $MBS --global-batch-size $GBS --split 100,0,0 \
# --lr 2.0e-5 --train-iters $TOT_STEPS --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path $DATA \
# --save-interval $SAVE_INTERVAL --eval-interval 500 \
# --window-size $SLIDING_WINDOW_SIZE 0 \
# --eval-iters 1000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /u/mtian8/LLM/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /u/mtian8/LLM/model/MG_model/llama2_7B_tp_4_pp_1 \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_${LEN_ABBR}_${SLIDING_WINDOW_SIZE} \
# --tensorboard-dir "/work/nvme/bbvf/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project $WANDB_PROJECT \
# --wandb-exp-name $WANDB_EXP --wandb-save-dir "/work/nvme/bbvf/mtian8/LLM/wandb" \



# llama2-7B 128k 2k sliding window resume
# 8M batch size
# SEQ_LENGTH=$((32768*4))
# LEN_ABBR=128k
# MBS=1
# GBS=$((256/4))
# # Rotary Base
# THETA=8000000
# # Total Steps
# TOT_STEPS=506
# # Data Path
# DATA=/projects/bdao/mtian8/slimpajama-per-source-length-upsample_${LEN_ABBR}
# SLIDING_WINDOW_SIZE=2048
# SAVE_INTERVAL=100
# # W&B setting
# WANDB_PROJECT=llama2_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}
# WANDB_EXP=llama2_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}_${GBS}_${THETA}

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=/work/nvme/bbvf/mtian8/LLM/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /work/nvme/bbvf/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --logging-level 30 --log-throughput --log-progress \
# --seq-length $SEQ_LENGTH  --max-position-embeddings $SEQ_LENGTH --rotary-base $THETA \
# --micro-batch-size $MBS --global-batch-size $GBS --split 100,0,0 \
# --lr 2.0e-5 --train-iters $TOT_STEPS --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path $DATA \
# --save-interval $SAVE_INTERVAL --eval-interval 500 \
# --window-size $SLIDING_WINDOW_SIZE 0 \
# --eval-iters 1000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /u/mtian8/LLM/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_${LEN_ABBR}_${SLIDING_WINDOW_SIZE} \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_${LEN_ABBR}_${SLIDING_WINDOW_SIZE} \
# --tensorboard-dir "/work/nvme/bbvf/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project $WANDB_PROJECT \
# --wandb-exp-name $WANDB_EXP --wandb-save-dir "/work/nvme/bbvf/mtian8/LLM/wandb" \


# llama2-7B 256k
# 8M batch size

# SEQ_LENGTH=$((32768*8))
# LEN_ABBR=256k
# MBS=1
# GBS=$((256/8))
# # Rotary Base
# THETA=50000000
# # Total Steps
# TOT_STEPS=506
# # Data Path
# DATA=/projects/bdao/mtian8/slimpajama-per-source-length-upsample-${LEN_ABBR}
# SLIDING_WINDOW_SIZE=-1
# SAVE_INTERVAL=100
# # W&B setting
# WANDB_PROJECT=llama2_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}
# WANDB_EXP=llama2_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}_${GBS}_${THETA}

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=/work/nvme/bbvf/mtian8/LLM/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /work/nvme/bbvf/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length $SEQ_LENGTH  --max-position-embeddings $SEQ_LENGTH --rotary-base $THETA \
# --micro-batch-size $MBS --global-batch-size $GBS --split 100,0,0 \
# --lr 2.0e-5 --train-iters $TOT_STEPS --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path $DATA \
# --save-interval $SAVE_INTERVAL --eval-interval 500 \
# --window-size $SLIDING_WINDOW_SIZE 0 \
# --eval-iters 1000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /u/mtian8/LLM/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /u/mtian8/LLM/model/MG_model/llama2_7B_tp_4_pp_1 \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_${LEN_ABBR}_${SLIDING_WINDOW_SIZE}_5000k \
# --tensorboard-dir "/work/nvme/bbvf/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project $WANDB_PROJECT \
# --wandb-exp-name $WANDB_EXP --wandb-save-dir "/work/nvme/bbvf/mtian8/LLM/wandb" \


# llama2-7B 256k resume
# 8M batch size

# SEQ_LENGTH=$((32768*8))
# LEN_ABBR=256k
# MBS=1
# GBS=$((256/8))
# # Rotary Base
# THETA=50000000
# # Total Steps
# TOT_STEPS=506
# # Data Path
# DATA=/projects/bdao/mtian8/slimpajama-per-source-length-upsample-${LEN_ABBR}
# SLIDING_WINDOW_SIZE=-1
# SAVE_INTERVAL=100
# # W&B setting
# WANDB_PROJECT=llama2_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}
# WANDB_EXP=llama2_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}_${GBS}_${THETA}

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=/work/nvme/bbvf/mtian8/LLM/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /work/nvme/bbvf/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --logging-level 30 --log-throughput --log-progress \
# --seq-length $SEQ_LENGTH  --max-position-embeddings $SEQ_LENGTH --rotary-base $THETA \
# --micro-batch-size $MBS --global-batch-size $GBS --split 100,0,0 \
# --lr 2.0e-5 --train-iters $TOT_STEPS --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path $DATA \
# --save-interval $SAVE_INTERVAL --eval-interval 500 \
# --window-size $SLIDING_WINDOW_SIZE 0 \
# --eval-iters 1000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /u/mtian8/LLM/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /u/mtian8/LLM/data/checkpoint_long_${LEN_ABBR}_${SLIDING_WINDOW_SIZE}_5000k \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_${LEN_ABBR}_${SLIDING_WINDOW_SIZE}_5000k \
# --tensorboard-dir "/work/nvme/bbvf/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project $WANDB_PROJECT \
# --wandb-exp-name $WANDB_EXP --wandb-save-dir "/work/nvme/bbvf/mtian8/LLM/wandb" \

# llama2-7B 256k 4k sliding window
# 8M batch size

# SEQ_LENGTH=$((32768*8))
# LEN_ABBR=256k
# MBS=1
# GBS=$((256/8))
# # Rotary Base
# THETA=50000000
# # Total Steps
# TOT_STEPS=506
# # Data Path
# DATA=/projects/bdao/mtian8/slimpajama-per-source-length-upsample-${LEN_ABBR}
# SLIDING_WINDOW_SIZE=4096
# SAVE_INTERVAL=100
# # W&B setting
# WANDB_PROJECT=llama2_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}
# WANDB_EXP=llama2_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}_${GBS}_${THETA}

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=/work/nvme/bbvf/mtian8/LLM/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /work/nvme/bbvf/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length $SEQ_LENGTH  --max-position-embeddings $SEQ_LENGTH --rotary-base $THETA \
# --micro-batch-size $MBS --global-batch-size $GBS --split 100,0,0 \
# --lr 2.0e-5 --train-iters $TOT_STEPS --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path $DATA \
# --save-interval $SAVE_INTERVAL --eval-interval 500 \
# --window-size $SLIDING_WINDOW_SIZE 0 \
# --eval-iters 1000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /u/mtian8/LLM/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /u/mtian8/LLM/model/MG_model/llama2_7B_tp_4_pp_1 \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_${LEN_ABBR}_${SLIDING_WINDOW_SIZE}_5000k \
# --tensorboard-dir "/work/nvme/bbvf/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project $WANDB_PROJECT \
# --wandb-exp-name $WANDB_EXP --wandb-save-dir "/work/nvme/bbvf/mtian8/LLM/wandb" \


# llama2-7B 256k 4k sliding window
# 8M batch size

# SEQ_LENGTH=$((32768*8))
# LEN_ABBR=256k
# MBS=1
# GBS=$((256/8))
# # Rotary Base
# THETA=50000000
# # Total Steps
# TOT_STEPS=506
# # Data Path
# DATA=/projects/bdao/mtian8/slimpajama-per-source-length-upsample-${LEN_ABBR}
# SLIDING_WINDOW_SIZE=4096
# SAVE_INTERVAL=100
# # W&B setting
# WANDB_PROJECT=llama2_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}
# WANDB_EXP=llama2_7B_${LEN_ABBR}_SW_${SLIDING_WINDOW_SIZE}_${GBS}_${THETA}

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=/work/nvme/bbvf/mtian8/LLM/tmp CUDA_DEVICE_MAX_CONNECTIONS=1  torchrun \
# --nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODEID \
# --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# /work/nvme/bbvf/mtian8/LLM/Megatron-LM/sft_gpt.py --log-interval 1 --recompute-granularity full --recompute-method uniform --recompute-num-layers 2 \
# --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --sequence-parallel \
# --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather \
# --use-checkpoint-args --no-load-optim --no-load-rng --logging-level 30 --log-throughput --log-progress \
# --seq-length $SEQ_LENGTH  --max-position-embeddings $SEQ_LENGTH --rotary-base $THETA \
# --micro-batch-size $MBS --global-batch-size $GBS --split 100,0,0 \
# --lr 2.0e-5 --train-iters $TOT_STEPS --lr-decay-style constant \
# --lr-warmup-fraction 0.001 --clip-grad 1.0 --bf16 --use-flash-attn \
# --norm-epsilon 1e-5 --no-masked-softmax-fusion \
# --optimizer adam --adam-beta1 0.9 --adam-beta2 0.999 --adam-eps 1e-08 \
# --exit-on-missing-checkpoint --dataloader-type cyclic \
# --data-path $DATA \
# --save-interval $SAVE_INTERVAL --eval-interval 500 \
# --window-size $SLIDING_WINDOW_SIZE 0 \
# --eval-iters 1000 --tokenizer-type Llama2Tokenizer --attention-softmax-in-fp32 \
# --untie-embeddings-and-output-weights --normalization RMSNorm --position-embedding-type rope \
# --tokenizer-model /u/mtian8/LLM/model/HF_model/Llama-2-7b-hf/tokenizer.model \
# --load /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_${LEN_ABBR}_${SLIDING_WINDOW_SIZE}_5000k \
# --save /work/nvme/bbvf/mtian8/LLM/data/checkpoint_long_${LEN_ABBR}_${SLIDING_WINDOW_SIZE}_5000k \
# --tensorboard-dir "/work/nvme/bbvf/mtian8/LLM/Megatron-LM/log" --tensorboard-queue-size 5 --wandb-project $WANDB_PROJECT \
# --wandb-exp-name $WANDB_EXP --wandb-save-dir "/work/nvme/bbvf/mtian8/LLM/wandb" \