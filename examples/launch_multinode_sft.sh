#!/bin/bash



# Path to your file

echo $HOME
FILE="$HOME/hostname.txt"
echo "hostname text: $FILE"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MEGATRON_DIR="$SCRIPT_DIR/.."
rank=0
#CONDA_COMMAND="conda activate mg-lm;"
CONDA_COMMAND=":"
# Function to perform SSH and run the command in the background

ssh_command() {

    local line=$1

    echo "Attempting to SSH into $line"

    ssh -o ConnectTimeout=5 "$line" "module unload nccl/2.19.3.awsplugin; apptainer exec --nv --bind /projects  --bind /work /sw/user/NGC_containers/pytorch_24.07-py3.sif bash -c 'export HOSTNAME_FILE=$FILE; $CONDA_COMMAND; cd $MEGATRON_DIR; bash ./scripts/run_sft_distributed.sh $rank'" < /dev/null
    # ssh -o ConnectTimeout=5 "$line" "apptainer exec --nv --bind /projects  --bind /work /sw/user/NGC_containers/pytorch_24.09-py3.sif bash -c 'export HOSTNAME_FILE=$FILE; $CONDA_COMMAND; cd $MEGATRON_DIR; bash ./scripts/run_sft_distributed.sh $rank'" < /dev/null
    # ssh -o ConnectTimeout=5 "$line" "module unload nccl/2.19.3.awsplugin; apptainer exec --nv --bind /projects --bind /work  --bind /sw/user/nccl/aws-ofi.1.6.0/lib:/opt/hpcx/nccl_rdma_sharp_plugin/lib --bind  /opt/cray/libfabric/1.15.2.0/lib64/libfabric.so.1:/usr/lib64/libfabric.so.1 --bind /usr/lib64/libcxi.so.1:/usr/lib64/libcxi.so.1 --bind /usr/lib64/libjson-c.so.3:/usr/lib64/libjson-c.so.3 /u/mtian8/LLM/pytorch_23.12-py3.sif bash -c 'export HOSTNAME_FILE=$FILE; $CONDA_COMMAND; cd $MEGATRON_DIR; bash ./scripts/run_sft_distributed.sh $rank'" < /dev/null
    # ssh -o ConnectTimeout=5 "$line" "apptainer exec --nv --bind /work /sw/user/NGC_containers/pytorch_24.07-py3.sif bash -c 'export HOSTNAME_FILE=$FILE; $CONDA_COMMAND; cd $MEGATRON_DIR; bash ./scripts/run_sft_distributed.sh $rank'" < /dev/null
    # ssh -o ConnectTimeout=5 "$line" "apptainer exec --nv --bind /projects --bind /work --bind /opt/cray/libfabric/1.15.2.0 --bind /sw/user/nccl/aws-ofi.1.6.0 --bind /opt/nvidia/hpc_sdk/Linux_aarch64/23.11/comm_libs/12.3/nccl /sw/user/NGC_containers/pytorch_24.07-py3.sif bash -c 'export HOSTNAME_FILE=$FILE; $CONDA_COMMAND; cd $MEGATRON_DIR; bash ./scripts/run_sft_distributed.sh $rank'" < /dev/null

    
    if [ $? -eq 0 ]; then

        echo "SSH to $line successful"

    else

        echo "SSH to $line failed"

    fi

    (( rank += 1))

}



# Loop through each line in the file and run SSH commands in parallel

while IFS= read -r line; do

    ssh_command "$line" &

    (( rank += 1))

done < "$FILE"



# Wait for all background jobs to complete

wait



echo "All SSH commands executed."
