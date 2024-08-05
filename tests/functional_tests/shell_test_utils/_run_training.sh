#!/bin/bash

# This script can be used for model onboarding and testing.

# For onboarding, it extract scalars from Tensorboard logs only.
# For testing, it compares extracted Tensorboard scalars against
# a set of `GOLDEN_VALUES`.

set -euxo pipefail

echo "------ARGUMENTS LIST --------"
for ARGUMENT in "$@"; do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)

    KEY_LENGTH=${#KEY}
    VALUE="${ARGUMENT:$KEY_LENGTH+1}"

    export "$KEY"="$VALUE"
    echo "$KEY=$VALUE"
done
echo "---------------------------------"

# Check that mandatory vars are set
MANDATORY_VARS=(
    "TRAINING_SCRIPT_PATH"
    "TRAINING_PARAMS_PATH"
    "OUTPUT_PATH"
    "TENSORBOARD_PATH"
    "CHECKPOINT_PATH"
    "DATA_PATH"
)
for mandatory_var in "${MANDATORY_VARS[@]}"; do
    if [[ -z "${!mandatory_var}" ]]; then
        echo 'Providing $'$mandatory_var' is mandatory.'
        exit 1
    fi
done

# Envsubst model_params
cat $TRAINING_PARAMS_PATH | envsubst >$TRAINING_PARAMS_PATH.tmp
mv $TRAINING_PARAMS_PATH.tmp $TRAINING_PARAMS_PATH

# Exit earlier to leave time for properly saving checkpoint
PARAMS="--exit-duration-in-mins $((($SLURM_JOB_END_TIME - $SLURM_JOB_START_TIME) / 60 - 15))"

# Run before script
SCRIPT=$(cat $TRAINING_PARAMS_PATH | yq .'BEFORE_SCRIPT')
if [[ "$SCRIPT" != null ]]; then
    eval "$SCRIPT"
fi;

# Extract training params
TRAINING_PARAMS_FROM_CONFIG=$(yq '... comments="" | .MODEL_ARGS | to_entries | .[] | with(select(.value == "true"); .value = "") | [.key + " " + .value] | join("")' $TRAINING_PARAMS_PATH | tr '\n' ' ')
PARAMS="$PARAMS $TRAINING_PARAMS_FROM_CONFIG"

# Pull env vars to export
ENV_VARS=$(yq '... comments="" | .ENV_VARS | to_entries | .[] | [.key + "=" + .value] | join(" ")' $TRAINING_PARAMS_PATH)
for ARGUMENT in $ENV_VARS; do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)

    KEY_LENGTH=${#KEY}
    VALUE="${ARGUMENT:$KEY_LENGTH+1}"

    export "$KEY"="$VALUE"
    echo "$KEY=$VALUE"
done

# Set PYTHONPATH
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

######## Distributed training settings. ########
echo "------ARGUMENTS for SLURM ---"
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NUM_NODES=${NUM_NODES:-${SLURM_NNODES}}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODE_RANK=${SLURM_NODEID:-${SLURM_NODEID}}
DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --node_rank $SLURM_NODEID
)

# Start training
torchrun ${DISTRIBUTED_ARGS[@]} $TRAINING_SCRIPT_PATH $PARAMS

