#!/bin/bash



# Path to your file

echo $HOME
FILE="$HOME/hostname.txt"
echo "hostname text: $FILE"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
MEGATRON_DIR="$SCRIPT_DIR/.."
#CONDA_COMMAND="conda activate mg-lm;"
# Function to perform SSH and run the command in the background

ssh_command() {

    local line=$1

    echo "Attempting to SSH into $line"

    # ssh -o ConnectTimeout=5 "$line" "bash /u/mtian8/LLM/Megatron-LM/scripts/kill.sh $line" < /dev/null
    ssh -o ConnectTimeout=5 "$line" "bash /work/nvme/bbvf/mtian8/LLM/Megatron-LM/scripts/kill.sh $line" < /dev/null
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
