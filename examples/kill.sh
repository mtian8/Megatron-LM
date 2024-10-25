#!/bin/bash

# Function to kill GPU processes
kill_gpu_processes() {
    # List all GPU processes
    echo "Listing GPU processes on $1:"
    nvidia-smi

    # Extract GPU process IDs (PIDs) from nvidia-smi output
    PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | awk '{print $1}')
    
    if [ -z "$PIDS" ]; then
        echo "No GPU processes to kill on $1."
    else
        for pid in $PIDS
        do
            echo "Killing process $pid on $1..."
            kill -9 $pid
        done
    fi
}


# kill process
kill_gpu_processes $1

