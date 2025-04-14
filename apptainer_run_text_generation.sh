CONTAINER_HOME=/sw/user/NGC_containers
SIF_FILE=pytorch_24.07-py3.sif
apptainer exec --nv \
    --bind /work \
    --bind /projects \
    ${CONTAINER_HOME}/${SIF_FILE} \
    bash ./rtg.sh "$1" "$2" "$3" "$4" "$5" "$6" 
