#!/bin/sh
# export CUDA_VISIBLE_DEVICES=$PMI_LOCAL_RANK
# export CUDA_LAUNCH_BLOCKING=1
export LOCAL_RANK=$PMI_LOCAL_RANK
export LOCAL_WORLD_SIZE=$(nvidia-smi -L | wc -l)
export RANK=$PMI_RANK
export SIZE=$PMI_SIZE
export WORLD_SIZE=$SIZE
if [ -z "${WORLD_SIZE}" ]; then
    export WORLD_SIZE=1
    export SIZE=1
fi
if [ -z "${RANK}" ]; then
    export RANK=0
    export LOCAL_RANK=0
fi
if [[ $RANK -eq 0 ]]; then
    hostname >hostname.$PBS_JOBID
    MASTER_ADDR=$(hostname)
    sleep 1
else
    sleep 1
    MASTER_ADDR=$(cat ./hostname.$PBS_JOBID)
fi
echo "Launcher [$(hostname)]: g_rank=$RANK, g_world_size=$SIZE, l_rank=$LOCAL_RANK, l_world_size=$LOCAL_WORLD_SIZE, master_addr=$MASTER_ADDR"
$@
rm -f "./hostname.$PBS_JOBID"
