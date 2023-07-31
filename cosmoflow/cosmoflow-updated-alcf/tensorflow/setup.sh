#!/bin/sh
module load conda/2022-07-19; conda activate
export MPICH_GPU_SUPPORT_ENABLED=0
#export CPATH=/home/hzheng/datascience_grand/mlperf_hpc/hpc-nvidia/boost_1_80_0/:$CPATH
#export LD_LIBRARY_PATH=/home/hzheng/datascience_grand/mlperf_hpc/hpc-nvidia/boost_1_80_0/stage/lib:$LD_LIBRARY_PATH
