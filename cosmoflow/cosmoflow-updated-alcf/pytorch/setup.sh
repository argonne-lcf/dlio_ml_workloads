#!/bin/sh
module load conda/2022-07-19; conda activate
export TMPDIR=./results
export MPICH_GPU_SUPPORT_ENABLED=0
export CPATH=/home/hzheng/datascience_grand/mlperf_hpc/hpc-nvidia/boost_1_80_0/:$CPATH
export LD_LIBRARY_PATH=/lus/grand/projects/datascience/kaushikv/dlio/ml_workloads/cosmoflow/cosmoflow-pytorch-huihuos/libboost/libbost:$LD_LIBRARY_PATH
 