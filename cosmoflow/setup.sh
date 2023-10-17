#!/bin/sh
source /home/hzheng/PolarisAT/dlio_ml_workloads/setup_ml_env.sh
export MPICH_GPU_SUPPORT_ENABLED=0
export CPATH=/grand/datascience/hzheng/mlperf-2022/optimized-hpc/boost_1_80_0/:$CPATH
export LD_LIBRARY_PATH=/grand/datascience/hzheng/mlperf-2022/optimized-hpc/boost_1_80_0/stage/lib:$LD_LIBRARY_PATH
