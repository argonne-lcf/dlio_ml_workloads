#!/bin/sh
source /home/hzheng/PolarisAT/dlio_ml_workloads/setup_ml_env.sh
BOOST=$WORKDIR/cosmoflow/boost
export MPICH_GPU_SUPPORT_ENABLED=0
export CPATH=${BOOST}/:$CPATH
export LD_LIBRARY_PATH=${BOOST}/stage/lib:$LD_LIBRARY_PATH
