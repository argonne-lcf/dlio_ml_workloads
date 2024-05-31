#!/bin/sh
source /home/hzheng/PolarisAT_eagle/dlio_ml_workloads/setup_ml_env.sh
BOOST=$WORKDIR/cosmoflow/boost
export CPATH=${BOOST}/:$CPATH
export LD_LIBRARY_PATH=${BOOST}/stage/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/hzheng/PolarisAT_eagle/dlio_ml_workloads/cosmoflow/utils/csrc/build/:$LD_LIBRARY_PATH
export PYTHONPATH=/home/hzheng/PolarisAT_eagle/dlio_ml_workloads/:$PYTHONPATH
