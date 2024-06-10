#!/bin/sh
source /home/hzheng/PolarisAT_eagle/dlio_ml_workloads/setup_ml_env.sh
BOOST_DIR=/eagle/DLIO/soft/boost-1.85.0/
export CPATH=${BOOST_DIR}/:$CPATH
export LD_LIBRARY_PATH=${BOOST_DIR}/stage/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/hzheng/PolarisAT_eagle/dlio_ml_workloads/cosmoflow/utils/csrc/build/:$LD_LIBRARY_PATH
export PYTHONPATH=/home/hzheng/PolarisAT_eagle/dlio_ml_workloads/:$PYTHONPATH
