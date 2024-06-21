#!/bin/bash -x
# Base PYTHON environment
module use /soft/modulefiles
module load conda/$DATE_TAG

# Build
export DATE_TAG=${DATE_TAG:-"2024-04-29"}
echo $DATE_TAG

# current directory
export WORKDIR=/home/hzheng/PolarisAT_eagle/dlio_ml_workloads/

export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072

export DLIO_PROFILER_ENABLE=1
export DLIO_PROFILER_INC_METADATA=1
export PYTHONPATH=${WORKDIR}/pfw_utils:$PYTHONPATH

# Please change the following path accordingly 
export ML_ENV=$HOME/PolarisAT_eagle/pyenvs/ml_workloads/$DATE_TAG

export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib:$LD_LIBRARY_PATH

if [[ -e $ML_ENV ]]; then
    source $ML_ENV/bin/activate
    export PYTHONPATH=$WORKDIR:$PYTHONPATH
    source $ML_ENV/bin/activate
else
    conda activate 
    python -m venv $ML_ENV --system-site-packages
    source $ML_ENV/bin/activate
    ./install_dlio_profiler.sh
    export PYTHONPATH=$WORKDIR/:$PYTHONPATH
    #install apex
    git clone https://github.com/NVIDIA/apex
    cd apex
    python setup.py install
    cd -
    git clone https://github.com/NVIDIA/mlperf-common.git /tmp/mlperf-common
    cd /tmp/mlperf-common
    python setup.py install 
    cd -
    git clone https://github.com/mlperf/logging.git /tmp/mlperf-logging
    cd /tmp/mlperf-logging
    python setup.py install
    cd -
fi
# INSTALL OTHER MISSING FILES
export LD_LIBRARY_PATH=/soft/compilers/cudatoolkit/cuda-11.8.0/lib64/:$LD_LIBRARY_PATH
