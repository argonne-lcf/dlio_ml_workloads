#!/bin/bash -x
# Define the workdir, please modify accordingly
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export WORKDIR=$SCRIPT_DIR

# Get python setup
source ${WORKDIR}/platforms/conda.sh

# Base PYTHON environment
export DATE_TAG=${DATE_TAG:-"2024-04-29"}

# DLIO profiler
export DLIO_PROFILER_ENABLE=1
export DLIO_PROFILER_INC_METADATA=1
export DARSHAN_DISABLE=1 
export PYTHONPATH=${WORKDIR}:$PYTHONPATH

# Please change the following path accordingly 
export ML_ENV=${WORKDIR}/soft/pyenvs/ml_workloads/$DATE_TAG
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$WORKDIR/soft/boost/1.85.0/lib:$LD_LIBRARY_PATH
if [[ -e $ML_ENV ]]; then
    export PYTHONPATH=$WORKDIR:$PYTHONPATH
    source $ML_ENV/bin/activate
else
    conda activate 
    python -m venv $ML_ENV --system-site-packages
    source $ML_ENV/bin/activate
    pip install --upgrade pip
    #install apex
    git clone https://github.com/NVIDIA/apex /tmp/$USER/apex
    cd /tmp/$USER/apex
    python setup.py install
    cd -
    git clone https://github.com/NVIDIA/mlperf-common.git /tmp/$USER/mlperf-common
    cd /tmp/$USER/mlperf-common
    python setup.py install 
    cd -
    git clone https://github.com/mlperf/logging.git /tmp/$USER/mlperf-logging
    cd /tmp/$USER/mlperf-logging
    python setup.py install
    cd -
    python -m pip install -r ./unet3d/requirements.txt
    python -m pip install -r ./cosmoflow/requirements.txt


    # install other dependencies
    cd soft/
    ./install_dlio_profiler.sh
    ./install_boost.sh
    ./install_libaio.sh
    cd -

    # install cosmoflow dependencies
    cd ./cosmoflow
    sh build_libCosmoflowExt.sh
    cd -
    
    rm -rf /tmp/$USER/
fi
# INSTALL OTHER MISSING FILES
export LD_LIBRARY_PATH=/soft/compilers/cudatoolkit/cuda-11.8.0/lib64/:$LD_LIBRARY_PATH

#NCCL related libarary
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072

