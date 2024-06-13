#!/bin/bash -x
module use /soft/modulefiles
export DATE_TAG=${DATE_TAG:-"2024-04-29"}
echo $DATE_TAG
module load conda/$DATE_TAG
export WORKDIR=/home/hzheng/PolarisAT_eagle/dlio_ml_workloads/
export DLIO_PROFILER_ENABLE=1
export DLIO_PROFILER_INC_METADATA=1
export PYTHONPATH=${WORKDIR}/pfw_utils:$PYTHONPATH

# Please change the following path accordingly 
export ML_ENV=$HOME/PolarisAT_eagle/pyenvs/ml_workloads/$DATE_TAG

export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib:$LD_LIBRARY_PATH
if [[ -e $ML_ENV ]]; then
    source $ML_ENV/bin/activate
    export PYTHONPATH=$WORKDIR:$PYTHONPATH
    conda activate $ML_ENV
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
