#!/bin/bash -x
export DATE_TAG=${DATE_TAG:-"2023-10-04"}
module load conda/$DATE_TAG
export RDMAV_HUGEPAGES_SAFE=1
export IBV_FORK_SAFE=1
export WORKDIR=/home/hzheng/PolarisAT/dlio_ml_workloads/
export DLIO_PROFILER_ENABLE=1
export DLIO_PROFILER_INC_METADATA=1
export PATH=${WORKDIR}/pfw_utils:$PATH
if [ -v PBS_NODEFILE ]; then
    export PBS_JOBSIZE=$(cat $PBS_NODEFILE | sort | uniq | sed -n $=)
fi
# Please change the following path accordingly 
export ML_ENV=$HOME/PolarisAT/pyenvs/ml_workloads/$DATE_TAG
if [[ -e $ML_ENV ]]; then
    conda activate $ML_ENV
    export LD_LIBRARY_PATH=${ML_ENV}/lib/python3.10/site-packages/dlio_profiler/lib:${ML_ENV}/lib/python3.10/site-packages/dlio_profiler/lib64/:$LD_LIBRARY_PATH
    export PYTHONPATH=$WORKDIR:$PYTHONPATH
else
    conda create  -p $ML_ENV --clone  /soft/datascience/conda/${DATE_TAG}/mconda3/
    conda activate $ML_ENV
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
