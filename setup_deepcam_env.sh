#!/bin/bash +x
module load conda/2023-10-04
export RDMAV_HUGEPAGES_SAFE=1
export IBV_FORK_SAFE=1
export WORKDIR=/home/hzheng/PolarisAT/dlio_ml_workloads/
if [ -v PBS_NODEFILE ]; then
    export PBS_JOBSIZE=$(cat $PBS_NODEFILE | sort | uniq | sed -n $=)
fi
# Please change the following path accordingly 
export ML_ENV=$HOME/PolarisAT/pyenvs/deepcam
if [[ -e $ML_ENV ]]; then
    conda activate $ML_ENV
    export PYTHONPATH=$WORKDIR:$PYTHONPATH
else
    conda create  -p $ML_ENV --clone  /soft/datascience/conda/2023-10-04/mconda3/
    conda activate $ML_ENV
    pip install git+https://github.com/hariharan-devarajan/dlio-profiler.git
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


