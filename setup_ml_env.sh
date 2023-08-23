#!/bin/bash +x
module load conda
export WORKDIR=/home/hzheng/PolarisAT/dlio_ml_workloads/
# Please change the following path accordingly 
export ML_ENV=$HOME/PolarisAT/pyenvs/ml_workload
if [[ -e $ML_ENV ]]; then
    conda activate $ML_ENV
    export PYTHONPATH=$WORKDIR:$PYTHONPATH
else
    conda create  -p $ML_ENV --clone  /soft/datascience/conda/2022-09-08/mconda3/
    conda activate $ML_ENV
    pip install git+https://github.com/hariharan-devarajan/dlio-profiler.git
    export PYTHONPATH=$WORKDIR/:$PYTHONPATH
    #install apex
    git clone https://github.com/NVIDIA/apex
    cd apex
    python setup.py install
    cd -
fi
# INSTALL OTHER MISSING FILES


