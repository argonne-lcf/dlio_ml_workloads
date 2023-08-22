#!/bin/bash +x
module load conda
# Please change the following path accordingly 
export ML_ENV=$HOME/PolarisAT/pyenvs/ml_workload
if [[ -e $ML_ENV ]]; then
    conda activate $ML_ENV
else
    conda create  -p $ML_ENV --clone  /soft/datascience/conda/2022-09-08/mconda3/
    conda activate $ML_ENV
    pip install git+https://github.com/hariharan-devarajan/dlio-profiler.git
fi
# INSTALL OTHER MISSING FILES


