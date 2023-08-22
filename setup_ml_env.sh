#!/bin/bash +x
module load conda
export ML_ENV=$HOME/PolarisAT/pyenvs/ml_workload
conda create  -p $ML_ENV --clone  /soft/datascience/conda/2022-09-08/mconda3/
pip install git+https://github.com/hariharan-devarajan/dlio-profiler.git
conda activate $ML_ENV



