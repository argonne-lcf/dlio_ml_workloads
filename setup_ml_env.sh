#!/bin/bash -x
module use /soft/modulefiles
export DATE_TAG=${DATE_TAG:-"2024-04-29"}
echo $DATE_TAG
module load conda/$DATE_TAG
export WORKDIR=/home/hzheng/PolarisAT_eagle/dlio_ml_workloads/
export DLIO_PROFILER_ENABLE=1
export DLIO_PROFILER_INC_METADATA=1
export PATH=${WORKDIR}/pfw_utils:$PATH
if [ -v PBS_NODEFILE ]; then
    export PBS_JOBSIZE=$(cat $PBS_NODEFILE | sort | uniq | sed -n $=)
fi
# Please change the following path accordingly 
export ML_ENV=$HOME/PolarisAT_eagle/pyenvs/ml_workloads/$DATE_TAG
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib:$LD_LIBRARY_PATH
if [[ -e $ML_ENV ]]; then
    conda activate $ML_ENV
    export LD_LIBRARY_PATH=${ML_ENV}/lib/python3.11/site-packages/dlio_profiler/lib:${ML_ENV}/lib/python3.11/site-packages/dlio_profiler/lib64/:$LD_LIBRARY_PATH
    export PYTHONPATH=$WORKDIR:$PYTHONPATH
else
    conda create --solver libmamba -c pytorch -c nvidia -p $ML_ENV "python==3.11.8"
    #conda create  -p $ML_ENV --clone  /soft/datascience/conda/${DATE_TAG}/mconda3/
    conda activate $ML_ENV
    pip install /soft/applications/conda/$DATE_TAG/wheels/*.whl
    ./install_dlio_profiler.sh
    export PYTHONPATH=$WORKDIR/:$PYTHONPATH
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
