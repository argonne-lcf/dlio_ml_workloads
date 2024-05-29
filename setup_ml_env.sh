#!/bin/bash -x
module use /home/hzheng/PolarisAT/soft/
export DATE_TAG="2024-04-29"
export DATE_TAG=${DATE_TAG:-"2024-04-29"}
module load conda/$DATE_TAG
export WORKDIR=/home/hzheng/PolarisAT/dlio_ml_workloads/
export DLIO_PROFILER_ENABLE=1
export DLIO_PROFILER_INC_METADATA=1
export PATH=${WORKDIR}/pfw_utils:$PATH
if [ -v PBS_NODEFILE ]; then
    export PBS_JOBSIZE=$(cat $PBS_NODEFILE | sort | uniq | sed -n $=)
fi
# Please change the following path accordingly 
export ML_ENV=$HOME/PolarisAT/pyenvs/ml_workloads/$DATE_TAG
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib:$LD_LIBRARY_PATH
export CPATH=/soft/libraries/hwloc/include:$CPATH
if [[ -e $ML_ENV ]]; then
    source $ML_ENV/bin/activate
    export LD_LIBRARY_PATH=${ML_ENV}/lib/python3.11/site-packages/dlio_profiler/lib:${ML_ENV}/lib/python3.11/site-packages/dlio_profiler/lib64/:$LD_LIBRARY_PATH
    export PYTHONPATH=$WORKDIR:$PYTHONPATH
else
    conda activate
    read a v <<<$(python --version)
    echo "Creating Conda environment with python == $v"
    python -m venv $ML_ENV --system-site-packages
    source $ML_ENV/bin/activate
    #conda create  -p $ML_ENV --system-site-packages "python==$v"
    #conda activate $ML_ENV
    #pip install /soft/applications/conda/${DATE_TAG}/wheels/*.whl
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
