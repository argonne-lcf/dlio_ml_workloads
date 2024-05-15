#!/bin/bash
# modify this accordingly
DATE_TAG=2024-04-29
module use /soft/modulefiles
export VENV_HOME=$HOME/PolarisAT_eagle/pyenvs/dlio/$DATE_TAG
export DLIO_PROFILER_ENABLE=1
export DLIO_PROFILER_INC_METADATA=1
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib:$LD_LIBRARY_PATH
if [[ -e $VENV_HOME ]]; then
    module load conda/$DATE_TAG
    source ${VENV_HOME}/bin/activate
else
    mkdir -p $VENV_HOME
    module load conda/$DATE_TAG
    conda activate
    python -m venv $VENV_HOME
    source $VENV_HOME/bin/activate
    ./install_dlio_profiler.sh
    ## Install mpi4py 
    ## Main package
    [ -e dlio/dlio_benchmark ] || git clone https://github.com/argonne-lcf/dlio_benchmark.git dlio/dlio_benchmark/
    cd dlio/dlio_benchmark/
    CC=cc CXX=CC MPICC=cc MPICXX=CC python setup.py install
    cd -
fi
export MPICH_GPU_SUPPORT_ENABLED=0
# install 


