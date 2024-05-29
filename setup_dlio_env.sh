#!/bin/bash
# modify this accordingly
DATE_TAG=2024-04-29
export VENV_HOME=$HOME/PolarisAT/pyenvs/dlio/$DATE_TAG
export DLIO_PROFILER_ENABLE=1
export DLIO_PROFILER_INC_METADATA=1
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/soft/compilers/cudatoolkit/12.4.1/
export LD_LIBRARY_PATH=${CUDA_HOME}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/soft/compilers/cudatoolkit/cuda-11.8.0/lib64/:$LD_LIBRARY_PATH
if [[ -e $VENV_HOME/bin/activate ]]; then
    module use /soft/modulefiles/
    module load conda/$DATE_TAG
    source ${VENV_HOME}/bin/activate
else
    mkdir -p $VENV_HOME
    module load conda/$DATE_TAG
    conda activate
    python -m venv $VENV_HOME --system-site-packages
    source $VENV_HOME/bin/activate
    ./install_dlio_profiler.sh
    ## Install mpi4py 
    ## Main package
    [ -e dlio/dlio_benchmark ] || git clone https://github.com/argonne-lcf/dlio_benchmark.git dlio/dlio_benchmark/
    cd dlio/dlio_benchmark/
    #CC=cc CXX=CC pip install -r requirements.txt
    python setup.py build
    python setup.py install
    cd -
fi
# install 


