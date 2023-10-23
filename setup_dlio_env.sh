#!/bin/bash
# modify this accordingly 
export VENV_HOME=$HOME/PolarisAT/pyenvs/dlio
export DLIO_PROFILER_ENABLE=1
export DLIO_PROFILER_INC_METADATA=1

if [[ -e $VENV_HOME ]]; then
    module load conda/2023-10-04
    source ${VENV_HOME}/bin/activate
else
    mkdir -p $VENV_HOME
    module load conda/2023-10-04
    conda activate
    python -m venv $VENV_HOME
    source $VENV_HOME/bin/activate
    ./install_dlio_profiler.sh
    ## Install mpi4py 
    ## Main package
    [ -e dlio/dlio_benchmark ] || git clone https://github.com/argonne-lcf/dlio_benchmark.git dlio/dlio_benchmark/
    cd dlio/dlio_benchmark/
    git pull
    CC=cc CXX=CC pip install -r requirements.txt
    python setup.py build
    python setup.py install
    cd -
fi

# install 


