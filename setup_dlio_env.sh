#!/bin/bash
# modify this accordingly 
export VENV_HOME=$HOME/PolarisAT/pyenvs/dlio
if [[ -e $VENV_HOME ]]; then
    source ${VENV_HOME}/bin/activate
else
    mkdir -p $VENV_HOME
    module load conda
    conda activate
    python -m venv $VENV_HOME
    source $VENV_HOME/bin/activate
    pip install git+https://github.com/hariharan-devarajan/dlio-profiler.git
    # Install DLIO
    ## Install mpi4py 
    CC=cc CXX=CC pip install mpi4py
    ## Main package
    git clone https://github.com/argonne-lcf/dlio_benchmark.git
    cd dlio_benchmark
    python setup.py build
    python setup.py install
    cd -
fi

# install 


