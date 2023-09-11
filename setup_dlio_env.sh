#!/bin/bash
# modify this accordingly 
export VENV_HOME=$HOME/PolarisAT/pyenvs/dlio
if [ -z "${PBS_NODEFILE}" ]; then
    export PBS_JOBSIZE=$(cat $PBS_NODEFILE | sort | uniq | sed -n $=)
fi
if [[ -e $VENV_HOME ]]; then
    module load conda
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
    git clone -b 3.1.4 https://github.com/mpi4py/mpi4py.git
    cd mpi4py
    CC=cc CXX=CC python setup.py build
    CC=cc CXX=CC python	setup.py install
    ## Main package
    git clone https://github.com/argonne-lcf/dlio_benchmark.git
    cd dlio_benchmark
    python setup.py build
    python setup.py install
    cd -
fi

# install 


