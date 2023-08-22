#!/bin/bash
# modify this accordingly 
export VENV_HOME=$HOME/PolarisAT/pyenvs/dlio 
mkdir -p $VENT_HOME
module load conda
conda activate
python -m venv $VENT_HOME
source $VENT_HOME/bin/activate
pip install git+https://github.com/hariharan-devarajan/dlio-profiler.git
git clone https://github.com/argonne-lcf/dlio_benchmark.git
cd dlio_benchmark
python setup.py build
python setup.py install
cd -



