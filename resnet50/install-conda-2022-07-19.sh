#!/bin/sh

module load conda/2022-07-19; conda activate

pip uninstall -y torch
pip uninstall -y horovod
pip uninstall -y dlio-benchmark
#pip uninstall -y torchvision

cd /lus/grand/projects/datascience/kaushikv/dlio/dlio_benchmark
rm -rf build/*
yes | pip install . --no-cache-dir
#pip uninstall -y torch

cd /lus/grand/projects/datascience/kaushikv/dlio/dependencies/dlio-profiler
export DLIO_LOGGER_USER=1
pip install .
#pip -v install --no-cache-dir git+https://github.com/hariharan-devarajan/dlio-profiler.git
yes | pip install --no-cache-dir horovod --force-reinstall


export LD_LIBRARY_PATH=/home/kaushikvelusamy/.local/polaris/conda/2022-07-19/lib64:$LD_LIBRARY_PATH
cd /lus/grand/projects/datascience/kaushikv/dlio/dlio_ml_workloads/resnet50
qsub -A datascience -q debug run-job-1node-159gb-data.sh

 
 