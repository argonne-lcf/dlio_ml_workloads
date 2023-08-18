#!/bin/sh

module load conda/2023-01-10-unstable
conda activate

# pip uninstall -y torch
# pip uninstall -y horovod
# pip uninstall -y dlio-benchmark

# cd /lus/grand/projects/datascience/kaushikv/dlio/dlio_benchmark
# rm -rf build/*
# yes | pip install . --no-cache-dir

#pip uninstall -y torch
# yes | pip install --no-cache-dir horovod --force-reinstall


# cd /lus/grand/projects/datascience/kaushikv/dlio/dependencies/dlio-profiler
# export DLIO_LOGGER_USER=1
# pip install .
# pip -v install --no-cache-dir git+https://github.com/hariharan-devarajan/dlio-profiler.git

export LD_LIBRARY_PATH=/home/kaushikvelusamy/.local/polaris/conda/2023-01-10-unstable/lib/python3.10/site-packages/:$LD_LIBRARY_PATH
cd /lus/grand/projects/datascience/kaushikv/dlio/dlio_ml_workloads/resnet50
qsub -A datascience -q debug run-job-1node-159gb-data.sh



#export LD_LIBRARY_PATH=/home/kaushikvelusamy/.local/polaris/conda/2023-01-10-unstable/lib/python3.10/site-packages/:/soft/compilers/cudatoolkit/cuda-11.8.0/extras/CUPTI/lib64:/soft/compilers/cudatoolkit/cuda-11.8.0/lib64:/soft/libraries/trt/TensorRT-8.5.2.2.Linux.x86_64-gnu.cuda-11.8.cudnn8.6/lib:/soft/libraries/nccl/nccl_2.16.2-1+cuda11.8_x86_64/lib:/soft/libraries/cudnn/cudnn-11-linux-x64-v8.6.0.163/lib:/opt/cray/pe/gcc/11.2.0/snos/lib64:/opt/cray/pe/papi/6.0.0.14/lib64:/opt/cray/libfabric/1.11.0.4.125/lib64:/dbhome/db2cat/sqllib/lib64:/dbhome/db2cat/sqllib/lib64/gskit:/dbhome/db2cat/sqllib/lib32