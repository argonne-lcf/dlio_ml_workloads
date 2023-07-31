#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=6:00:00
#PBS -l nodes=128:ppn=4
#PBS -M kaushik.v@anl.gov
#PBS -A datascience
#PBS -l filesystems=home:grand

# qsub -I -l select=2,walltime=1:00:00 -q debug-scaling -l filesystems=home:grand:eagle -A CSC250STDM10

cd $PBS_O_WORKDIR
nodes=0
for i in `get_hosts.py`
do
    nodes=$((nodes+1))
done


source ./setup.sh

source $HOME/datascience_grand/http_proxy_polaris

# python -m venv --system-site-packages ./new_env_k 
# source ./new_env_k/bin/activate 
# pip install "git+https://github.com/mlperf/logging.git"
# pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
# git clone https://github.com/NVIDIA/apex
# cd apex
# pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# pip list
# cd ..


export PYTHONPATH=./new_env_k/lib/python3.8/site-packages/:$PYTHONPATH

#aprun -n 512 -N 4 python ./main.py +mpi.local_size=4 ++data.stage=/local/scratch/ +log.timestamp=ms_${nodes} +log.experiment_id=${PBS_JOBID} --config-name test_128x4x1_tfr
#aprun -n 8 -N 4 python ./main.py +mpi.local_size=4 ++data.stage=/local/scratch/ +log.timestamp=ms_2 +log.experiment_id=${PBS_JOBID} --config-name test_128x4x1_tfr
aprun -n 8 -N 4 python ./main.py +mpi.local_size=4 ++data.stage=/local/scratch/ +log.timestamp=ms_2 +log.experiment_id=${PBS_JOBID} --config-name test_tfr



