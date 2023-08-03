#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=6:00:00
#PBS -l nodes=128:ppn=4
#PBS -M kaushik.v@anl.gov
#PBS -A datascience
#PBS -l filesystems=home:grand


# Download Dataset directory :  /lus/grand/projects/datascience/MlPerf-datasets/cosmoflow/cosmoUniverse_2019_05_4parE_tf_v2/train
# original code repo : /grand/datascience/hzheng/mlperf-2022/optimized-hpc/cosmoflow/
# root_dir for config : "/grand/datascience/MlPerf-datasets/cosmoflow/cosmoUniverse_2019_05_4parE_tf_v2"
# root_dir: "/home/hzheng/datascience_grand/mlperf_hpc/hpc-nvidia/datasets/cosmoflow/tf_v2_256"


# cd $PBS_O_WORKDIR
# nodes=0
# for i in `get_hosts.py`
# do
#     nodes=$((nodes+1))
# done


# source $HOME/datascience_grand/http_proxy_polaris

# To setup the conda and python environments 
 

module load conda/2022-07-19; conda activate
export TMPDIR=/lus/eagle/projects/PolarisAT/kaushikv/dlio_ml_workloads/cosmoflow/cosmoflow-updated-alcf/pytorch/logs/results
export MPICH_GPU_SUPPORT_ENABLED=0
export CPATH=/lus/eagle/projects/PolarisAT/kaushikv/dlio_ml_workloads/cosmoflow/cosmoflow-updated-alcf/pytorch/dependencies/boost_1_82/boost_install_dir/:$CPATH
export LD_LIBRARY_PATH=/lus/eagle/projects/PolarisAT/kaushikv/dlio_ml_workloads/cosmoflow/cosmoflow-updated-alcf/pytorch/dependencies/boost_1_82/boost_install_dir/lib/:$LD_LIBRARY_PATH
 

cd /lus/eagle/projects/PolarisAT/kaushikv/dlio_ml_workloads/cosmoflow/cosmoflow-updated-alcf/pytorch
# python -m venv /lus/eagle/projects/PolarisAT/kaushikv/dlio_ml_workloads/cosmoflow/cosmoflow-updated-alcf/pytorch/dependencies/new-cos-py-env
source /lus/eagle/projects/PolarisAT/kaushikv/dlio_ml_workloads/cosmoflow/cosmoflow-updated-alcf/pytorch/dependencies/new-cos-py-env/bin/activate 
export PYTHONPATH=/lus/eagle/projects/PolarisAT/kaushikv/dlio_ml_workloads/cosmoflow/cosmoflow-updated-alcf/pytorch/dependencies/new-cos-py-env/lib/python3.8/site-packages/:$PYTHONPATH


# cd /lus/eagle/projects/PolarisAT/kaushikv/dlio_ml_workloads/cosmoflow/cosmoflow-updated-alcf/pytorch/dependencies

# Dependencies
# pip install --upgrade pip
# pip install "git+https://github.com/mlperf/logging.git"
# pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
# pip install packaging 
# pip install torch

# Dependency 3
# git clone https://github.com/NVIDIA/apex
# cd /lus/eagle/projects/PolarisAT/kaushikv/dlio_ml_workloads/cosmoflow/cosmoflow-updated-alcf/pytorch/dependencies/apex
# pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# cd ..

# Dependency 4
# module load PrgEnv-gnu
# export CC=`which gcc`    
# export CXX=`which g++`  
# python -m pip install --no-cache-dir git+https://github.com/hariharan-devarajan/dlio-profiler.git
# test : >>> from dlio_profiler.logger import dlio_logger


# Dependency 5 
# cd /lus/eagle/projects/PolarisAT/kaushikv/dlio_ml_workloads/cosmoflow/cosmoflow-updated-alcf/pytorch/dependencies/boost_1_82
# install boost libarary and add the path to LD_LIBRARY_PATH and CPATH in setup.sh 



# To Run from batch script

#aprun -n 512 -N 4 python ./main.py +mpi.local_size=4 ++data.stage=/local/scratch/ +log.timestamp=ms_${nodes} +log.experiment_id=${PBS_JOBID} --config-name test_128x4x1_tfr
#aprun -n 8 -N 4 python ./main.py +mpi.local_size=4 ++data.stage=/local/scratch/ +log.timestamp=ms_2 +log.experiment_id=${PBS_JOBID} --config-name test_128x4x1_tfr
#aprun -n 4 -N 4 python ./main.py +mpi.local_size=4 ++data.stage=/local/scratch/ +log.timestamp=ms_2 +log.experiment_id=${PBS_JOBID} --config-name test_tfr


#From interactive compute node 

qsub -I -l select=1,walltime=00:20:00 -q debug -l filesystems=eagle -A datascience

export LOGDIR=/lus/eagle/projects/PolarisAT/kaushikv/dlio_ml_workloads/cosmoflow/cosmoflow-updated-alcf/pytorch/logs/results
aprun -n 4 -N 4 python ./main.py +mpi.local_size=4 ++data.stage=/local/scratch/ +log.timestamp=ms_2 +log.experiment_id=${PBS_JOBID} --config-name test_tfr --output-dir /lus/eagle/projects/PolarisAT/kaushikv/dlio_ml_workloads/cosmoflow/cosmoflow-updated-alcf/pytorch/logs/outputs


deactivate
conda deactivate


# cosflow code update
#  def epoch_step(self, ------ value={"throughput": self._config["data"]["batch_size"] * self._config["data"]["num_nodes"] * 4 /
