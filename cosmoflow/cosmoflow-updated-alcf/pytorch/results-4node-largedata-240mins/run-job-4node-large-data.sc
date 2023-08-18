#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=04:00:00
#PBS -l nodes=4:ppn=4
#PBS -A datascience
#PBS -l filesystems=grand
#PBS -q prod
#qsub -A datascience -q prod run-job-4node-large-data.sc


# Download Dataset directory : /lus/grand/projects/datascience/MlPerf-datasets/cosmoflow/cosmoUniverse_2019_05_4parE_tf_v2/
#                            : /grand/datascience/hzheng/mlperf_hpc/hpc/cosmoflow/data/c64s16m/cosmoUniverse_2019_05_4parE_tf_v2/
#                            : /grand/datascience/MlPerf-datasets/cosmoflow/cosmoUniverse_2019_05_4parE_tf_v2

# original code repo         : /grand/datascience/hzheng/mlperf-2022/optimized-hpc/cosmoflow/

# cd $PBS_O_WORKDIR
# nodes=0
# for i in `get_hosts.py`
# do
#     nodes=$((nodes+1))
# done

#source $HOME/datascience_grand/http_proxy_polaris

#export CC=/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1/bin/mpicc  
#export CXX=/opt/cray/pe/mpich/8.1.16/ofi/gnu/9.1/bin/mpic++

module load PrgEnv-cray
module load conda/2022-07-19; conda activate
# pip -v install --no-cache-dir git+https://github.com/hariharan-devarajan/dlio-profiler.git
export MPICH_GPU_SUPPORT_ENABLED=0

export CPATH=/lus/grand/projects/datascience/kaushikv/dlio/dependencies/boost_lib:$CPATH
export LD_LIBRARY_PATH=/lus/grand/projects/datascience/kaushikv/dlio/dependencies/boost_lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/home/kaushikvelusamy/.local/polaris/conda/2022-07-19/lib64:$LD_LIBRARY_PATH

export LOGDIR=/lus/grand/projects/datascience/kaushikv/dlio/dlio_ml_workloads/cosmoflow/cosmoflow-updated-alcf/pytorch/logs

echo $LD_LIBRARY_PATH
echo $PYTHONPATH
echo $CPATH
echo $CC
echo $CXX
which cc
which pip
which python
python --version

cd /lus/grand/projects/datascience/kaushikv/dlio/dlio_ml_workloads/cosmoflow/cosmoflow-updated-alcf/pytorch
aprun -n 16 -N 4 python ./main.py +mpi.local_size=4 ++data.stage=/local/scratch/ +log.timestamp=ms_4node_240mins +log.experiment_id=${PBS_JOBID} --config-name test_tfr 
