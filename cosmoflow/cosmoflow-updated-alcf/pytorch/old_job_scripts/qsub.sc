#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=4:00:00
#PBS -M huihuo.zheng@anl.gov
#PBS -A datascience
cd $PBS_O_WORKDIR
export nodes=$(sed -n $= $PBS_NODEFILE)
source ./setup.sh
source $HOME/datascience_grand/http_proxy_polaris
aprun -n $((nodes*4)) -N 4 python ./main.py +mpi.local_size=4 ++data.stage=/dev/shm +log.timestamp=ms_${nodes} +log.experiment_id=${PBS_JOBID} --config-name test_128x4x1_tfr
