#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=4
#PBS -M huihuo.zheng@anl.gov
#PBS -A datascience
#PBS -q debug
#PBS -l filesystems=home:grand:eagle
cd ${PBS_O_WORKDIR}
function getrank()
{
    return ${PMI_LOCAL_RANK}
}
source ./setup.sh
export TAG=$(date +"%Y-%m-%d-%H-%M-%S")
CONFIG=test_tfr
aprun -n ${NODES} -N 4 python ./main.py +mpi.local_size=4 ++data.stage=/local/scratch/ ++data.dont_use_mmap=True \
      +log.timestamp=ms_${nodes} \
      +log.experiment_id=${PBS_JOBID} +log.folder=results/${CONFIG}/${TAG} \
      --config-name ${CONFIG}
