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
aprun -n 4 -N 4 python ./main.py +mpi.local_size=4 ++data.stage=/local/scratch/ +log.timestamp=ms_${nodes} +log.experiment_id=${PBS_JOBID} --config-name test_tfr
