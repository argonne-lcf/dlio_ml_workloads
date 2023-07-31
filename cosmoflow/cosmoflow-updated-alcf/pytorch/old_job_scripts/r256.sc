#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=4:00:00
#PBS -l nodes=256:ppn=4
#PBS -M huihuo.zheng@anl.gov
#PBS -A datascience
#PBS -l filesystems=home:grand
cd $PBS_O_WORKDIR
nodes=0
for i in `get_hosts.py`
do
    nodes=$((nodes+1))
done
source ./setup.sh
source $HOME/datascience_grand/http_proxy_polaris
aprun -n 1024 -N 4 python ./main.py +mpi.local_size=4 ++data.stage=/local/scratch/ +log.timestamp=ms_${nodes} +log.experiment_id=${PBS_JOBID} --config-name test_256x4x1_tfr

