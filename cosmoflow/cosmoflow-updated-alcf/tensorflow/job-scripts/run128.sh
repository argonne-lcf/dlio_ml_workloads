#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=128:ppn=4
#PBS -l walltime=4:00:00
#PBS -M huihuo.zheng@anl.gov
#PBS -A datascience
#PBS -q R329190
cd $PBS_O_WORKDIR
source $HOME/datascience_grand/http_proxy_polaris
source ./setup.sh
aprun -n 512 -N 4 python train.py configs/test_polaris.yaml -d --wandb --mlperf --cache RAM
