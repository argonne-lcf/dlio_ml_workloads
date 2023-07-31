#!/bin/bash
#PBS -S /bin/bash
#PBS -l nodes=64:ppn=4
#PBS -l walltime=4:00:00
#PBS -M huihuo.zheng@anl.gov
#PBS -A datascience
cd $PBS_O_WORKDIR
source $HOME/datascience_grand/http_proxy_polaris
source ./setup.sh
aprun -n 256 -N 4 python train.py configs/test_polaris.yaml -d --wandb --cache "RAM"
