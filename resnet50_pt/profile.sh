#!/bin/bash
#PBS -l nodes=1:ppn=4
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A datascience
#PBS -l filesystems=home:grand:eagle

cd $PBS_O_WORKDIR
source /home/hzheng/PolarisAT/dlio_ml_workloads/setup_ml_env.sh
export TAG=$(date +"%Y-%m-%d-%H-%M-%S")
export PBS_JOBSIZE=$(cat $PBS_NODEFILE | uniq | sed -n $=)
mkdir -p profile/n${PBS_JOBSIZE}x4/$TAG/
LD_PRELOAD=/soft/perftools/mpitrace/lib/libmpitrace.so aprun -n $((PBS_JOBSIZE*4)) -N 4 --cc depth -e OMP_NUM_THREADS=16 -d 16 python resnet50_hvd.py --output_folder profile/n${PBS_JOBSIZE}x4/$TAG/ --profile --dummy --num_workers=0 --no_cuda
