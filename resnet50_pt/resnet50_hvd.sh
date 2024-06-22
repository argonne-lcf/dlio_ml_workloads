#!/bin/bash
#PBS -l nodes=2:ncpus=4
#PBS -l walltime=1:00:00
#PBS -q debug-scaling
#PBS -A datascience
#PBS -l filesystems=home:grand:eagle

cd $PBS_O_WORKDIR
source /home/hzheng/PolarisAT/dlio_ml_workloads/setup_ml_env.sh
export TAG=$(date +"%Y-%m-%d-%H-%M-%S")
#aprun -n 1 -N 1 --cc depth -e OMP_NUM_THREADS=64 -d 64 python resnet50_hvd.py --output_folder results/$TAG/n1.g1 --profile
#aprun -n 4 -N 4 --cc depth -e OMP_NUM_THREADS=16 -d 16 python resnet50_hvd.py --output_folder results/$TAG/n1.g4 --profile
aprun -n 8 -N 4 --cc depth -e OMP_NUM_THREADS=16 -d 16 python resnet50_hvd.py --output_folder results/$TAG/n2.g4 --multiprocessing_context='spawn'
export TAG=$(date +"%Y-%m-%d-%H-%M-%S")
aprun -n 8 -N 4 --cc depth -e OMP_NUM_THREADS=16 -d 16 python resnet50_hvd.py --output_folder results/$TAG/n2.g4 --multiprocessing_context='fork'
