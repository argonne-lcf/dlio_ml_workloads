#!/bin/bash
#PBS -l nodes=1:ppn=4
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A datascience
#PBS -N resnet50_tf
#PBS -l filesystems=home:eagle

cd $PBS_O_WORKDIR
export WORKDIR=/home/hzheng/PolarisAT_eagle/dlio_ml_workloads
source ${WORKDIR}/setup_ml_env.sh
export TAG=$(date +"%Y-%m-%d-%H-%M-%S")
aprun -n 1 -N 1 --cc depth -e OMP_NUM_THREADS=64 -d 64 \
        python resnet50_hvd.py --output_folder results/n1.g1.w4/$TAG/  \
        --num_workers 4 --epochs 5 --steps 100 --batch-size 400
aprun -n 4 -N 4 --cc depth -e OMP_NUM_THREADS=16 -d 16 \
        python resnet50_hvd.py --output_folder results/n1.g4.w4/$TAG/  \
        --num_workers 4 --epochs 5 --steps 100 --batch-size 400

aprun -n 1 -N 1 --cc depth -e OMP_NUM_THREADS=64 -d 64 \
        python resnet50_hvd.py --output_folder results/n1.g1.w8/$TAG/  \
        --num_workers 8 --epochs 5 --steps 100 --batch-size 400
        
aprun -n 4 -N 4 --cc depth -e OMP_NUM_THREADS=16 -d 16 \
        python resnet50_hvd.py --output_folder results/n1.g4.w8/$TAG/  \
        --num_workers 8 --epochs 5 --steps 100 --batch-size 400
