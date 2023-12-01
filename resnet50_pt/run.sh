#!/bin/bash -x
#PBS -l nodes=1:ppn=4
#PBS -l walltime=0:10:00
#PBS -q debug
#PBS -N resnet50_pt
#PBS -A datascience
#PBS -l filesystems=home:eagle

cd $PBS_O_WORKDIR
source /home/hzheng/PolarisAT/dlio_ml_workloads/setup_ml_env.sh
export LD_PRELOAD=/home/hzheng/PolarisAT_eagle/msccl/executor/msccl-executor-nccl/build/lib/libnccl.so.2
export TAG=$(date +"%Y-%m-%d-%H-%M-%S")
mkdir -p results/n1.g1.w8/$TAG/npkit_logs 
NPKIT_DUMP_DIR=results/n1.g1.w8/$TAG/npkit_logs aprun -n 1 -N 1 --cc depth -e OMP_NUM_THREADS=64 -d 64 python resnet50_hvd.py --output_folder results/n1.g1.w8/$TAG/ --profile --num_workers 8 --epochs 5 --steps 100 --custom_image_loader --shuffle
mkdir -p results/n1.g4.w8/${TAG}/npkit_logs
NPKIT_DUMP_DIR=results/n1.g4.w8/$TAG/npkit_logs aprun -n 4 -N 4 --cc depth -e OMP_NUM_THREADS=16 -d 16 python resnet50_hvd.py --output_folder results/n1.g4.w8/$TAG/ --profile --num_workers 8 --epochs 5 --steps 100 --custom_image_loader --shuffle
