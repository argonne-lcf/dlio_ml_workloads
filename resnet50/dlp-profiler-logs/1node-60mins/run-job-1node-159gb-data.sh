#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=00:60:00
#PBS -l nodes=1:ppn=4
#PBS -A datascience
#PBS -l filesystems=grand
#PBS -q debug
#qsub -A datascience -q debug run-job-1node-159gb-data.sh

# qsub -I -l select=1,walltime=00:60:00 -q debug -l filesystems=home:grand -A datascience
#echo "Using Horovod"

module load conda/2022-07-19; conda activate
echo "load env complete"

export LD_LIBRARY_PATH=/home/kaushikvelusamy/.local/polaris/conda/2022-07-19/lib64:$LD_LIBRARY_PATH
cd /lus/grand/projects/datascience/kaushikv/dlio/dlio_ml_workloads/resnet50
aprun -n 4 -N 4 python resnet_hvd.py --batch-size 64 --epochs 1 --save_model 1 
# dataset location is /eagle/datascience/ImageNet/ILSVRC/Data/CLS-LOC/
