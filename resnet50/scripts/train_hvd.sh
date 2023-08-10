#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=00:10:00
#PBS -l nodes=1:ppn=4
#PBS -A datascience
#PBS -l filesystems=grand
#PBS -q debug
#qsub -A datascience -q debug train_hvd.sh

# qsub -I -l select=2,walltime=00:20:00 -q debug -l filesystems=home:grand -A datascience

module load conda/2022-07-19; conda activate
echo "load env complete"

export LD_LIBRARY_PATH=/home/kaushikvelusamy/.local/polaris/conda/2022-07-19/lib64:$LD_LIBRARY_PATH


#cd /eagle/datascience/zhaozhenghao/workspace/methods/resnet50/
cd /lus/grand/projects/datascience/kaushikv/dlio/dlio_ml_workloads/resnet50
echo "Using Horovod"
aprun -n 4 -N 4 python resnet_hvd.py --batch-size 64 --steps 11 --print-freq 1 
# aprun -n 4 -N 4 python resnet_hvd.py --batch-size 64 --steps 11 --print-freq 1 &
# FOO_PID=$!
# sleep 15
# kill -6 ${FOO_PID}

# dataset location is /eagle/datascience/ImageNet/ILSVRC/Data/CLS-LOC/





