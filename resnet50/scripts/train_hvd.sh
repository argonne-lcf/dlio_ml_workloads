#!/bin/bash
# qsub -I -l select=2,walltime=1:00:00 -q debug-scaling -l filesystems=home:grand:eagle -A CSC250STDM10

module load conda
conda activate base
echo "load env complete"

#cd /eagle/datascience/zhaozhenghao/workspace/methods/resnet50/
cd /lus/grand/projects/datascience/kaushikv/dlio/ml_workloads/resnet50/dl_scaling/resnet50/

echo "Using Horovod"
aprun -n 8 -N 4 python resnet_hvd.py --batch-size 64 --steps 11 --print-freq 1

# dataset location is /eagle/datascience/ImageNet/ILSVRC/Data/CLS-LOC/





