#!/bin/bash

module load conda
conda activate base
echo "load env complete"

cd /eagle/datascience/zhaozhenghao/workspace/methods/resnet50/

echo "Using DDP"
aprun -n 16 -N 4 python resnet_ddp.py --batch-size 64 --steps 11