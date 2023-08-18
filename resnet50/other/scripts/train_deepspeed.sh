#!/bin/bash

module load conda
conda activate

# Remove hostfile if it exists
if [ -f "hostfile" ]; then
    rm hostfile
fi

# Remove .deepspeed_env if it exists
if [ -f ".deepspeed_env" ]; then
    rm .deepspeed_env
fi

cat $PBS_NODEFILE > hostfile
# sed -e 's/$/ slots=4/' -i hostfile

echo "PATH=${PATH}" >> .deepspeed_env
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> .deepspeed_env
echo "http_proxy=${http_proxy}" >> .deepspeed_env
echo "https_proxy=${https_proxy}" >> .deepspeed_env

# deepspeed --hostfile=hostfile mnist_ds.py --deepspeed --deepspeed_config mnist_ds_config.json
mpirun --ppn 4 --hostfile=hostfile python resnet_ds.py --deepspeed --deepspeed_config resnet_ds_config.json