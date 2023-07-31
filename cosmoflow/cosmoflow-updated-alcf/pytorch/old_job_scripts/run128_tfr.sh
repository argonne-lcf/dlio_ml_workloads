#!/bin/sh
function getrank()
{
    return ${PMI_LOCAL_RANK}
}
source ./setup.sh
aprun -n 4 -N 4 python ./main.py +mpi.local_size=4 ++data.stage=/dev/shm +log.timestamp=ms +log.experiment_id=10 --config-name test_128x4x1_tfr
#source ./setup.sh
#export n=$1
#export ntrain=$((524288/$n))
#export nval=$((65536/$n))
#echo $ntrain $nval
#aprun -n 4 -N 4 python ./main.py +mpi.local_size=4 ++data.train_samples=$ntrain ++data.valid_samples=$nval --config-name test_dummy 
